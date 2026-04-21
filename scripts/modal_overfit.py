"""Run ``scripts/overfit_processed_chain.py`` on a Modal Labs GPU.

One-time setup (once per machine)::

    pip install modal
    modal setup                       # browser-based auth

Launch a run (the defaults match the full paper-spec alphafold2 profile)::

    modal run scripts/modal_overfit.py --chain-id 6m0j_E

All the flags from the local script are exposed as Modal CLI options with
the same dashed names — e.g. ``--no-freeze-crop-and-cluster``, ``--steps
5000``, ``--model-profile medium``. The Modal function streams stdout
live, and every artifact the overfit script writes (predicted PDB,
ground-truth PDB, PyMOL view script, ``metrics.json``, per-step
``losses.json``) lands in the ``minalphafold-artifacts`` Modal Volume so
it survives after the container exits. Fetch them back with::

    modal volume get minalphafold-artifacts overfit_processed_chain ./local/

The image bakes in only what the training loop actually needs — the
``minalphafold/`` package, ``configs/``, ``scripts/``, and the *preprocessed*
NPZ caches under ``data/processed_{features,labels}/``. Raw OpenProteinSet
MSAs and mmCIFs stay local, so the image is small. Run
``scripts/download_openproteinset.py`` and
``scripts/preprocess_openproteinset.py`` locally before the first remote run.
"""

from __future__ import annotations

from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]

app = modal.App("minalphafold-overfit")

# Container image — Debian + Python 3.11 + CUDA-enabled torch. Recent
# pip-installed torch wheels include CUDA runtime, so we don't need a
# custom base image for GPU support.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.3", "numpy")
    .add_local_dir(str(REPO_ROOT / "minalphafold"), remote_path="/root/minalphafold")
    .add_local_dir(str(REPO_ROOT / "configs"), remote_path="/root/configs")
    .add_local_dir(str(REPO_ROOT / "scripts"), remote_path="/root/scripts")
    .add_local_dir(
        str(REPO_ROOT / "data" / "processed_features"),
        remote_path="/root/data/processed_features",
    )
    .add_local_dir(
        str(REPO_ROOT / "data" / "processed_labels"),
        remote_path="/root/data/processed_labels",
    )
)

# Persistent output volume so artifacts survive container exit and can be
# pulled back with ``modal volume get minalphafold-artifacts ...``.
artifacts = modal.Volume.from_name("minalphafold-artifacts", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",  # 40GB OOMs on crop=256 + 48 Evoformer blocks (triangle attn stores (B,H,N,N,N) across all blocks for backward); 80GB has headroom
    volumes={"/root/artifacts": artifacts},
    timeout=60 * 60 * 12,  # up to 12 hours — 10k steps of alphafold2 fits comfortably
)
def run_overfit(argv: list[str]) -> None:
    """Invoke ``scripts/overfit_processed_chain.main`` with ``argv`` on a GPU.

    The overfit script's own argparse handles every option; this wrapper just
    forwards ``argv`` and commits the output volume once training finishes.
    """
    import os
    import sys

    # The overfit script expects CWD to be the repo root so relative paths
    # resolve to ``/root/data/processed_*`` and writes land under
    # ``/root/artifacts/`` (the mounted Modal Volume).
    os.chdir("/root")
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from overfit_processed_chain import main as overfit_main

    overfit_main(argv)
    artifacts.commit()


@app.local_entrypoint()
def main(
    chain_id: str = "6m0j_E",
    model_profile: str = "alphafold2",
    steps: int = 10000,
    log_every: int = 10,
    eval_every: int = 25,
    crop_size: int = 256,
    msa_depth: int = 128,
    extra_msa_depth: int = 1024,
    max_templates: int = 4,
    freeze_crop_and_cluster: bool = True,
    use_clamped_fape: float | None = None,
    violations_after_step: int | None = None,
    fine_tune_lr_scale: float = 0.5,
    violation_ramp_steps: int = 500,
    unclamp_fape_on_finetune: bool = False,
    seed: int = 0,
) -> None:
    """Translate these Modal-CLI kwargs into the overfit script's argv and ship it."""
    argv = [
        "--chain-id", chain_id,
        "--model-profile", model_profile,
        "--steps", str(steps),
        "--log-every", str(log_every),
        "--eval-every", str(eval_every),
        "--crop-size", str(crop_size),
        "--msa-depth", str(msa_depth),
        "--extra-msa-depth", str(extra_msa_depth),
        "--max-templates", str(max_templates),
        "--seed", str(seed),
        "--out-dir", f"/root/artifacts/overfit_processed_chain/{chain_id}",
    ]
    if freeze_crop_and_cluster:
        argv.append("--freeze-crop-and-cluster")
    if use_clamped_fape is not None:
        argv += ["--use-clamped-fape", str(use_clamped_fape)]
    if violations_after_step is not None:
        argv += [
            "--violations-after-step", str(violations_after_step),
            "--fine-tune-lr-scale", str(fine_tune_lr_scale),
            "--violation-ramp-steps", str(violation_ramp_steps),
        ]
        if unclamp_fape_on_finetune:
            argv.append("--unclamp-fape-on-finetune")

    print(f"[modal] remote argv:\n  {' '.join(argv)}")
    run_overfit.remote(argv)
