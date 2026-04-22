"""Run ``scripts/train_af2.py`` on a Modal Labs GPU.

Pipeline for a full paper-spec run:

1. **One-time**: upload the preprocessed features/labels to a Modal
   Volume so multiple training runs can share them without re-baking
   the image. ::

       modal volume put minalphafold-data ./data/processed_features /processed_features
       modal volume put minalphafold-data ./data/processed_labels   /processed_labels

2. **Initial stage** (supplement §1.11.1 stage 1, Table 4). Run
   repeatedly until the target sample count lands; each invocation
   auto-resumes from the previous container's checkpoint. ::

       modal run scripts/modal_train_af2.py --stage initial

3. **Fine-tune stage** (stage 2, Table 4). Seed from the initial
   checkpoint and run until its target samples land. ::

       modal run scripts/modal_train_af2.py --stage finetune \\
         --init-from-path /checkpoints/initial_latest.pt

Everything — initial_latest.pt, initial_best.pt, finetune_latest.pt,
finetune_best.pt, EMA shadow state, optimizer state, counters — lives in
the ``minalphafold-checkpoints`` Volume. Pull artifacts back locally
with ``modal volume get minalphafold-checkpoints ./checkpoints``.

``MODAL_GPU_TYPE=H200`` pins a single GPU class for cost or benchmark
pinning; otherwise Modal picks from the fallback list (H200 first,
A100-80GB second) — the H200's 141 GB is what makes the fine-tune-stage
384-residue crop comfortable.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]

# See modal_overfit.py for the GPU-fallback rationale. The fine-tune
# stage's 384-residue crop + 512 MSA sequences is the memory-hungriest
# workload in the repo.
_DEFAULT_GPU_FALLBACK: list[str] = ["H200", "A100-80GB"]
_GPU_OVERRIDE = os.environ.get("MODAL_GPU_TYPE")
_GPU_SPEC = _GPU_OVERRIDE if _GPU_OVERRIDE else _DEFAULT_GPU_FALLBACK

app = modal.App("minalphafold-train-af2")

# Image: repo code only. Data comes from a separately-uploaded Volume
# so the image stays small (full AF2 training data is ~100 GB of
# preprocessed NPZ caches — far too much to bake in).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.3", "numpy")
    .add_local_dir(str(REPO_ROOT / "minalphafold"), remote_path="/root/minalphafold")
    .add_local_dir(str(REPO_ROOT / "configs"), remote_path="/root/configs")
    .add_local_dir(str(REPO_ROOT / "scripts"), remote_path="/root/scripts")
)

# Data and checkpoints live in separate Volumes so training runs can be
# restarted without re-uploading the 100 GB of preprocessed features.
data_volume = modal.Volume.from_name("minalphafold-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("minalphafold-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu=_GPU_SPEC,
    volumes={
        "/root/data": data_volume,
        "/root/checkpoints": checkpoints_volume,
    },
    # 24 h = Modal's default function-timeout ceiling. Full paper-spec
    # initial training needs multiple such runs back-to-back; ``--stage
    # initial`` auto-resumes from ``checkpoints/initial_latest.pt`` on
    # subsequent invocations.
    timeout=60 * 60 * 24,
)
def run_train(argv: list[str], auto_resume_stage: str | None = None) -> None:
    """Execute ``train_af2.main(argv)`` inside the container.

    ``auto_resume_stage`` is the remote-side auto-resume hook: if set,
    we look up ``/root/checkpoints/<stage>_latest.pt`` and, if it
    exists, append ``--resume <path>`` to ``argv``. This keeps the
    "first run starts fresh, subsequent runs auto-pick-up" ergonomic
    without the local side having to guess at remote filesystem state.
    """
    import sys

    os.chdir("/root")
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")

    if auto_resume_stage is not None and "--resume" not in argv:
        candidate = Path(f"/root/checkpoints/{auto_resume_stage}_latest.pt")
        if candidate.exists():
            print(f"[auto-resume] found {candidate}, appending --resume")
            argv = [*argv, "--resume", str(candidate)]
        else:
            print(f"[auto-resume] {candidate} not present — starting fresh.")

    from train_af2 import main as train_main

    train_main(argv)
    # Commit the checkpoint volume so the latest state is visible to the
    # next run (auto-resume) and to ``modal volume get`` locally.
    checkpoints_volume.commit()


@app.local_entrypoint()
def main(
    stage: str = "initial",
    model_config: str = "alphafold2",
    training_protocol: str = "alphafold2",
    checkpoint_dir: str = "/root/checkpoints",
    processed_features_dir: str = "/root/data/processed_features",
    processed_labels_dir: str = "/root/data/processed_labels",
    val_fraction: float = 0.0,
    batch_size: int = 1,
    grad_accum_steps: int | None = None,
    num_workers: int = 4,
    seed: int = 0,
    n_cycles: int = 4,
    n_ensemble: int = 1,
    epochs: int | None = None,
    init_from_path: str | None = None,
    resume_path: str | None = None,
    auto_resume: bool = True,
) -> None:
    """Translate Modal-CLI kwargs into ``train_af2.py`` argv and launch.

    ``auto_resume`` is the important ergonomic bit: if set (default) and
    ``resume_path`` is unset, the driver auto-picks up
    ``<checkpoint_dir>/<stage>_latest.pt`` so the user can re-issue the
    same ``modal run`` command as many times as it takes to burn through
    the stage's sample budget — no manual ``--resume`` juggling.
    """
    if stage == "finetune" and init_from_path is None and resume_path is None and not auto_resume:
        raise SystemExit(
            "finetune requires either --init-from-path or --resume-path (or --auto-resume)"
        )

    argv: list[str] = [
        "--stage", stage,
        "--checkpoint-dir", checkpoint_dir,
        "--model-config", model_config,
        "--training-protocol", training_protocol,
        "--processed-features-dir", processed_features_dir,
        "--processed-labels-dir", processed_labels_dir,
        "--val-fraction", str(val_fraction),
        "--device", "cuda",
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--seed", str(seed),
        "--n-cycles", str(n_cycles),
        "--n-ensemble", str(n_ensemble),
    ]
    if grad_accum_steps is not None:
        argv += ["--grad-accum-steps", str(grad_accum_steps)]
    if epochs is not None:
        argv += ["--epochs", str(epochs)]
    if resume_path is not None:
        argv += ["--resume", resume_path]
    if init_from_path is not None:
        argv += ["--init-from", init_from_path]

    # Auto-resume is resolved inside the container (it's the only side
    # that can see whether the checkpoint file actually exists). Pass
    # the stage name so the remote driver knows what to look for.
    auto_resume_stage = stage if (auto_resume and resume_path is None) else None

    print(f"[modal] stage={stage}  protocol={training_protocol}  model={model_config}")
    print(f"[modal] gpu spec: {_GPU_SPEC}  (set MODAL_GPU_TYPE=<name> to pin)")
    if auto_resume_stage is not None:
        print(
            f"[modal] auto-resume: remote will check "
            f"{checkpoint_dir}/{stage}_latest.pt"
        )
    print(f"[modal] remote argv:\n  {' '.join(argv)}")
    run_train.remote(argv, auto_resume_stage=auto_resume_stage)
