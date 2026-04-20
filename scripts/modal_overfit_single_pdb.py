"""Run ``scripts/overfit_single_pdb.py`` on a Modal Labs GPU.

Sibling of ``scripts/modal_overfit.py``, but for the single-PDB overfit
that runs without MSAs or templates — the model sees only the target
sequence (synthetic 1-row MSA, no template features). Useful as a
sanity check that the architecture can fit a chain end-to-end purely
from sequence.

One-time setup (once per machine)::

    pip install modal
    modal setup

Launch a run::

    modal run scripts/modal_overfit_single_pdb.py --pdb ./artifacts/ground_truth_1a0m_A.pdb

All flags from the local script are exposed as Modal CLI options with
dashed names. The PDB file is streamed from your machine to the
container at call time, so the image stays small. Artifacts (predicted
PDB, metrics.json, losses.json, PyMOL view script) land in the
``minalphafold-artifacts`` Modal Volume. Fetch them back with::

    modal volume get minalphafold-artifacts overfit_single_pdb ./local/
"""

from __future__ import annotations

from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]

app = modal.App("minalphafold-overfit-single-pdb")

# Same image recipe as ``modal_overfit.py`` minus the data/ mounts — the
# single-PDB overfit needs no OpenProteinSet caches, only the package,
# config profiles, and scripts/.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.3", "numpy")
    .add_local_dir(str(REPO_ROOT / "minalphafold"), remote_path="/root/minalphafold")
    .add_local_dir(str(REPO_ROOT / "configs"), remote_path="/root/configs")
    .add_local_dir(str(REPO_ROOT / "scripts"), remote_path="/root/scripts")
)

artifacts = modal.Volume.from_name("minalphafold-artifacts", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/root/artifacts": artifacts},
    timeout=60 * 60 * 6,  # 6 hours — single-PDB overfit is much shorter than the full pipeline
)
def run_overfit(pdb_bytes: bytes, pdb_name: str, argv: list[str]) -> None:
    """Write the streamed PDB to /root/input/ and invoke overfit_single_pdb.main."""
    import os
    import sys

    os.chdir("/root")
    input_dir = Path("/root/input")
    input_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = input_dir / pdb_name
    pdb_path.write_bytes(pdb_bytes)

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from overfit_single_pdb import main as overfit_main

    overfit_main(argv)
    artifacts.commit()


@app.local_entrypoint()
def main(
    pdb: str,
    chain_id: str | None = None,
    model_profile: str = "alphafold2",
    steps: int = 1000,
    learning_rate: float = 1e-3,
    grad_clip_norm: float = 0.1,
    n_cycles: int = 1,
    log_every: int = 10,
    resolution: float = 2.0,
    use_clamped_fape: float | None = None,
    violations_after_step: int | None = None,
    seed: int = 0,
) -> None:
    """Stream a local PDB to Modal and run the single-PDB overfit on GPU."""
    pdb_path = Path(pdb).expanduser().resolve()
    if not pdb_path.is_file():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    pdb_bytes = pdb_path.read_bytes()
    pdb_name = pdb_path.name
    remote_pdb = f"/root/input/{pdb_name}"
    stem = chain_id or pdb_path.stem

    argv = [
        "--pdb", remote_pdb,
        "--model-profile", model_profile,
        "--steps", str(steps),
        "--learning-rate", str(learning_rate),
        "--grad-clip-norm", str(grad_clip_norm),
        "--n-cycles", str(n_cycles),
        "--log-every", str(log_every),
        "--resolution", str(resolution),
        "--seed", str(seed),
        "--device", "cuda",
        "--out-dir", f"/root/artifacts/overfit_single_pdb/{stem}",
    ]
    if chain_id is not None:
        argv += ["--chain-id", chain_id]
    if use_clamped_fape is not None:
        argv += ["--use-clamped-fape", str(use_clamped_fape)]
    if violations_after_step is not None:
        argv += ["--violations-after-step", str(violations_after_step)]

    print(f"[modal] streaming PDB: {pdb_path} ({len(pdb_bytes)} bytes)")
    print(f"[modal] remote argv:\n  {' '.join(argv)}")
    run_overfit.remote(pdb_bytes, pdb_name, argv)
