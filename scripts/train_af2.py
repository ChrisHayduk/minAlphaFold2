"""Paper-spec AlphaFold2 training driver (supplement §1.11, Table 4).

Threads a :class:`~minalphafold.trainer.TrainingProtocol` into the trainer's
``fit`` one stage at a time, handling the cross-stage weight hand-off,
checkpoint naming, resume, and derivation of ``epochs`` from a sample
budget.

Usage (reproduce the paper exactly)::

    # Stage 1: initial training from random init.
    python scripts/train_af2.py \\
      --stage initial \\
      --checkpoint-dir checkpoints/af2

    # Stage 2: fine-tune from the initial-stage checkpoint.
    python scripts/train_af2.py \\
      --stage finetune \\
      --checkpoint-dir checkpoints/af2 \\
      --init-from checkpoints/af2/initial_latest.pt

Within-stage resume::

    python scripts/train_af2.py \\
      --stage initial \\
      --checkpoint-dir checkpoints/af2 \\
      --resume checkpoints/af2/initial_latest.pt

By default the script loads ``configs/alphafold2.toml`` (model) and
``configs/training_alphafold2.toml`` (training protocol), giving
Table 4 values verbatim. Override with ``--model-config`` or
``--training-protocol`` to plug in alternative profiles.

The paper's effective mini-batch size is 128 (1 example per TPU core).
On fewer accelerators the same effective batch is hit via gradient
accumulation — we default ``--grad-accum-steps`` to
``protocol.mini_batch_size // batch_size`` so the scaling is automatic.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minalphafold.trainer import (
    DataConfig,
    StageConfig,
    TrainingConfig,
    TrainingProtocol,
    default_device,
    fit,
    load_model_config,
    load_training_protocol,
)


def _epochs_for_target_samples(target_samples: int, dataset_size: int) -> int:
    """Convert a sample budget into the smallest ``epochs`` that covers it.

    The trainer is epoch-based: each epoch iterates the full dataset once.
    We pick ``ceil(target_samples / dataset_size)`` so the run sees at
    least ``target_samples`` training examples. With the paper's
    ``total_samples = 10⁷`` and an OpenProteinSet-sized training split
    (~10⁵ chains), this comes out to ~100 epochs of initial training.
    """
    if dataset_size <= 0:
        return 1
    return max(math.ceil(target_samples / dataset_size), 1)


def _count_training_chains(labels_dir: Path) -> int:
    """Rough dataset size estimate for the epochs calculation.

    One NPZ per chain, so the label count is the upper bound on training
    examples per epoch. The actual train/val split will lop ``val_fraction``
    off the top; we round up the epoch count to tolerate that.
    """
    if not labels_dir.exists():
        return 0
    return sum(1 for _ in labels_dir.glob("*.npz"))


def data_config_for_stage(
    stage: StageConfig,
    *,
    processed_features_dir: Path,
    processed_labels_dir: Path,
    val_fraction: float,
    chains_manifest: Path | None = None,
) -> DataConfig:
    """Build a :class:`DataConfig` with this stage's crop + MSA sizing."""
    return DataConfig(
        processed_features_dir=processed_features_dir,
        processed_labels_dir=processed_labels_dir,
        val_fraction=val_fraction,
        crop_size=stage.crop_size,
        msa_depth=stage.msa_depth,
        extra_msa_depth=stage.extra_msa_depth,
        max_templates=stage.max_templates,
        chains_manifest=chains_manifest,
    )


def training_config_for_stage(
    protocol: TrainingProtocol,
    stage: StageConfig,
    *,
    device: str,
    seed: int,
    batch_size: int,
    grad_accum_steps: int,
    num_workers: int,
    n_cycles: int,
    n_ensemble: int,
    epochs: int,
    is_finetune: bool,
    latest_checkpoint_path: Path,
    best_checkpoint_path: Path | None,
    resume_from_checkpoint: Path | None,
    init_weights_from_checkpoint: Path | None,
) -> TrainingConfig:
    """Compose a :class:`TrainingConfig` from the protocol + infra knobs.

    Notes on a few fields that aren't just pass-through:

    * The stage's ``learning_rate`` is the *actual* LR for this stage
      (1e-3 for initial, 5e-4 for fine-tune). Since we pass that as the
      base LR and set ``finetune_lr_scale = 1.0``, the trainer's
      fine-tune-halving logic is a no-op — the Table 4 value takes over
      untouched.
    * The schedule is entirely samples-driven: ``warmup_samples`` and
      the protocol's ``lr_decay_samples`` / ``lr_decay_factor`` match
      §1.11.3 verbatim. ``lr_schedule`` is left at ``"constant"`` so
      the step-based path is disabled.
    """
    optimizer = protocol.optimizer
    return TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        learning_rate=stage.learning_rate,
        adam_beta1=optimizer.adam_beta1,
        adam_beta2=optimizer.adam_beta2,
        adam_eps=optimizer.adam_eps,
        lr_schedule="constant",
        warmup_steps=0,
        warmup_samples=stage.warmup_samples,
        lr_decay_samples=optimizer.lr_decay_samples,
        lr_decay_factor=optimizer.lr_decay_factor,
        grad_clip_norm=optimizer.grad_clip_norm,
        ema_decay=optimizer.ema_decay,
        device=device,
        seed=seed,
        num_workers=num_workers,
        n_cycles=n_cycles,
        n_ensemble=n_ensemble,
        finetune=is_finetune,
        finetune_lr_scale=1.0,  # stage LR already encodes the Table 4 halving
        latest_checkpoint_path=latest_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        resume_from_checkpoint=resume_from_checkpoint,
        init_weights_from_checkpoint=init_weights_from_checkpoint,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-spec AlphaFold2 training (supplement §1.11 + Table 4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=["initial", "finetune"], required=True)
    parser.add_argument(
        "--checkpoint-dir", type=Path, required=True,
        help="Run directory for <stage>_latest.pt / <stage>_best.pt.",
    )
    parser.add_argument("--model-config", type=str, default="alphafold2")
    parser.add_argument("--training-protocol", type=str, default="alphafold2")

    parser.add_argument("--processed-features-dir", type=Path, default=Path("data/processed_features"))
    parser.add_argument("--processed-labels-dir", type=Path, default=Path("data/processed_labels"))
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument(
        "--chains-manifest", type=Path, default=None,
        help="Path to a JSON manifest from scripts/filter_openproteinset.py. "
             "Restricts training to §1.2.5-accepted chains.",
    )

    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--grad-accum-steps", type=int, default=None,
        help="If unset, derived from protocol.mini_batch_size so "
             "batch_size * grad_accum_steps equals the paper's 128.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-cycles", type=int, default=4,
                        help="Paper default per §1.10 / Algorithm 31.")
    parser.add_argument("--n-ensemble", type=int, default=1,
                        help="Paper default at training per §1.11.2.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="If unset, derived from stage.total_samples / dataset_size.")
    parser.add_argument(
        "--init-from", type=Path, default=None,
        help="(Fine-tune only) checkpoint whose model weights seed the "
             "fine-tune run. Required when --stage=finetune and --resume unset.",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Within-stage resume: restore model + optimizer + EMA + counters.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    protocol = load_training_protocol(args.training_protocol)
    model_config = load_model_config(args.model_config)
    stage = protocol.stage(args.stage)

    # Effective batch = batch_size × grad_accum_steps. Default scales
    # grad_accum_steps so the product matches the paper's mini_batch_size.
    grad_accum_steps = args.grad_accum_steps
    if grad_accum_steps is None:
        grad_accum_steps = max(protocol.optimizer.mini_batch_size // args.batch_size, 1)
    effective_batch = args.batch_size * grad_accum_steps

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.checkpoint_dir / f"{args.stage}_latest.pt"
    best_path = args.checkpoint_dir / f"{args.stage}_best.pt" if args.val_fraction > 0 else None

    # --- Fine-tune init vs resume resolution --------------------------
    # Fine-tune from scratch requires --init-from (cross-stage weight
    # hand-off). Resume (within-stage) supersedes init-from — it already
    # restores the live model state.
    init_weights_from: Path | None = None
    if args.resume is None and args.stage == "finetune":
        if args.init_from is None:
            raise SystemExit(
                "--init-from is required when --stage=finetune and --resume is not set "
                "(finetuning starts from the initial-stage checkpoint, supplement §1.11.1)."
            )
        init_weights_from = args.init_from
    elif args.resume is not None and args.init_from is not None:
        print(
            "[train] --resume set, ignoring --init-from "
            "(resume already restores model weights)."
        )

    # Epoch count: prefer explicit, otherwise derive from Table 4 target.
    if args.epochs is not None:
        epochs = args.epochs
    else:
        dataset_size = _count_training_chains(args.processed_labels_dir)
        epochs = _epochs_for_target_samples(stage.total_samples, dataset_size)
        print(
            f"[train] dataset_size≈{dataset_size} chains → "
            f"epochs={epochs} for target_samples={stage.total_samples:,}"
        )

    data_config = data_config_for_stage(
        stage,
        processed_features_dir=args.processed_features_dir,
        processed_labels_dir=args.processed_labels_dir,
        val_fraction=args.val_fraction,
        chains_manifest=args.chains_manifest,
    )
    training_config = training_config_for_stage(
        protocol, stage,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        grad_accum_steps=grad_accum_steps,
        num_workers=args.num_workers,
        n_cycles=args.n_cycles,
        n_ensemble=args.n_ensemble,
        epochs=epochs,
        is_finetune=(args.stage == "finetune"),
        latest_checkpoint_path=latest_path,
        best_checkpoint_path=best_path,
        resume_from_checkpoint=args.resume,
        init_weights_from_checkpoint=init_weights_from,
    )

    print(
        f"[train] stage={args.stage} protocol={protocol.protocol} "
        f"model={model_config.model_profile}"
    )
    print(
        f"[train] epochs={epochs} micro_batch={args.batch_size} "
        f"grad_accum={grad_accum_steps} effective_batch={effective_batch}"
    )
    print(
        f"[train] crop={stage.crop_size} msa={stage.msa_depth} "
        f"extra_msa={stage.extra_msa_depth} templ={stage.max_templates}"
    )
    print(
        f"[train] LR={stage.learning_rate} warmup_samples={stage.warmup_samples:,} "
        f"violation_weight={stage.violation_loss_weight}"
    )
    print(f"[train] checkpoints → {args.checkpoint_dir}")

    fit(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
    )


if __name__ == "__main__":
    main()
