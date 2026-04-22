"""Training loop and config plumbing for min-AlphaFold2.

Mirrors the setup described in supplement Section 1.11:

* ``DataConfig`` carries the crop / MSA / template sizes from **Table 4**
  ("Initial training" column). The "Fine-tuning" column uses larger values
  (crop 384, Nseq 512, N_extra_seq 5120) — the user is expected to override
  those when entering the fine-tuning stage.
* ``TrainingConfig`` carries the optimisation hyperparameters from **1.11.3**
  (Adam + warmup + factor-0.95 decay), the two-stage protocol from **1.11.1**
  (``finetune`` / ``finetune_start_step``), and a ``finetune_lr_scale``
  matching the "reduce the base learning rate by half" rule.
* ``fit`` runs the loop, switching between the pre-training and fine-tuning
  ``AlphaFoldLoss`` instances via ``use_finetune_loss`` and applying the LR
  schedule described in 1.11.3.

Deviations from the paper for pedagogical reasons are called out inline:

* ``batch_size`` defaults to 1 (paper uses 128 across TPU cores).
* Gradient clipping is applied globally over the mini-batch rather than
  per-example — at ``batch_size=1`` these are identical.
* The LR schedule offers ``"constant"`` and ``"warmup_cosine"`` variants;
  the paper's exact schedule is "linear warmup → constant → one-shot
  ×0.95 decay at 6.4·10⁶ samples", which neither option reproduces — we
  document the tradeoff here rather than adding a bespoke schedule.
"""

from __future__ import annotations

import argparse
import tomllib
from collections.abc import Sized
from dataclasses import asdict, dataclass, is_dataclass, replace
from functools import partial
import math
from pathlib import Path
import random
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ProcessedOpenProteinSetDataset, collate_batch
from .losses import AlphaFoldLoss
from .model import AlphaFold2
from .model_config import ModelConfig


# Model profiles live in ``configs/<name>.toml`` at the repo root; see
# :func:`load_model_config`. Keeping them as data (not code) means
# experimenters can edit or ``cp`` a profile without touching this module.
CONFIGS_DIR: Path = Path(__file__).resolve().parents[1] / "configs"


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_config(name_or_path: str | Path) -> ModelConfig:
    """Load a model profile from ``configs/<name>.toml`` or a direct path.

    ``name_or_path`` may be a bare profile name (``"tiny"``, ``"medium"``,
    ``"alphafold2"`` — looked up in :data:`CONFIGS_DIR`) or a path to any
    TOML file with the same schema. The TOML is parsed with :mod:`tomllib`
    (stdlib, read-only in Python 3.11+) and the top-level table is passed
    straight to :class:`ModelConfig`, so a typo'd key in either the file
    or the schema surfaces immediately as a ``TypeError`` instead of
    later as an ``AttributeError`` during a forward pass.

    Tiny is a shrunk-to-CPU version of the AlphaFold2 monomer config
    (supplement 1.6 / Table 4); medium is a mid-sized profile for local
    experiments; alphafold2 matches the paper exactly (see
    ``configs/alphafold2.toml``).
    """
    path = Path(name_or_path)
    if not path.exists():
        candidate = CONFIGS_DIR / f"{name_or_path}.toml"
        if candidate.exists():
            path = candidate
        else:
            raise FileNotFoundError(
                f"No model config found at '{name_or_path}' or "
                f"'{candidate}'. Available profiles in {CONFIGS_DIR}: "
                f"{list_available_profiles()}"
            )
    with path.open("rb") as f:
        data = tomllib.load(f)
    return ModelConfig(**data)


@dataclass
class DataConfig:
    """Dataset + feature-sizing configuration (supplement Table 4).

    The defaults below match the **Initial training** column of Table 4:
    crop 256, Nseq 128, N_extra_seq 1024, Ntempl 4. The **Fine-tuning** column
    uses 384 / 512 / 5120 / 4 — bump ``crop_size``, ``msa_depth``, and
    ``extra_msa_depth`` when entering fine-tuning (supplement 1.11.1).

    ``block_delete_*`` controls the MSA block-deletion augmentation from
    supplement 1.2.6 (Algorithm 1). ``masked_msa_probability = 0.15`` is the
    BERT-style masking rate from supplement 1.2.7 that produces the targets
    for the masked-MSA loss (1.9.9 / equation 42).
    """

    processed_features_dir: str | Path = "data/processed_features"
    processed_labels_dir: str | Path = "data/processed_labels"
    val_fraction: float = 0.1
    crop_size: int = 256          # N_res — Table 4 initial training
    msa_depth: int = 128          # N_seq — Table 4 initial training
    extra_msa_depth: int = 1024   # N_extra_seq — Table 4 initial training
    max_templates: int = 4        # N_templ — Table 4 (both stages)
    # JSON manifest produced by ``scripts/filter_openproteinset.py`` —
    # limits the dataset to chains that pass the supplement §1.2.5
    # filters (resolution, single-AA dominance, min length). ``None``
    # (default) means use every chain that has a feature + label NPZ.
    chains_manifest: str | Path | None = None
    block_delete_training_msa: bool = True
    block_delete_msa_fraction: float = 0.3
    block_delete_msa_randomize_num_blocks: bool = False
    block_delete_msa_num_blocks: int = 5
    masked_msa_probability: float = 0.15
    fixed_feature_seed: int | None = None


@dataclass
class TrainingConfig:
    """Optimiser, schedule, and recycling settings (supplement 1.11).

    Defaults match supplement 1.11.3 / Table 4 where practical:

    * ``learning_rate = 1e-3``, ``adam_beta{1,2} = (0.9, 0.999)``,
      ``adam_eps = 1e-6`` per 1.11.3.
    * ``grad_clip_norm = 0.1`` per 1.11.3 ("clipping value 0.1"). The paper
      clips per-example; we clip over the full mini-batch, which is
      equivalent at ``batch_size = 1``.
    * ``finetune_lr_scale = 0.5`` implements "we reduce the base learning
      rate by half" (1.11.3) during the fine-tuning stage; matches Table 4's
      fine-tuning LR of 5·10⁻⁴.
    * ``batch_size = 1`` deviates from the paper's 128 (one example per
      TPU-core), since a pedagogical CPU/single-GPU run can't afford 128.
      ``grad_accum_steps`` closes the gap: the effective mini-batch is
      ``batch_size * grad_accum_steps``, so setting ``grad_accum_steps=128``
      at ``batch_size=1`` reproduces the paper's per-update sample count.
    * ``lr_schedule`` supports ``"constant"`` or ``"warmup_cosine"`` for
      step-based schedules. Setting either ``warmup_samples > 0`` or
      ``lr_decay_samples`` instead switches to the paper's samples-based
      "linear warm-up → constant → one-shot ×``lr_decay_factor`` drop"
      schedule from 1.11.3, which is the faithful option.
    * ``ema_decay`` enables the parameter EMA from 1.11.7 (paper: 0.999).
      The EMA shadow is used for validation-time loss and checkpoint
      selection, while ``model.state_dict()`` continues to hold the
      live training params. ``None`` disables EMA entirely.
    * ``resume_from_checkpoint`` points to a previously-saved checkpoint;
      when set, ``fit`` restores model + optimizer + EMA state and
      continues from the stored ``global_step`` / ``global_samples``.
    * ``n_cycles`` and ``n_ensemble`` default to 1; the paper uses 4 and 1
      respectively during training (supplement 1.10 and 1.11.2; ensembling
      is inference-only).
    """

    epochs: int = 1
    batch_size: int = 1                       # paper: 128 (TPU mini-batch)
    grad_accum_steps: int = 1                 # multiply with batch_size for effective batch
    learning_rate: float = 1e-3               # supplement 1.11.3
    min_learning_rate: float = 0.0
    weight_decay: float = 0.0
    adam_beta1: float = 0.9                   # supplement 1.11.3
    adam_beta2: float = 0.999                 # supplement 1.11.3
    adam_eps: float = 1e-6                    # supplement 1.11.3
    lr_schedule: str = "constant"
    warmup_steps: int = 0                     # paper: 128k samples / batch=128 = 1000 steps
    # Samples-based schedule hooks (supplement 1.11.3). When either of
    # these is set to a non-default value the trainer switches to a
    # samples-based "linear warm-up → constant → one-shot ×0.95 drop"
    # schedule that matches the paper exactly; otherwise we fall back to
    # the step-based ``lr_schedule`` above.
    warmup_samples: int = 0                   # §1.11.3 linear warm-up window
    lr_decay_samples: int | None = None       # §1.11.3 one-shot drop trigger
    lr_decay_factor: float = 1.0              # §1.11.3 multiplicative factor (paper: 0.95)
    grad_clip_norm: float | None = 0.1        # supplement 1.11.3
    ema_decay: float | None = None            # §1.11.7 parameter EMA (paper: 0.999)
    device: str = default_device()
    seed: int = 0
    num_workers: int = 0
    n_cycles: int = 1                         # paper: 4 (supplement 1.10)
    n_ensemble: int = 1                       # paper: 1 at training (1.11.2)
    finetune: bool = False                    # supplement 1.11.1 stage toggle
    finetune_start_step: int | None = None
    finetune_lr_scale: float = 0.5            # supplement 1.11.3 ("half the base LR")
    detach_rotations: bool = True
    latest_checkpoint_path: str | Path | None = None
    best_checkpoint_path: str | Path | None = None
    resume_from_checkpoint: str | Path | None = None
    # Seed model weights from a previous run without restoring optimizer
    # / EMA / counter state — exactly what the paper does at the start
    # of fine-tuning (§1.11.1). Ignored when ``resume_from_checkpoint``
    # is set, since resume already handles model weights.
    init_weights_from_checkpoint: str | Path | None = None


# ----------------------------------------------------------------------
# Two-stage training protocol — supplement Table 4 + §1.11.
#
# ``TrainingProtocol`` captures the paper's training recipe as data: the
# initial-training column, the fine-tuning column, and the shared Adam /
# gradient-clip / EMA settings that don't vary between stages. The file
# schema lives in ``configs/training_<name>.toml`` — Phase 2 of the
# housekeeping pass wires this into ``fit`` so a full paper-spec run is
# driven by a single protocol load. The dataclasses themselves stay pure
# configuration: no torch, no IO.
# ----------------------------------------------------------------------


@dataclass
class StageConfig:
    """One column of Supplementary Table 4.

    Fixes the crop and MSA sizing (§1.2.8, §1.11.1), the linear-warm-up
    schedule for this stage (§1.11.3), whether the structural-violation
    loss is active (§1.9.11), and the total number of training samples
    to run for the stage. Operational settings (device, seed, number of
    DataLoader workers, checkpoint paths) are intentionally absent —
    those are run-time infra, not part of the paper's protocol.
    """

    crop_size: int               # N_res   — §1.2.8
    msa_depth: int               # N_seq   — §1.11
    extra_msa_depth: int         # N_extra_seq
    max_templates: int           # N_templ — §1.2.3 / §1.7.1
    learning_rate: float         # §1.11.3 — base LR for this stage
    warmup_samples: int          # §1.11.3 — linear warm-up window
    violation_loss_weight: float # §1.9.11 — 0.0 initial, 1.0 fine-tune
    total_samples: int           # Table 4 — stage target sample count


@dataclass
class OptimizerConfig:
    """Shared optimizer + EMA settings — supplement §1.11.3 and §1.11.7.

    Identical across both training stages. The one-shot ×0.95 LR decay at
    6.4M samples is shared because it's specified as an absolute count
    within a single contiguous run (§1.11.3); ``mini_batch_size = 128``
    reflects the paper's 1-example-per-TPU-core arrangement and is the
    *effective* batch the trainer must hit via gradient accumulation on
    hardware with fewer accelerators.
    """

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-6
    grad_clip_norm: float = 0.1
    ema_decay: float = 0.999
    mini_batch_size: int = 128
    lr_decay_samples: int = 6_400_000
    lr_decay_factor: float = 0.95


@dataclass
class TrainingProtocol:
    """Paper's two-stage training recipe (Table 4 + §1.11)."""

    protocol: str
    optimizer: OptimizerConfig
    initial: StageConfig
    finetune: StageConfig

    def stage(self, name: str) -> StageConfig:
        """Return the stage with the given name ('initial' or 'finetune')."""
        if name == "initial":
            return self.initial
        if name == "finetune":
            return self.finetune
        raise ValueError(
            f"Unknown training stage '{name}'. Expected 'initial' or 'finetune'."
        )


def list_available_profiles() -> list[str]:
    """Return the model-profile names shipped in :data:`CONFIGS_DIR`.

    Excludes training-protocol files (``training_*.toml``) — those belong
    to :func:`list_available_training_protocols`.
    """
    return sorted(
        p.stem for p in CONFIGS_DIR.glob("*.toml")
        if not p.stem.startswith("training_")
    )


def list_available_training_protocols() -> list[str]:
    """Return the training-protocol names shipped in :data:`CONFIGS_DIR`.

    Files must be named ``training_<name>.toml`` — the ``training_``
    prefix is what distinguishes them from model profiles on disk.
    """
    return sorted(
        p.stem.removeprefix("training_")
        for p in CONFIGS_DIR.glob("training_*.toml")
    )


def load_training_protocol(name_or_path: str | Path) -> TrainingProtocol:
    """Load a training protocol from ``configs/training_<name>.toml``.

    ``name_or_path`` is either a bare profile name (``"alphafold2"`` →
    ``configs/training_alphafold2.toml``) or a direct path to any TOML
    file with the same schema. Every documented key is required; typos
    on either side surface as ``TypeError`` at load time via the
    dataclass signature, not at first attribute access mid-training.
    """
    path = Path(name_or_path)
    if not path.exists():
        candidate = CONFIGS_DIR / f"training_{name_or_path}.toml"
        if candidate.exists():
            path = candidate
        else:
            raise FileNotFoundError(
                f"No training protocol found at '{name_or_path}' or "
                f"'{candidate}'. Available protocols in {CONFIGS_DIR}: "
                f"{list_available_training_protocols()}"
            )
    with path.open("rb") as f:
        data = tomllib.load(f)
    return TrainingProtocol(
        protocol=data["protocol"],
        optimizer=OptimizerConfig(**data["optimizer"]),
        initial=StageConfig(**data["initial"]),
        finetune=StageConfig(**data["finetune"]),
    )


def copy_model_config(model_config: ModelConfig, **overrides: Any) -> ModelConfig:
    """Return a copy of ``model_config`` with the given fields overridden."""
    return replace(model_config, **overrides)


def zero_dropout_model_config(model_config: ModelConfig) -> ModelConfig:
    """Clone a config with every dropout rate set to 0.

    Used for overfit / memorisation experiments (``scripts/overfit_*.py``)
    where the stochastic regularisation from supplement 1.11.6 would prevent
    the model from fitting a single example.
    """
    return replace(
        model_config,
        template_pair_dropout=0.0,
        extra_msa_dropout=0.0,
        extra_pair_dropout=0.0,
        evoformer_msa_dropout=0.0,
        evoformer_pair_dropout=0.0,
        structure_module_dropout_ipa=0.0,
        structure_module_dropout_transition=0.0,
        model_profile=f"{model_config.model_profile}_no_dropout",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return device


def build_dataloader(
    split: str,
    data_config: DataConfig,
    *,
    training: bool,
    batch_size: int = 1,
    num_workers: int = 0,
    device: str = "cpu",
    seed: int = 0,
    n_cycles: int = 1,
    n_ensemble: int = 1,
) -> DataLoader:
    dataset = ProcessedOpenProteinSetDataset(
        data_config.processed_features_dir,
        data_config.processed_labels_dir,
        split=split,
        val_fraction=data_config.val_fraction,
        seed=seed,
        chains_manifest=data_config.chains_manifest,
    )
    collate_fn = partial(
        collate_batch,
        crop_size=data_config.crop_size,
        msa_depth=data_config.msa_depth,
        extra_msa_depth=data_config.extra_msa_depth,
        max_templates=data_config.max_templates,
        training=training,
        block_delete_training_msa=data_config.block_delete_training_msa,
        block_delete_msa_fraction=data_config.block_delete_msa_fraction,
        block_delete_msa_randomize_num_blocks=data_config.block_delete_msa_randomize_num_blocks,
        block_delete_msa_num_blocks=data_config.block_delete_msa_num_blocks,
        masked_msa_probability=data_config.masked_msa_probability,
        random_seed=data_config.fixed_feature_seed,
        num_recycling_samples=n_cycles,
        num_ensemble_samples=n_ensemble,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=collate_fn,
        generator=generator,
    )


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def learning_rate_for_step(training_config: TrainingConfig, step: int, total_steps: int) -> float:
    """Pre-training learning-rate schedule (supplement 1.11.3).

    * ``"constant"``: base lr every step.
    * ``"warmup_cosine"``: linear warmup for ``warmup_steps``, then cosine
      decay from the base lr down to ``min_learning_rate`` over the rest of
      training. Cosine is a pedagogical stand-in for the paper's exact
      "constant then one-shot ×0.95 at 6.4·10⁶ samples" rule.

    Fine-tuning uses a separate code path in the training loop that
    bypasses this schedule (no warmup per 1.11.3, constant
    ``learning_rate * finetune_lr_scale``).
    """
    base_lr = training_config.learning_rate
    if training_config.lr_schedule == "constant":
        return base_lr

    if training_config.lr_schedule != "warmup_cosine":
        raise ValueError(f"Unsupported learning-rate schedule: {training_config.lr_schedule}")

    warmup_steps = max(training_config.warmup_steps, 0)
    min_lr = training_config.min_learning_rate
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(warmup_steps, 1))

    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps + 1) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def learning_rate_for_samples(
    base_learning_rate: float,
    samples_seen: int,
    warmup_samples: int,
    lr_decay_samples: int | None,
    lr_decay_factor: float,
) -> float:
    """Paper's samples-based LR schedule (supplement 1.11.3).

    ``base_lr`` linearly ramps from 0 over the first ``warmup_samples``
    training examples, sits at ``base_lr`` through the constant phase,
    then multiplies by ``lr_decay_factor`` once ``samples_seen`` crosses
    ``lr_decay_samples`` (the paper's one-shot ×0.95 drop at 6.4·10⁶
    samples). The fine-tuning stage sets ``warmup_samples = 0`` per the
    supplement's "no learning rate warm-up" rule.

    The warm-up fraction uses ``samples_seen / warmup_samples`` rather
    than ``+1`` so a run that starts at ``samples_seen = 0`` begins
    training at LR = 0 and crosses ``base_lr`` exactly at
    ``samples_seen = warmup_samples``, matching the paper's description
    of a linear warm-up window.
    """
    if warmup_samples > 0 and samples_seen < warmup_samples:
        return base_learning_rate * samples_seen / warmup_samples
    if lr_decay_samples is not None and samples_seen >= lr_decay_samples:
        return base_learning_rate * lr_decay_factor
    return base_learning_rate


def _uses_samples_schedule(training_config: TrainingConfig) -> bool:
    """Return True iff the config has opted into the paper's samples schedule."""
    return training_config.warmup_samples > 0 or training_config.lr_decay_samples is not None


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def build_optimizer(model: AlphaFold2, training_config: TrainingConfig) -> torch.optim.Optimizer:
    """Construct Adam with supplement 1.11.3 hyperparameters."""
    return torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(training_config.adam_beta1, training_config.adam_beta2),
        eps=training_config.adam_eps,
        weight_decay=training_config.weight_decay,
    )


def build_ema_model(model: AlphaFold2, ema_decay: float) -> torch.optim.swa_utils.AveragedModel:
    """Wrap ``model`` in an ``AveragedModel`` maintaining its EMA (§1.11.7).

    The averaging function implements a plain exponential moving average:
    ``ema ← decay·ema + (1 − decay)·current``. We special-case the first
    ``update_parameters`` call (``num_averaged == 0``) to copy ``current``
    outright, which makes the EMA start from the first post-step model
    rather than a decayed blend of the identical init params — this
    matches the behaviour described in supplement 1.11.7.
    """

    def avg_fn(
        averaged_param: torch.Tensor,
        current_param: torch.Tensor,
        num_averaged: torch.Tensor | int,
    ) -> torch.Tensor:
        if int(num_averaged) == 0:
            return current_param.detach().clone()
        return ema_decay * averaged_param + (1.0 - ema_decay) * current_param

    return torch.optim.swa_utils.AveragedModel(model, avg_fn=avg_fn)


def load_checkpoint_for_resume(
    path: str | Path,
    model: AlphaFold2,
    optimizer: torch.optim.Optimizer,
    ema_model: torch.optim.swa_utils.AveragedModel | None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Restore model / optimizer / EMA state from a checkpoint.

    Returns a dict of resume metadata (``epoch``, ``global_step``,
    ``global_samples``, ``best_val_loss``, ``history``) so the training
    loop can continue from where the saved run left off. ``epoch`` is
    offset by one — the caller starts the next epoch, not the saved one.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if ema_model is not None and "ema_state_dict" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_state_dict"])
    history_raw = checkpoint.get("history", [])
    return {
        "epoch": int(checkpoint.get("epoch", 0)) + 1,
        "global_step": int(checkpoint.get("global_step", 0)),
        "global_samples": int(checkpoint.get("global_samples", 0)),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "history": list(history_raw),
    }


def use_finetune_loss(training_config: TrainingConfig, global_step: int) -> bool:
    """Decide whether to use the fine-tuning loss at the current step.

    Supplement 1.11.1 splits training into two stages. ``finetune_start_step``
    lets the caller continue a single run into fine-tuning after a warmup of
    pre-training; ``finetune=True`` starts in fine-tuning from step 0.
    """
    if training_config.finetune_start_step is not None:
        return global_step >= training_config.finetune_start_step
    return training_config.finetune


def learning_rate_at_step(
    training_config: TrainingConfig,
    step: int,
    total_steps: int,
    *,
    is_finetune: bool,
    samples_seen: int = 0,
) -> float:
    """LR for the current step, applying the fine-tuning rule from 1.11.3.

    Three paths:

    * **Fine-tuning** (``is_finetune=True``): constant
      ``learning_rate * finetune_lr_scale`` with no warmup — matches
      supplement 1.11.3 ("during the fine-tuning stage we have no
      learning rate warm-up, but we reduce the base learning rate by
      half"). The ``lr_decay_samples`` one-shot drop is inherited from
      the optimizer config if it fires during fine-tuning.
    * **Paper samples-based schedule** (either ``warmup_samples > 0`` or
      ``lr_decay_samples`` set): delegates to
      :func:`learning_rate_for_samples`, which reproduces the paper's
      linear warm-up → constant → one-shot ×0.95 drop exactly.
    * **Step-based schedule** (default): delegates to
      :func:`learning_rate_for_step` (``"constant"`` or
      ``"warmup_cosine"``), kept for pedagogical runs that want a
      gentler stand-in.

    ``samples_seen`` is only consulted when the samples-based schedule is
    active (or during fine-tuning when ``lr_decay_samples`` is set);
    otherwise the value is ignored.
    """
    if is_finetune:
        finetune_lr = training_config.learning_rate * training_config.finetune_lr_scale
        if (
            training_config.lr_decay_samples is not None
            and samples_seen >= training_config.lr_decay_samples
        ):
            return finetune_lr * training_config.lr_decay_factor
        return finetune_lr
    if _uses_samples_schedule(training_config):
        return learning_rate_for_samples(
            training_config.learning_rate,
            samples_seen,
            training_config.warmup_samples,
            training_config.lr_decay_samples,
            training_config.lr_decay_factor,
        )
    return learning_rate_for_step(training_config, step, total_steps)


def model_inputs_from_batch(batch: dict[str, Any], training_config: TrainingConfig) -> dict[str, torch.Tensor | int]:
    """Unpack a collated batch into the kwargs ``AlphaFold2.forward`` expects."""
    return {
        "target_feat": batch["target_feat"],
        "residue_index": batch["residue_index"],
        "msa_feat": batch["msa_feat"],
        "extra_msa_feat": batch["extra_msa_feat"],
        "template_pair_feat": batch["template_pair_feat"],
        "aatype": batch["aatype"],
        "template_angle_feat": batch["template_angle_feat"],
        "template_mask": batch["template_mask"],
        "template_residue_mask": batch["template_residue_mask"],
        "seq_mask": batch["seq_mask"],
        "msa_mask": batch["msa_mask"],
        "extra_msa_mask": batch["extra_msa_mask"],
        "n_cycles": training_config.n_cycles,
        "n_ensemble": training_config.n_ensemble,
        "detach_rotations": training_config.detach_rotations,
    }


def collapse_sampled_batch_tensor(
    tensor: torch.Tensor,
    *,
    recycle_index: int | None = None,
    ensemble_index: int = 0,
) -> torch.Tensor:
    """Strip the per-cycle / per-ensemble outer dims off a pre-sampled tensor.

    The data pipeline may materialise ``N_cycle × N_ensemble`` samples of the
    masked-MSA target by prepending two outer axes (supplement 1.11.2). The
    loss only consumes one sample — by default the last cycle's zeroth
    ensemble member, which matches the forward pass's final iteration.
    """
    if tensor.ndim >= 5:
        if recycle_index is None:
            recycle_index = tensor.shape[0] - 1
        return tensor[recycle_index, ensemble_index]
    return tensor


def loss_inputs_from_batch(batch: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    """Assemble the kwargs ``AlphaFoldLoss`` expects from a batch + model outputs.

    The returned dict intentionally mixes tensors with the raw
    ``structure_model_prediction`` sub-dict and an optional ``resolution``
    scalar, so we type it as ``dict[str, Any]`` rather than forcing a
    narrower promise we don't keep.
    """
    recycle_index = max(int(outputs.get("sampled_n_cycles", 1)) - 1, 0)
    ensemble_index = 0
    return {
        "structure_model_prediction": outputs,
        "true_rotations": batch["true_rotations"],
        "true_translations": batch["true_translations"],
        "true_atom_positions": batch["true_atom_positions"],
        "true_atom_mask": batch["true_atom_mask"],
        "true_atom_positions_alt": batch["true_atom_positions_alt"],
        "true_atom_mask_alt": batch["true_atom_mask_alt"],
        "true_atom_is_ambiguous": batch["true_atom_is_ambiguous"],
        "true_torsion_angles": batch["true_torsion_angles"],
        "true_torsion_angles_alt": batch["true_torsion_angles_alt"],
        "true_torsion_mask": batch["true_torsion_mask"],
        "true_rigid_group_frames_R": batch["true_rigid_group_frames_R"],
        "true_rigid_group_frames_t": batch["true_rigid_group_frames_t"],
        "true_rigid_group_frames_R_alt": batch["true_rigid_group_frames_R_alt"],
        "true_rigid_group_frames_t_alt": batch["true_rigid_group_frames_t_alt"],
        "true_rigid_group_exists": batch["true_rigid_group_exists"],
        "experimentally_resolved_pred": outputs["experimentally_resolved_logits"],
        "experimentally_resolved_true": batch["experimentally_resolved_true"],
        "experimentally_resolved_exists": batch["atom37_exists"],
        "resolution": batch.get("resolution"),
        "masked_msa_pred": outputs["masked_msa_logits"],
        "masked_msa_target": collapse_sampled_batch_tensor(
            batch["masked_msa_target"],
            recycle_index=recycle_index,
            ensemble_index=ensemble_index,
        ),
        "masked_msa_mask": collapse_sampled_batch_tensor(
            batch["masked_msa_mask"],
            recycle_index=recycle_index,
            ensemble_index=ensemble_index,
        ),
        "plddt_pred": outputs["plddt_logits"],
        "distogram_pred": outputs["distogram_logits"],
        "tm_pred": outputs["tm_logits"],
        "res_types": batch["res_types"],
        "residue_index": batch["residue_index"],
        "seq_mask": batch["seq_mask"],
    }


def train_step(
    model: AlphaFold2,
    loss_fn: AlphaFoldLoss,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    training_config: TrainingConfig,
) -> dict[str, float]:
    """Single forward/backward/step iteration.

    Gradient clipping follows supplement 1.11.3 (clip by global norm = 0.1
    by default). The paper clips per-example within a mini-batch; we clip
    over the whole mini-batch, which is equivalent at ``batch_size = 1``
    (the default pedagogical setting).
    """
    device = resolve_device(training_config.device)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    batch = move_to_device(batch, device)
    outputs = model(**model_inputs_from_batch(batch, training_config))
    per_example_loss = loss_fn(**loss_inputs_from_batch(batch, outputs))
    loss = per_example_loss.mean()
    loss.backward()

    if training_config.grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip_norm)

    optimizer.step()
    return {"loss": float(loss.item())}


def evaluate(
    model: AlphaFold2 | torch.optim.swa_utils.AveragedModel,
    loss_fn: AlphaFoldLoss,
    dataloader: DataLoader,
    training_config: TrainingConfig,
) -> dict[str, float]:
    """Return mean per-example loss over a dataloader (no gradient updates).

    Accepts either the live :class:`AlphaFold2` or an
    :class:`AveragedModel` wrapping one — the latter is what
    :func:`fit` passes in when EMA is enabled (supplement 1.11.7).
    ``AveragedModel.__call__`` forwards to its wrapped module, so the
    call site is identical.
    """
    device = resolve_device(training_config.device)
    model.eval()

    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = move_to_device(batch, device)
            outputs = model(**model_inputs_from_batch(batch, training_config))
            per_example_loss = loss_fn(**loss_inputs_from_batch(batch, outputs))
            total_loss += float(per_example_loss.sum().item())
            total_examples += int(per_example_loss.shape[0])

    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty dataloader.")

    return {"loss": total_loss / total_examples}


def config_to_dict(config: Any) -> Any:
    """Serialise a config to a plain dict for checkpointing.

    Accepts dataclass instances (but not dataclass *classes* — hence the
    ``not isinstance(config, type)`` guard that keeps ``asdict`` happy),
    plain dicts, or anything with ``__dict__`` (e.g. ``SimpleNamespace``
    model configs).
    """
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return config


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    model: AlphaFold2,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float | None,
    history: list[dict[str, float | int]],
    data_config: DataConfig,
    training_config: TrainingConfig,
    model_config: Any,
    global_step: int = 0,
    global_samples: int = 0,
    ema_model: torch.optim.swa_utils.AveragedModel | None = None,
) -> None:
    """Persist training state to ``path``.

    ``global_step`` / ``global_samples`` are the counters needed to resume
    a samples-based LR schedule correctly; ``ema_state_dict`` carries the
    EMA shadow parameters and is only included when an EMA is in use.
    Older checkpoints without these keys are still loadable via
    :func:`load_checkpoint_for_resume` — the loader defaults to zero
    counters and no EMA restore in their absence.
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "global_samples": global_samples,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "history": history,
        "data_config": config_to_dict(data_config),
        "training_config": config_to_dict(training_config),
        "model_config": config_to_dict(model_config),
    }
    if ema_model is not None:
        payload["ema_state_dict"] = ema_model.state_dict()
    torch.save(payload, checkpoint_path)


def fit(
    model_config: Any | None = None,
    data_config: DataConfig | None = None,
    training_config: TrainingConfig | None = None,
) -> tuple[AlphaFold2, list[dict[str, float | int]]]:
    """Top-level training loop — supplement 1.11.1 / 1.11.3 / 1.11.7.

    Each training iteration:

    1. Decide pre-training vs fine-tuning by ``use_finetune_loss``.
    2. Forward + backward one micro-batch; divide loss by
       ``grad_accum_steps`` so the accumulated gradient has the same
       expected scale as a single full-batch backward.
    3. Clip the micro-batch's gradient by global-norm (§1.11.3). At
       ``batch_size=1`` this matches the paper's per-example clipping;
       at larger batch sizes it's per-micro-batch and slightly
       different. See :class:`TrainingConfig` for the caveat.
    4. Either accumulate into a dedicated buffer (when
       ``grad_accum_steps > 1``) and continue, or copy the clipped
       gradient into ``.grad``, set the LR via
       :func:`learning_rate_at_step` (samples-based schedule when
       configured), call ``optimizer.step``, and update the EMA
       (§1.11.7) if one is active.

    Stage transitions (Table 4) — crop-size bump, LR halving, violation
    loss on — are the caller's responsibility: swap the configs and
    either resume from the previous checkpoint or start a new ``fit``.
    """
    model_config = load_model_config("tiny") if model_config is None else model_config
    data_config = DataConfig() if data_config is None else data_config
    training_config = TrainingConfig() if training_config is None else training_config

    set_seed(training_config.seed)
    device = resolve_device(training_config.device)

    train_loader = build_dataloader(
        "train",
        data_config,
        training=True,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        device=str(device),
        seed=training_config.seed,
        n_cycles=training_config.n_cycles,
        n_ensemble=training_config.n_ensemble,
    )
    # ``DataLoader.dataset`` is typed as the non-Sized base ``Dataset``; our
    # concrete ``ProcessedOpenProteinSetDataset`` (and the fallback in tests)
    # all define ``__len__`` so the cast is safe.
    if len(cast(Sized, train_loader.dataset)) == 0:
        raise ValueError("Training split is empty. Check the processed cache paths and val_fraction.")

    val_loader = build_dataloader(
        "val",
        data_config,
        training=False,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        device=str(device),
        seed=training_config.seed,
        n_cycles=training_config.n_cycles,
        n_ensemble=training_config.n_ensemble,
    )
    has_validation = data_config.val_fraction > 0.0 and len(cast(Sized, val_loader.dataset)) > 0

    model = AlphaFold2(model_config).to(device)
    pretrain_loss_fn = AlphaFoldLoss(finetune=False).to(device)
    finetune_loss_fn = AlphaFoldLoss(finetune=True).to(device)
    optimizer = build_optimizer(model, training_config)

    # --- Cross-stage weight init (supplement 1.11.1) --------------------
    # Fine-tuning starts from the initial-stage checkpoint's weights but
    # with a fresh optimizer, zero counters, and (when enabled) a fresh
    # EMA seeded from the just-loaded weights. Ignored when resuming,
    # since resume's own restore covers the model weights.
    if (
        training_config.init_weights_from_checkpoint is not None
        and training_config.resume_from_checkpoint is None
    ):
        init_payload = torch.load(
            training_config.init_weights_from_checkpoint,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(init_payload["model_state_dict"])
        print(
            f"[init-from] loaded model weights from "
            f"{training_config.init_weights_from_checkpoint}"
        )

    # --- Parameter EMA (supplement 1.11.7) --------------------------------
    ema_model: torch.optim.swa_utils.AveragedModel | None = None
    if training_config.ema_decay is not None:
        ema_model = build_ema_model(model, training_config.ema_decay).to(device)

    # --- Resume-from-checkpoint -----------------------------------------
    history: list[dict[str, float | int]] = []
    best_val_loss: float | None = None
    global_step = 0
    global_samples = 0
    start_epoch = 1
    if training_config.resume_from_checkpoint is not None:
        restored = load_checkpoint_for_resume(
            training_config.resume_from_checkpoint,
            model,
            optimizer,
            ema_model,
            map_location=device,
        )
        start_epoch = restored["epoch"]
        global_step = restored["global_step"]
        global_samples = restored["global_samples"]
        best_val_loss = restored["best_val_loss"]
        history = cast(list[dict[str, float | int]], restored["history"])
        print(
            f"[resume] epoch={start_epoch - 1} global_step={global_step} "
            f"global_samples={global_samples}"
        )

    # --- Gradient accumulation ------------------------------------------
    # The buffer doubles gradient memory (one extra param-sized float
    # tensor) but is the only way to clip per-micro-batch before summing,
    # which in turn approximates the paper's per-example clipping rule at
    # batch_size=1. At grad_accum_steps=1 the buffer is unused.
    grad_accum_steps = max(training_config.grad_accum_steps, 1)
    grad_accumulator: list[torch.Tensor] | None = None
    if grad_accum_steps > 1:
        grad_accumulator = [torch.zeros_like(p) for p in model.parameters()]

    steps_per_epoch = max(len(train_loader) // grad_accum_steps, 1)
    total_steps = training_config.epochs * steps_per_epoch

    for epoch in range(start_epoch, training_config.epochs + 1):
        total_train_loss = 0.0
        total_train_examples = 0
        accum_count = 0
        optimizer.zero_grad(set_to_none=True)

        # Seed ``current_lr`` before the inner loop so an empty
        # ``train_loader`` still leaves a valid LR for ``epoch_metrics``.
        current_lr = learning_rate_at_step(
            training_config,
            global_step,
            total_steps,
            is_finetune=use_finetune_loss(training_config, global_step),
            samples_seen=global_samples,
        )

        for batch in train_loader:
            is_finetune = use_finetune_loss(training_config, global_step)
            loss_fn = finetune_loss_fn if is_finetune else pretrain_loss_fn

            model.train()
            batch = move_to_device(batch, device)
            outputs = model(**model_inputs_from_batch(batch, training_config))
            per_example_loss = loss_fn(**loss_inputs_from_batch(batch, outputs))
            micro_batch_loss = per_example_loss.mean()
            (micro_batch_loss / grad_accum_steps).backward()

            batch_size = int(batch["aatype"].shape[0])
            total_train_loss += float(micro_batch_loss.item()) * batch_size
            total_train_examples += batch_size
            global_samples += batch_size
            accum_count += 1

            # Per-micro-batch clip (§1.11.3). At batch_size=1 this is
            # per-example; at larger batches it approximates.
            if training_config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip_norm)

            if grad_accumulator is not None:
                for accum, parameter in zip(grad_accumulator, model.parameters()):
                    if parameter.grad is not None:
                        accum.add_(parameter.grad)
                optimizer.zero_grad(set_to_none=True)

            if accum_count < grad_accum_steps:
                continue

            # --- End of accumulation window: step the optimizer --------
            if grad_accumulator is not None:
                for accum, parameter in zip(grad_accumulator, model.parameters()):
                    parameter.grad = accum.clone()
                    accum.zero_()

            current_lr = learning_rate_at_step(
                training_config,
                global_step,
                total_steps,
                is_finetune=is_finetune,
                samples_seen=global_samples,
            )
            set_optimizer_learning_rate(optimizer, current_lr)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if ema_model is not None:
                ema_model.update_parameters(model)

            accum_count = 0
            global_step += 1

        # Partial accumulation at end of epoch is discarded — next epoch
        # starts fresh with a cleared optimizer and accumulator. This
        # keeps epoch boundaries crisp for checkpointing.
        if accum_count > 0:
            optimizer.zero_grad(set_to_none=True)
            if grad_accumulator is not None:
                for accum in grad_accumulator:
                    accum.zero_()
            accum_count = 0

        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": total_train_loss / max(total_train_examples, 1),
            "learning_rate": current_lr,
            "global_step": global_step,
            "global_samples": global_samples,
        }

        if has_validation:
            # §1.11.7: eval against the EMA shadow when one is configured.
            eval_model: AlphaFold2 | torch.optim.swa_utils.AveragedModel = (
                ema_model if ema_model is not None else model
            )
            val_loss_fn = (
                finetune_loss_fn
                if use_finetune_loss(training_config, global_step)
                else pretrain_loss_fn
            )
            val_metrics = evaluate(eval_model, val_loss_fn, val_loader, training_config)
            epoch_metrics["val_loss"] = val_metrics["loss"]

        history.append(epoch_metrics)

        if "val_loss" in epoch_metrics:
            print(
                f"epoch {epoch}/{training_config.epochs} "
                f"train_loss={epoch_metrics['train_loss']:.4f} "
                f"val_loss={epoch_metrics['val_loss']:.4f} "
                f"samples={global_samples}"
            )
        else:
            print(
                f"epoch {epoch}/{training_config.epochs} "
                f"train_loss={epoch_metrics['train_loss']:.4f} "
                f"samples={global_samples}"
            )

        if training_config.latest_checkpoint_path is not None:
            save_checkpoint(
                training_config.latest_checkpoint_path,
                epoch=epoch,
                global_step=global_step,
                global_samples=global_samples,
                model=model,
                optimizer=optimizer,
                ema_model=ema_model,
                best_val_loss=best_val_loss,
                history=history,
                data_config=data_config,
                training_config=training_config,
                model_config=model_config,
            )

        if training_config.best_checkpoint_path is not None and "val_loss" in epoch_metrics:
            current_val_loss = float(epoch_metrics["val_loss"])
            if best_val_loss is None or current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                save_checkpoint(
                    training_config.best_checkpoint_path,
                    epoch=epoch,
                    global_step=global_step,
                    global_samples=global_samples,
                    model=model,
                    optimizer=optimizer,
                    ema_model=ema_model,
                    best_val_loss=best_val_loss,
                    history=history,
                    data_config=data_config,
                    training_config=training_config,
                    model_config=model_config,
                )

    return model, history


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pedagogical AlphaFold2 model on processed OpenProteinSet caches.")
    parser.add_argument(
        "--model-config",
        type=str,
        default="tiny",
        help=(
            "Model profile to train. Either a shipped profile name "
            f"({', '.join(list_available_profiles())}) resolved under "
            f"{CONFIGS_DIR}, or a path to any TOML file with the same schema."
        ),
    )
    parser.add_argument("--processed-features-dir", type=str, default="data/processed_features")
    parser.add_argument("--processed-labels-dir", type=str, default="data/processed_labels")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--msa-depth", type=int, default=128)
    parser.add_argument("--extra-msa-depth", type=int, default=1024)
    parser.add_argument("--max-templates", type=int, default=4)
    parser.add_argument("--disable-block-delete-training-msa", action="store_true")
    parser.add_argument("--block-delete-msa-fraction", type=float, default=0.3)
    parser.add_argument("--block-delete-msa-num-blocks", type=int, default=5)
    parser.add_argument("--block-delete-msa-randomize-num-blocks", action="store_true")
    parser.add_argument("--masked-msa-probability", type=float, default=0.15)
    parser.add_argument("--fixed-feature-seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1,
        help="Accumulate this many micro-batches before each optimizer step "
             "(effective batch = batch_size * grad_accum_steps; paper uses 128).",
    )
    # Optimiser defaults mirror supplement 1.11.3 (Adam base lr 10⁻³, ε=10⁻⁶,
    # β=(0.9, 0.999), gradient-clipping norm 0.1).
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-6)
    parser.add_argument("--lr-schedule", choices=["constant", "warmup_cosine"], default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    # Samples-based schedule (§1.11.3). Setting either activates the paper
    # schedule and overrides ``--lr-schedule`` / ``--warmup-steps``.
    parser.add_argument("--warmup-samples", type=int, default=0)
    parser.add_argument("--lr-decay-samples", type=int, default=None)
    parser.add_argument("--lr-decay-factor", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.1)
    parser.add_argument(
        "--ema-decay", type=float, default=None,
        help="Enable parameter EMA with this decay (paper: 0.999, §1.11.7).",
    )
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-cycles", type=int, default=1)
    parser.add_argument("--n-ensemble", type=int, default=1)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--finetune-start-step", type=int, default=None)
    parser.add_argument("--finetune-lr-scale", type=float, default=0.5)
    parser.add_argument("--latest-checkpoint-path", type=str, default=None)
    parser.add_argument("--best-checkpoint-path", type=str, default=None)
    parser.add_argument(
        "--resume-from-checkpoint", type=str, default=None,
        help="Restore model+optimizer+EMA state from this checkpoint path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> tuple[AlphaFold2, list[dict[str, float | int]]]:
    args = parse_args(argv)

    data_config = DataConfig(
        processed_features_dir=args.processed_features_dir,
        processed_labels_dir=args.processed_labels_dir,
        val_fraction=args.val_fraction,
        crop_size=args.crop_size,
        msa_depth=args.msa_depth,
        extra_msa_depth=args.extra_msa_depth,
        max_templates=args.max_templates,
        block_delete_training_msa=not args.disable_block_delete_training_msa,
        block_delete_msa_fraction=args.block_delete_msa_fraction,
        block_delete_msa_randomize_num_blocks=args.block_delete_msa_randomize_num_blocks,
        block_delete_msa_num_blocks=args.block_delete_msa_num_blocks,
        masked_msa_probability=args.masked_msa_probability,
        fixed_feature_seed=args.fixed_feature_seed,
    )
    grad_clip_norm = None if args.grad_clip_norm is not None and args.grad_clip_norm <= 0 else args.grad_clip_norm
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        warmup_samples=args.warmup_samples,
        lr_decay_samples=args.lr_decay_samples,
        lr_decay_factor=args.lr_decay_factor,
        grad_clip_norm=grad_clip_norm,
        ema_decay=args.ema_decay,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        n_cycles=args.n_cycles,
        n_ensemble=args.n_ensemble,
        finetune=args.finetune,
        finetune_start_step=args.finetune_start_step,
        finetune_lr_scale=args.finetune_lr_scale,
        latest_checkpoint_path=args.latest_checkpoint_path,
        best_checkpoint_path=args.best_checkpoint_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    return fit(
        model_config=load_model_config(args.model_config),
        data_config=data_config,
        training_config=training_config,
    )


if __name__ == "__main__":
    main()
