"""Overfit test on a single preprocessed OpenProteinSet chain — full pipeline.

Where :mod:`scripts.overfit_single_pdb` parses a single PDB and builds
*minimal* features by hand (no real MSA, no templates, no crops), this
script consumes the NPZ cache produced by
``scripts/preprocess_openproteinset.py`` so every training step sees the
same features the full AlphaFold2 training loop would see on this chain:

* Real UniRef90 MSA — sampled per step, block-deleted (Alg 1), clustered
  (supplement 1.2.7), BERT-masked (1.2.7).
* Real templates from pdb70 — up to ``--max-templates`` per step.
* Real supervision (rigid-group frames, torsions, atom37 masks) from the
  mmCIF ground truth, resolved via :mod:`minalphafold.geometry`.
* Random residue crops (supplement 1.2.8) when the chain is longer than
  ``--crop-size``.

Use this to pressure-test the training loop end-to-end on one chain. The
model should be able to drive loss + Cα RMSD close to zero: every
augmentation is stochastic, but there is only one underlying target.

Example:
    python scripts/overfit_processed_chain.py --chain-id 6m0j_E --steps 500

Output goes to ``artifacts/overfit_processed_chain/<chain_id>/``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from minalphafold.data import (
    ProcessedOpenProteinSetDataset,
    block_delete_msa,
    build_extra_msa_feat,
    build_msa_feat,
    build_supervision,
    build_target_feat,
    build_template_angle_feat,
    build_template_pair_feat,
    cluster_statistics,
    crop_example,
    hhblits_profile,
    masked_msa_inputs,
    sample_cluster_and_extra,
)
from minalphafold.losses import AlphaFoldLoss, select_best_atom14_ground_truth
from minalphafold.model import AlphaFold2
from minalphafold.pdbio import write_atom14_pdb, write_model_output_pdb
from minalphafold.trainer import (
    DataConfig,
    TrainingConfig,
    build_dataloader,
    build_optimizer,
    collapse_sampled_batch_tensor,
    list_available_profiles,
    load_model_config,
    loss_inputs_from_batch,
    model_inputs_from_batch,
    move_to_device,
    set_optimizer_learning_rate,
    set_seed,
    zero_dropout_model_config,
)


ARTIFACT_ROOT = ROOT / "artifacts" / "overfit_processed_chain"


def kabsch_align(pred: torch.Tensor, truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rigid-body alignment of ``pred`` onto ``truth`` (min-RMSD orthogonal Procrustes)."""
    pred_center = pred.mean(dim=0)
    truth_center = truth.mean(dim=0)
    covariance = (pred - pred_center).transpose(0, 1) @ (truth - truth_center)
    u, _, vh = torch.linalg.svd(covariance)
    rotation = vh.transpose(0, 1) @ u.transpose(0, 1)
    if torch.det(rotation) < 0:
        vh[-1] = -vh[-1]
        rotation = vh.transpose(0, 1) @ u.transpose(0, 1)
    translation = truth_center - pred_center @ rotation.transpose(0, 1)
    aligned = pred @ rotation.transpose(0, 1) + translation
    return aligned, rotation, translation


def rmsd(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(torch.sum((a - b) ** 2, dim=-1))).item())


def structure_metrics(outputs: dict, batch: dict) -> dict[str, float]:
    """Kabsch-aligned backbone / Cα / all-atom RMSDs + peptide-bond stats."""
    pred = outputs["atom14_coords"][0].detach().cpu()
    pred_mask = outputs["atom14_mask"][0].detach().cpu()
    truth = batch["true_atom_positions"][0].detach().cpu()
    truth_mask = batch["true_atom_mask"][0].detach().cpu()

    truth, truth_mask, _ = select_best_atom14_ground_truth(
        pred.unsqueeze(0),
        truth.unsqueeze(0),
        truth_mask.unsqueeze(0),
        batch["true_atom_positions_alt"][:1].detach().cpu(),
        batch["true_atom_mask_alt"][:1].detach().cpu(),
        batch["true_atom_is_ambiguous"][:1].detach().cpu(),
    )
    truth = truth[0]
    truth_mask = truth_mask[0]

    bb_atoms = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    bb_common = ((truth_mask > 0.5) & (pred_mask > 0.5))[:, bb_atoms]
    pred_bb = pred[:, bb_atoms][bb_common]
    truth_bb = truth[:, bb_atoms][bb_common]

    if pred_bb.shape[0] < 3:
        # Not enough backbone atoms to align — skip.
        return {
            "backbone_rmsd_after_alignment": float("nan"),
            "ca_rmsd_after_alignment": float("nan"),
            "all_atom_rmsd_after_alignment": float("nan"),
            "peptide_bond_mean": float("nan"),
            "peptide_bond_max": float("nan"),
            "peptide_bond_min": float("nan"),
        }

    _, R, t = kabsch_align(pred_bb, truth_bb)
    aligned = pred @ R.transpose(0, 1) + t

    ca_valid = truth_mask[:, 1] > 0.5
    common = (truth_mask > 0.5) & (pred_mask > 0.5)

    # C_i → N_{i+1} peptide-bond lengths (ideal 1.329 Å, PyMOL draws up to ~1.9 Å).
    c_i = pred[:-1, 2]
    n_next = pred[1:, 0]
    peptide_bonds = torch.linalg.norm(c_i - n_next, dim=-1)
    return {
        "backbone_rmsd_after_alignment": rmsd(aligned[:, bb_atoms][bb_common], truth_bb),
        "ca_rmsd_after_alignment": rmsd(aligned[ca_valid, 1], truth[ca_valid, 1]),
        "all_atom_rmsd_after_alignment": rmsd(aligned[common], truth[common]),
        "peptide_bond_mean": float(peptide_bonds.mean().item()),
        "peptide_bond_max": float(peptide_bonds.max().item()),
        "peptide_bond_min": float(peptide_bonds.min().item()),
    }


def apply_kabsch_to_outputs(outputs: dict, batch: dict) -> dict:
    """Return ``outputs`` with ``atom14_coords`` Kabsch-aligned to the ground truth."""
    pred = outputs["atom14_coords"][0].detach().cpu()
    pred_mask = outputs["atom14_mask"][0].detach().cpu()
    truth = batch["true_atom_positions"][0].detach().cpu()
    truth_mask = batch["true_atom_mask"][0].detach().cpu()

    truth, truth_mask, _ = select_best_atom14_ground_truth(
        pred.unsqueeze(0),
        truth.unsqueeze(0),
        truth_mask.unsqueeze(0),
        batch["true_atom_positions_alt"][:1].detach().cpu(),
        batch["true_atom_mask_alt"][:1].detach().cpu(),
        batch["true_atom_is_ambiguous"][:1].detach().cpu(),
    )
    bb_atoms = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    bb_common = ((truth_mask[0] > 0.5) & (pred_mask > 0.5))[:, bb_atoms]
    pred_bb = pred[:, bb_atoms][bb_common]
    truth_bb = truth[0, :, bb_atoms][bb_common]

    _, R, t = kabsch_align(pred_bb, truth_bb)
    aligned = pred @ R.transpose(0, 1) + t

    result = dict(outputs)
    result["atom14_coords"] = aligned.unsqueeze(0)
    return result


def write_pymol_script(path: Path, predicted_pdb: Path, truth_pdb: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f"load {truth_pdb}, truth",
                f"load {predicted_pdb}, predicted",
                "color grey60, truth",
                "color red, predicted",
                "align predicted, truth",
                "show cartoon",
                "bg_color white",
            ]
        )
        + "\n"
    )


def _cycle(iterable):
    """Endless iterator over the dataloader (one chain → cycles forever)."""
    return itertools.chain.from_iterable(itertools.repeat(iterable))


# ---------------------------------------------------------------------------
# Frozen-context path (--freeze-crop-and-cluster): compose the data.py
# primitives directly so the crop + cluster selection are fixed once at
# startup, while block-delete + MSA masking still fire fresh per step. The
# production pipeline (``collate_batch`` → ``build_msa_features``) is left
# untouched — this code is overfit-test-only scaffolding.
# ---------------------------------------------------------------------------


def _build_frozen_context(args, device: torch.device) -> dict[str, Any]:
    """Run the deterministic parts of the data pipeline once.

    Produces the cropped example (no crop, since we set ``crop_size >= N_res``),
    pre-sampled cluster / extra MSA (fixed seed), the MSA profile (over the
    full raw MSA per supplement 1.9.9), the template + target + supervision
    tensors, and every other field the batch needs that doesn't depend on
    block-delete / masking. Returned as a CPU-side dict; ``_assemble_step_batch``
    moves per-step tensors to ``device``.
    """
    dataset = ProcessedOpenProteinSetDataset(
        args.processed_features_dir,
        args.processed_labels_dir,
        split="train",
        val_fraction=0.0,
        seed=args.seed,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(
            "ProcessedOpenProteinSetDataset is empty — check --processed-features-dir."
        )
    chain_index = next(
        (i for i, cid in enumerate(dataset.chain_ids) if cid == args.chain_id),
        None,
    )
    if chain_index is None:
        raise FileNotFoundError(
            f"chain_id '{args.chain_id}' not found in {args.processed_features_dir}"
        )
    raw = dataset[chain_index]

    # ``training=False`` picks a deterministic centre crop (see
    # ``_crop_start``), so a fixed window is used every step — no per-step
    # randomness, but the crop is honoured when ``crop_size < n_res`` so
    # memory scales with ``args.crop_size`` instead of the raw chain length.
    cropped = crop_example(
        raw,
        crop_size=args.crop_size,
        training=False,
        torch_generator=None,
    )
    # Downstream features (``seq_mask``, ``residue_index`` fallback, reported
    # ``n_res``) must be built against the *cropped* length — otherwise masks
    # and indices retain the raw chain's shape and blow up at the first
    # ``pair_mask`` broadcast in the model.
    n_res = int(cropped["aatype"].shape[0])

    # Fixed-seed cluster / extra sampling.
    cluster_generator = torch.Generator()
    cluster_generator.manual_seed(args.seed)
    cluster_msa, cluster_deletions, extra_msa, extra_deletions = sample_cluster_and_extra(
        cropped["msa"],
        cropped["deletions"],
        msa_depth=args.msa_depth,
        extra_msa_depth=args.extra_msa_depth,
        training=True,
        torch_generator=cluster_generator,
        python_random=random.Random(args.seed),
    )

    # MSA profile computed from the raw (pre-sampling) MSA — matches what
    # build_msa_features does in production and keeps the masked-MSA
    # replacement distribution consistent.
    msa_profile = hhblits_profile(cropped["msa"])

    # Templates (capped at ``max_templates``) — shapes ``(T, N_res, ...)``.
    template_aatype = cropped["template_aatype"][: args.max_templates]
    template_positions = cropped["template_atom14_positions"][: args.max_templates]
    template_atom14_mask = cropped["template_atom14_mask"][: args.max_templates]
    template_residue_mask = template_atom14_mask.amax(dim=-1)
    template_mask = (template_residue_mask.sum(dim=-1) > 0).float()

    # Static residue / target / supervision features.
    target_feat = build_target_feat(
        cropped["aatype"],
        cropped.get("between_segment_residues"),
    )
    template_pair_feat = build_template_pair_feat(
        template_aatype, template_positions, template_atom14_mask,
    )
    template_angle_feat = build_template_angle_feat(
        template_aatype, template_positions, template_atom14_mask,
    )
    supervision = build_supervision(
        cropped["aatype"],
        cropped["atom14_positions"],
        cropped["atom14_mask"],
    )

    residue_index = cropped.get("residue_index")
    if residue_index is None:
        residue_index = torch.arange(n_res, dtype=torch.long)
    else:
        residue_index = residue_index.long()

    resolution = cropped.get("resolution", torch.tensor(0.0))
    if not torch.is_tensor(resolution):
        resolution = torch.as_tensor(resolution)
    resolution = resolution.float()

    return {
        "chain_id": cropped["chain_id"],
        "aatype": cropped["aatype"],
        "residue_index": residue_index,
        "target_feat": target_feat,
        "template_pair_feat": template_pair_feat,
        "template_angle_feat": template_angle_feat,
        "template_mask": template_mask,
        "template_residue_mask": template_residue_mask,
        "seq_mask": torch.ones(n_res, dtype=torch.float32),
        "resolution": resolution,
        # Per-step inputs (frozen cluster / extra MSA and profile).
        "cluster_msa_frozen": cluster_msa,
        "cluster_deletions_frozen": cluster_deletions,
        "extra_msa_frozen": extra_msa,
        "extra_deletions_frozen": extra_deletions,
        "msa_profile": msa_profile,
        "n_res": n_res,
        **supervision,
    }


_SUPERVISION_KEYS = (
    "true_rotations",
    "true_translations",
    "true_atom_positions",
    "true_atom_mask",
    "true_atom_positions_alt",
    "true_atom_mask_alt",
    "true_atom_is_ambiguous",
    "true_torsion_angles",
    "true_torsion_angles_alt",
    "true_torsion_mask",
    "true_rigid_group_frames_R",
    "true_rigid_group_frames_t",
    "true_rigid_group_frames_R_alt",
    "true_rigid_group_frames_t_alt",
    "true_rigid_group_exists",
    "atom37_exists",
    "experimentally_resolved_true",
    "res_types",
    "backbone_mask",
    "pseudo_beta_mask",
    "pseudo_beta_positions",
)


def _assemble_step_batch(frozen: dict[str, Any], args, device: torch.device) -> dict[str, Any]:
    """Compose a training batch from the frozen context + fresh per-step noise.

    Per step we apply block-delete to the frozen cluster MSA and BERT-masking
    to the survivors, using Python's system random state (``torch_generator=
    None`` / no seed) so the noise varies every call. Everything else — crop,
    which rows are cluster vs extra, template features, supervision — is
    pulled from ``frozen`` unchanged.
    """
    cluster_msa, cluster_deletions = block_delete_msa(
        frozen["cluster_msa_frozen"],
        frozen["cluster_deletions_frozen"],
        training=True,
        enabled=not args.disable_msa_augmentation,
        msa_fraction_per_block=args.block_delete_msa_fraction,
        randomize_num_blocks=False,
        num_blocks=args.block_delete_msa_num_blocks,
        torch_generator=None,
    )

    mask_prob = 0.0 if args.disable_msa_augmentation else args.masked_msa_probability
    masked_cluster_msa, masked_msa_target, masked_msa_mask = masked_msa_inputs(
        cluster_msa,
        frozen["msa_profile"],
        training=True,
        mask_probability=mask_prob,
        torch_generator=None,
    )

    cluster_profile, cluster_deletion_mean = cluster_statistics(
        masked_cluster_msa,
        cluster_deletions,
        frozen["extra_msa_frozen"],
        frozen["extra_deletions_frozen"],
    )
    msa_feat = build_msa_feat(
        masked_cluster_msa, cluster_deletions, cluster_profile, cluster_deletion_mean,
    )
    extra_msa_feat = build_extra_msa_feat(
        frozen["extra_msa_frozen"], frozen["extra_deletions_frozen"],
    )

    def _b(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0).to(device)

    batch: dict[str, Any] = {
        "chain_id": [frozen["chain_id"]],
        "aatype": _b(frozen["aatype"]),
        "residue_index": _b(frozen["residue_index"]),
        "target_feat": _b(frozen["target_feat"]),
        "template_aatype": _b(torch.zeros(
            (0, frozen["n_res"]), dtype=torch.long,
        )) if frozen["template_pair_feat"].shape[0] == 0
        else _b(torch.zeros(
            (frozen["template_pair_feat"].shape[0], frozen["n_res"]), dtype=torch.long,
        )),
        "template_pair_feat": _b(frozen["template_pair_feat"]),
        "template_angle_feat": _b(frozen["template_angle_feat"]),
        "template_mask": _b(frozen["template_mask"]),
        "template_residue_mask": _b(frozen["template_residue_mask"]),
        "seq_mask": _b(frozen["seq_mask"]),
        "resolution": _b(frozen["resolution"]),
        "msa_feat": _b(msa_feat),
        "extra_msa_feat": _b(extra_msa_feat),
        "msa_mask": _b(torch.ones(masked_cluster_msa.shape, dtype=torch.float32)),
        "extra_msa_mask": _b(torch.ones(frozen["extra_msa_frozen"].shape, dtype=torch.float32)),
        "masked_msa_target": _b(masked_msa_target),
        "masked_msa_mask": _b(masked_msa_mask.float()),
    }
    for key in _SUPERVISION_KEYS:
        batch[key] = _b(frozen[key])
    return batch


def _evaluate_with_known_ground_truth(
    model: AlphaFold2,
    batch: dict,
    training_config: TrainingConfig,
    loss_fn: AlphaFoldLoss,
) -> tuple[dict, dict, dict[str, float]]:
    """Single forward pass (train mode off) — returns outputs, RMSDs, per-term losses."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(**model_inputs_from_batch(batch, training_config))
        per_example_total, loss_terms = loss_fn(
            return_breakdown=True, **loss_inputs_from_batch(batch, outputs),
        )
    if was_training:
        model.train()
    metrics = structure_metrics(outputs, batch)
    term_values = {
        key: float(value.mean().item()) if torch.is_tensor(value) else float(value)
        for key, value in loss_terms.items()
    }
    term_values["loss"] = float(per_example_total.mean().item())
    return outputs, metrics, term_values


# Ordered (key, label) so the console table has a stable column layout.
_LOSS_BREAKDOWN_LAYOUT: tuple[tuple[str, str], ...] = (
    ("loss", "total"),
    ("fape_loss", "fape"),
    ("backbone_loss", "bb_fape"),
    ("sidechain_fape_loss", "sc_fape"),
    ("torsion_loss", "tors"),
    ("distogram_loss", "dist"),
    ("msa_loss", "msa"),
    ("plddt_loss", "plddt"),
)
_VIOLATION_KEYS: tuple[tuple[str, str], ...] = (
    ("structural_violation_loss", "viol"),
    ("experimentally_resolved_loss", "exp"),
    ("tm_score_loss", "tm"),
)


def _format_loss_breakdown(term_values: dict[str, float]) -> str:
    """One-line loss breakdown for the console log — unweighted per-term values."""
    parts = []
    for key, label in _LOSS_BREAKDOWN_LAYOUT:
        if key in term_values:
            parts.append(f"{label}={term_values[key]:6.3f}")
    for key, label in _VIOLATION_KEYS:
        if key in term_values:
            parts.append(f"{label}={term_values[key]:6.3f}")
    return "  ".join(parts)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-id", type=str, required=True,
                        help="Chain ID present in the processed features/labels dirs, e.g. '6m0j_E'.")
    parser.add_argument("--processed-features-dir", type=str, default="data/processed_features")
    parser.add_argument("--processed-labels-dir", type=str, default="data/processed_labels")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=0.1)
    parser.add_argument(
        "--model-profile",
        type=str,
        default="medium",
        help=(
            "Profile name resolved under configs/ (available: "
            f"{', '.join(list_available_profiles())}) or a path to any "
            "TOML file with the same schema."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-cycles", type=int, default=1,
                        help="Number of recycling iterations (paper uses 4).")
    parser.add_argument("--n-ensemble", type=int, default=1,
                        help="Number of ensemble samples (inference only in the paper).")
    parser.add_argument("--crop-size", type=int, default=256,
                        help="Residue crop size per step (supplement 1.2.8). "
                             "Paper uses 256 for initial training, 384 for fine-tuning.")
    parser.add_argument("--msa-depth", type=int, default=128,
                        help="MSA cluster depth (paper: 128 initial / 512 fine-tune).")
    parser.add_argument("--extra-msa-depth", type=int, default=256,
                        help="Extra MSA depth (paper: 1024 initial / 5120 fine-tune). "
                             "Reduced default for CPU overfit feasibility.")
    parser.add_argument("--max-templates", type=int, default=1,
                        help="Templates per step (paper: 4).")
    parser.add_argument("--disable-msa-augmentation", action="store_true",
                        help="Turn off block-deletion + BERT masking (keeps crops + clustering).")
    parser.add_argument(
        "--freeze-crop-and-cluster",
        action="store_true",
        help="Compose the data-pipeline primitives at the overfit-script "
             "level so the crop and MSA cluster/extra split are fixed once "
             "at startup (same rows every step), while block-deletion and "
             "BERT MSA masking still run fresh per step. Use this for "
             "single-protein overfit; leave off for production-faithful "
             "training. The core pipeline (collate_batch / build_msa_features) "
             "is untouched regardless of this flag.",
    )
    parser.add_argument("--block-delete-msa-fraction", type=float, default=0.3,
                        help="Fraction of the MSA removed per block (supplement 1.2.6 default).")
    parser.add_argument("--block-delete-msa-num-blocks", type=int, default=5,
                        help="Number of deleted blocks per call (supplement 1.2.6 default).")
    parser.add_argument("--masked-msa-probability", type=float, default=0.15,
                        help="BERT-style MSA masking rate (supplement 1.2.7).")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=25,
                        help="Run a no-augmentation forward + compute RMSD every N steps.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--use-clamped-fape", type=float, default=None,
                        help="Mix-weight of clamped FAPE; None = fully clamped (supplement 1.11.5).")
    parser.add_argument("--violations-after-step", type=int, default=None,
                        help=(
                            "Step after which to enable the fine-tuning loss terms "
                            "(structural-violation loss, weight 1.0, supplement 1.9.11 eq 44-47; "
                            "experimentally-resolved loss, weight 0.01, supplement 1.9.10). "
                            "OFF by default — supplement 1.9.11 is explicit: 'We apply this violation "
                            "loss only during the fine-tuning training phase. Switching it on in the "
                            "early training leads to strong instabilities in the training dynamics.' "
                            "Paper's Table 4 starts fine-tuning at ~87%% of total samples "
                            "(10M initial / 11.5M total); for an N-step overfit, ~0.8*N is a "
                            "reasonable paper-faithful default if you opt in."
                        ))
    parser.add_argument("--fine-tune-lr-scale", type=float, default=0.5,
                        help=(
                            "LR multiplier applied when `--violations-after-step` fires. "
                            "Default 0.5 mirrors Table 4: initial LR 1e-3 -> fine-tune LR 5e-4."
                        ))
    args = parser.parse_args(argv)

    out_dir = args.out_dir or (ARTIFACT_ROOT / args.chain_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device)

    # ------------------------------------------------------------
    # Dataset: point at the single-chain cache. val_fraction=0.0 keeps
    # every chain in the "train" split; since there's only one chain,
    # the dataloader yields it every iteration (with fresh augmentation
    # per step, matching supplement 1.2).
    # ------------------------------------------------------------
    features_dir = Path(args.processed_features_dir)
    labels_dir = Path(args.processed_labels_dir)
    feature_cache = features_dir / f"{args.chain_id}.npz"
    label_cache = labels_dir / f"{args.chain_id}.npz"
    if not feature_cache.exists() or not label_cache.exists():
        raise FileNotFoundError(
            f"Processed cache not found. Expected:\n"
            f"  {feature_cache}\n  {label_cache}\n"
            "Run scripts/preprocess_openproteinset.py first."
        )

    data_config = DataConfig(
        processed_features_dir=str(features_dir),
        processed_labels_dir=str(labels_dir),
        val_fraction=0.0,
        crop_size=args.crop_size,
        msa_depth=args.msa_depth,
        extra_msa_depth=args.extra_msa_depth,
        max_templates=args.max_templates,
        block_delete_training_msa=not args.disable_msa_augmentation,
        masked_msa_probability=(
            0.0 if args.disable_msa_augmentation else args.masked_msa_probability
        ),
        block_delete_msa_fraction=args.block_delete_msa_fraction,
        block_delete_msa_num_blocks=args.block_delete_msa_num_blocks,
    )

    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        grad_clip_norm=args.grad_clip_norm,
        device=str(device),
        seed=args.seed,
        n_cycles=args.n_cycles,
        n_ensemble=args.n_ensemble,
    )

    # ------------------------------------------------------------
    # Model: dropout off so overfitting is possible; profile loaded
    # from configs/ (same JSON files the trainer uses).
    # ------------------------------------------------------------
    model_config = zero_dropout_model_config(load_model_config(args.model_profile))
    model = AlphaFold2(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[overfit] chain       : {args.chain_id}")
    print(f"[overfit] profile     : {args.model_profile} ({n_params / 1e6:.1f}M params)")
    print(f"[overfit] crop_size   : {args.crop_size}")
    print(f"[overfit] msa_depth   : {args.msa_depth} (extra {args.extra_msa_depth}), "
          f"templates: {args.max_templates}")
    print(f"[overfit] device      : {device}")
    print(f"[overfit] msa_aug     : {'ON' if not args.disable_msa_augmentation else 'OFF'}")

    # ------------------------------------------------------------
    # Training infrastructure.
    # ------------------------------------------------------------
    if args.freeze_crop_and_cluster:
        frozen = _build_frozen_context(args, device)
        print(f"[overfit] frozen clusters: {frozen['cluster_msa_frozen'].shape[0]} rows")
        print(f"[overfit] frozen extras  : {frozen['extra_msa_frozen'].shape[0]} rows")
        print(f"[overfit] frozen crop    : {frozen['n_res']} residues (deterministic centre crop, no per-step randomness)")

        def _next_batch() -> dict:
            return _assemble_step_batch(frozen, args, device)

        eval_batch = _assemble_step_batch(frozen, argparse.Namespace(**{
            **vars(args), "disable_msa_augmentation": True,
        }), device)
    else:
        train_loader = build_dataloader(
            "train",
            data_config,
            training=True,
            batch_size=1,
            num_workers=0,
            device=str(device),
            seed=args.seed,
            n_cycles=args.n_cycles,
            n_ensemble=args.n_ensemble,
        )

        # Separate deterministic loader — no augmentation, fixed crop — for
        # periodic RMSD eval so metrics aren't swamped by MSA-sampling noise.
        eval_data_config = DataConfig(
            processed_features_dir=str(features_dir),
            processed_labels_dir=str(labels_dir),
            val_fraction=0.0,
            crop_size=args.crop_size,
            msa_depth=args.msa_depth,
            extra_msa_depth=args.extra_msa_depth,
            max_templates=args.max_templates,
            block_delete_training_msa=False,
            masked_msa_probability=0.0,
            fixed_feature_seed=args.seed,
        )
        eval_loader = build_dataloader(
            "train",
            eval_data_config,
            training=False,
            batch_size=1,
            num_workers=0,
            device=str(device),
            seed=args.seed,
            n_cycles=args.n_cycles,
            n_ensemble=args.n_ensemble,
        )
        batch_iter = _cycle(train_loader)

        def _next_batch() -> dict:
            return move_to_device(next(batch_iter), device)

        eval_batch = move_to_device(next(iter(eval_loader)), device)

    optimizer = build_optimizer(model, training_config)
    # Two-stage training mirroring supplement 1.11.1 / Table 4: `finetune=False`
    # during initial training (violation weight 0.0, exp-resolved weight 0.0),
    # flipped to True at `--violations-after-step` to enable the fine-tuning
    # loss terms and halve the LR (Table 4: 1e-3 -> 5e-4).
    loss_fn = AlphaFoldLoss(finetune=False, use_clamped_fape=args.use_clamped_fape).to(device)
    set_optimizer_learning_rate(optimizer, args.learning_rate)
    model.train()

    history: list[dict] = []
    best = {"ca_rmsd_after_alignment": float("inf")}
    best_state: dict | None = None
    start_time = time.time()

    for step in range(1, args.steps + 1):
        if (
            args.violations_after_step is not None
            and not loss_fn.finetune
            and step > args.violations_after_step
        ):
            loss_fn.finetune = True
            new_lr = args.learning_rate * args.fine_tune_lr_scale
            set_optimizer_learning_rate(optimizer, new_lr)
            print(
                f"[overfit] step {step}: entering fine-tuning stage "
                f"(supplement 1.11.1) — enabling structural-violation + "
                f"experimentally-resolved losses, LR {args.learning_rate:.2e} -> {new_lr:.2e}"
            )

        batch = _next_batch()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**model_inputs_from_batch(batch, training_config))
        per_example_loss = loss_fn(**loss_inputs_from_batch(batch, outputs))
        loss = per_example_loss.mean()
        loss.backward()
        if training_config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip_norm)
        optimizer.step()

        loss_value = float(loss.item())
        entry = {"step": step, "loss": loss_value}

        if step % args.log_every == 0 or step == args.steps or step == 1:
            print(f"[overfit] step {step:5d}/{args.steps}  loss={loss_value:.4f}  "
                  f"({time.time() - start_time:.0f}s)")

        if step % args.eval_every == 0 or step == args.steps or step == 1:
            _, metrics, term_values = _evaluate_with_known_ground_truth(
                model, eval_batch, training_config, loss_fn,
            )
            eval_loss = term_values["loss"]
            entry["eval_loss"] = eval_loss
            entry.update(metrics)
            entry["eval_loss_terms"] = term_values
            print(
                f"[overfit] eval {step:5d}/{args.steps}  "
                f"bb_rmsd={metrics['backbone_rmsd_after_alignment']:.3f}  "
                f"ca_rmsd={metrics['ca_rmsd_after_alignment']:.3f}  "
                f"aa_rmsd={metrics['all_atom_rmsd_after_alignment']:.3f}  "
                f"pep={metrics['peptide_bond_mean']:.2f}"
                f"[{metrics['peptide_bond_min']:.2f},{metrics['peptide_bond_max']:.2f}]"
            )
            print(f"[overfit] losses {step:4d}/{args.steps}  {_format_loss_breakdown(term_values)}")
            ca_rmsd = metrics["ca_rmsd_after_alignment"]
            if not np.isnan(ca_rmsd) and ca_rmsd < best["ca_rmsd_after_alignment"]:
                best = {"step": step, "loss": loss_value, "eval_loss": eval_loss, **metrics}
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        history.append(entry)

    # ---- Final artefacts ----
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[overfit] restoring best-by-ca-rmsd checkpoint: step {best['step']}")

    _, final_metrics, final_loss_terms = _evaluate_with_known_ground_truth(
        model, eval_batch, training_config, loss_fn,
    )
    with torch.no_grad():
        model.eval()
        final_outputs = model(**model_inputs_from_batch(eval_batch, training_config))
    aligned = apply_kabsch_to_outputs(final_outputs, eval_batch)
    print(f"[overfit] final losses: {_format_loss_breakdown(final_loss_terms)}")

    predicted_pdb = out_dir / f"predicted_{args.chain_id}.pdb"
    truth_pdb = out_dir / f"ground_truth_{args.chain_id}.pdb"
    write_model_output_pdb(predicted_pdb, aligned, eval_batch, example_index=0)
    write_atom14_pdb(
        truth_pdb,
        eval_batch["aatype"][0].detach().cpu(),
        eval_batch["true_atom_positions"][0].detach().cpu(),
        eval_batch["true_atom_mask"][0].detach().cpu(),
        residue_index=eval_batch["residue_index"][0].detach().cpu(),
    )
    write_pymol_script(out_dir / "view_in_pymol.pml", predicted_pdb, truth_pdb)

    metrics_payload = {
        "chain_id": args.chain_id,
        "steps": args.steps,
        "model_profile": args.model_profile,
        "crop_size": args.crop_size,
        "msa_depth": args.msa_depth,
        "extra_msa_depth": args.extra_msa_depth,
        "max_templates": args.max_templates,
        "msa_augmentation": not args.disable_msa_augmentation,
        "freeze_crop_and_cluster": args.freeze_crop_and_cluster,
        "final_metrics": final_metrics,
        "final_loss_terms": final_loss_terms,
        "best": best,
        "predicted_pdb": str(predicted_pdb),
        "ground_truth_pdb": str(truth_pdb),
        "pymol_script": str(out_dir / "view_in_pymol.pml"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, default=str))
    (out_dir / "losses.json").write_text(json.dumps(history, indent=2, default=str))

    print()
    print(f"[overfit] final metrics: {final_metrics}")
    print(f"[overfit] best step    : {best.get('step', 'n/a')}")
    print(f"[overfit] artefacts    : {out_dir}")
    print(f"[overfit] view in PyMOL: pymol {out_dir / 'view_in_pymol.pml'}")
    return metrics_payload


if __name__ == "__main__":
    main()
