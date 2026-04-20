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
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from minalphafold.losses import AlphaFoldLoss, select_best_atom14_ground_truth
from minalphafold.model import AlphaFold2
from minalphafold.pdbio import write_atom14_pdb, write_model_output_pdb
from minalphafold.trainer import (
    DataConfig,
    TrainingConfig,
    alphafold2_model_config,
    build_dataloader,
    build_optimizer,
    collapse_sampled_batch_tensor,
    default_model_config,
    loss_inputs_from_batch,
    medium_model_config,
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


def _evaluate_with_known_ground_truth(
    model: AlphaFold2,
    batch: dict,
    training_config: TrainingConfig,
    loss_fn: AlphaFoldLoss,
) -> tuple[dict, dict, float]:
    """Single forward pass (train mode off) — for periodic metrics."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(**model_inputs_from_batch(batch, training_config))
        per_example_loss = loss_fn(**loss_inputs_from_batch(batch, outputs))
    if was_training:
        model.train()
    metrics = structure_metrics(outputs, batch)
    return outputs, metrics, float(per_example_loss.mean().item())


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
        choices=["tiny", "medium", "alphafold2"],
        default="medium",
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
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=25,
                        help="Run a no-augmentation forward + compute RMSD every N steps.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--use-clamped-fape", type=float, default=None,
                        help="Mix-weight of clamped FAPE; None = fully clamped (supplement 1.11.5).")
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
        masked_msa_probability=0.0 if args.disable_msa_augmentation else 0.15,
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
    # Model: dropout off so overfitting is possible; same config
    # profiles as the trainer.
    # ------------------------------------------------------------
    profile_builders = {
        "tiny": default_model_config,
        "medium": medium_model_config,
        "alphafold2": alphafold2_model_config,
    }
    model_config = zero_dropout_model_config(profile_builders[args.model_profile]())
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

    optimizer = build_optimizer(model, training_config)
    loss_fn = AlphaFoldLoss(finetune=False, use_clamped_fape=args.use_clamped_fape).to(device)
    set_optimizer_learning_rate(optimizer, args.learning_rate)
    model.train()

    batch_iter = _cycle(train_loader)
    eval_batch = move_to_device(next(iter(eval_loader)), device)

    history: list[dict] = []
    best = {"ca_rmsd_after_alignment": float("inf")}
    best_state: dict | None = None
    start_time = time.time()

    for step in range(1, args.steps + 1):
        batch = move_to_device(next(batch_iter), device)

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
            _, metrics, eval_loss = _evaluate_with_known_ground_truth(
                model, eval_batch, training_config, loss_fn,
            )
            entry["eval_loss"] = eval_loss
            entry.update(metrics)
            print(
                f"[overfit] eval {step:5d}/{args.steps}  "
                f"eval_loss={eval_loss:.4f}  "
                f"bb_rmsd={metrics['backbone_rmsd_after_alignment']:.3f}  "
                f"ca_rmsd={metrics['ca_rmsd_after_alignment']:.3f}  "
                f"aa_rmsd={metrics['all_atom_rmsd_after_alignment']:.3f}  "
                f"pep={metrics['peptide_bond_mean']:.2f}"
                f"[{metrics['peptide_bond_min']:.2f},{metrics['peptide_bond_max']:.2f}]"
            )
            ca_rmsd = metrics["ca_rmsd_after_alignment"]
            if not np.isnan(ca_rmsd) and ca_rmsd < best["ca_rmsd_after_alignment"]:
                best = {"step": step, "loss": loss_value, "eval_loss": eval_loss, **metrics}
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        history.append(entry)

    # ---- Final artefacts ----
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[overfit] restoring best-by-ca-rmsd checkpoint: step {best['step']}")

    model.eval()
    with torch.no_grad():
        final_outputs = model(**model_inputs_from_batch(eval_batch, training_config))
    final_metrics = structure_metrics(final_outputs, eval_batch)
    aligned = apply_kabsch_to_outputs(final_outputs, eval_batch)

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
        "final_metrics": final_metrics,
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
