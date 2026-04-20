"""Minimal self-contained single-protein overfit driver.

Given one ground-truth PDB file, builds the full set of features a single
``AlphaFold2.forward`` call needs (MSA = query-only, no templates, real
atom14 coordinates from the PDB), then runs a plain training loop until
the loss plateaus. Reports training loss, backbone/Cα/all-atom RMSD after
Kabsch alignment, and writes a predicted PDB alongside the ground truth
for PyMOL viewing.

This is the "does the pipeline actually learn geometry?" test. It does
NOT touch OpenProteinSet, MSAs from JackHMMER/HHblits, templates, or any
of the production data caching — those all live in ``autoresearch_overfit.py``
for running many experiments at scale. Use this script to verify end-to-end
behaviour on a single PDB in under a minute on CPU.

Example:
    python scripts/overfit_single_pdb.py \\
        --pdb artifacts/overfit_1a0m_A/ground_truth_1a0m_A.pdb \\
        --steps 1000

Output goes to ``artifacts/overfit_single_pdb/<chain_id>/`` — prediction
PDB, ground-truth PDB (copied), PyMOL view script, metrics JSON, per-step
loss log.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minalphafold"))


from a3m import MASK_ID, MSA_ALPHABET_SIZE, sequence_to_ids
from data import build_processed_example_from_cropped, collate_batch
from losses import AlphaFoldLoss, select_best_atom14_ground_truth
from model import AlphaFold2
from pdbio import write_atom14_pdb, write_model_output_pdb
from residue_constants import restype_1to3, restype_name_to_atom14_names, restypes
from trainer import (
    alphafold2_model_config,
    build_optimizer,
    default_model_config,
    medium_model_config,
    model_inputs_from_batch,
    move_to_device,
    set_seed,
    set_optimizer_learning_rate,
    zero_dropout_model_config,
    TrainingConfig,
)


ARTIFACT_ROOT = ROOT / "artifacts" / "overfit_single_pdb"


def parse_pdb(path: Path) -> dict:
    """Parse a PDB file into the fields ``build_processed_example_from_cropped`` expects."""
    three_to_one = {restype_1to3[r]: r for r in restypes}

    residues: list[tuple[int, str, dict[str, tuple[float, float, float]]]] = []
    current = None
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        resnum = int(line[22:26])
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        if current is None or current[0] != resnum:
            current = (resnum, resname, {})
            residues.append(current)
        current[2][atom_name] = (x, y, z)

    if not residues:
        raise ValueError(f"No ATOM records found in {path}")

    sequence = "".join(three_to_one.get(r[1], "X") for r in residues)
    n_res = len(residues)
    aatype = torch.from_numpy(sequence_to_ids(sequence)).long()

    atom14_positions = torch.zeros((n_res, 14, 3), dtype=torch.float32)
    atom14_mask = torch.zeros((n_res, 14), dtype=torch.float32)
    for i, (_, resname_3, atoms) in enumerate(residues):
        slot_names = restype_name_to_atom14_names.get(resname_3, [])
        for slot, name in enumerate(slot_names):
            if name and name in atoms:
                atom14_positions[i, slot] = torch.tensor(atoms[name], dtype=torch.float32)
                atom14_mask[i, slot] = 1.0

    return {
        "sequence": sequence,
        "aatype": aatype,
        "atom14_positions": atom14_positions,
        "atom14_mask": atom14_mask,
    }


def build_minimal_example(
    chain_id: str,
    parsed: dict,
    *,
    resolution: float = 2.0,
) -> dict:
    """Assemble the single-example dict that ``collate_batch`` consumes.

    MSA contains only the query sequence (``N_seq = 1``) and no templates,
    so the model trains purely on its own target-feat + query MSA.
    """
    aatype = parsed["aatype"]
    n_res = aatype.shape[0]

    msa = aatype.unsqueeze(0).clone()  # (1, N_res) — one row: the query
    deletions = torch.zeros((1, n_res), dtype=torch.long)

    return {
        "chain_id": chain_id,
        "aatype": aatype,
        "msa": msa,
        "deletions": deletions,
        "between_segment_residues": torch.zeros(n_res, dtype=torch.long),
        "residue_index": torch.arange(n_res, dtype=torch.long),
        "template_aatype": torch.zeros((0, n_res), dtype=torch.long),
        "template_atom14_positions": torch.zeros((0, n_res, 14, 3), dtype=torch.float32),
        "template_atom14_mask": torch.zeros((0, n_res, 14), dtype=torch.float32),
        "atom14_positions": parsed["atom14_positions"],
        "atom14_mask": parsed["atom14_mask"],
        "resolution": torch.tensor(resolution, dtype=torch.float32),
        "crop_start": 0,
    }


def loss_inputs_from_batch(batch: dict, outputs: dict) -> dict:
    """Shallow copy of ``trainer.loss_inputs_from_batch`` — no cycle/ensemble dim stripping."""
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
        "resolution": batch["resolution"],
        "masked_msa_pred": outputs["masked_msa_logits"],
        "masked_msa_target": batch["masked_msa_target"],
        "masked_msa_mask": batch["masked_msa_mask"],
        "plddt_pred": outputs["plddt_logits"],
        "distogram_pred": outputs["distogram_logits"],
        "tm_pred": outputs["tm_logits"],
        "res_types": batch["res_types"],
        "residue_index": batch["residue_index"],
        "seq_mask": batch["seq_mask"],
    }


def kabsch_align(pred: torch.Tensor, truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    """Kabsch-aligned backbone / Cα / all-atom RMSDs, plus peptide-bond stats."""
    pred = outputs["atom14_coords"][0].detach().cpu()
    pred_mask = outputs["atom14_mask"][0].detach().cpu()
    truth = batch["true_atom_positions"][0].detach().cpu()
    truth_mask = batch["true_atom_mask"][0].detach().cpu()

    # Pick the symmetry-renamed ground truth so FAPE-style ambiguity doesn't
    # double-count mis-aligned chi flips.
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

    _, R, t = kabsch_align(pred_bb, truth_bb)
    aligned = pred @ R.transpose(0, 1) + t

    ca_valid = truth_mask[:, 1] > 0.5
    common = (truth_mask > 0.5) & (pred_mask > 0.5)

    # C_i → N_{i+1} peptide bonds. Ideal length is 1.329 Å (1.341 Å for Xaa→Pro).
    # We only look at the raw prediction — Kabsch alignment is a rigid motion so
    # it does not change bond lengths, but reading from `pred` keeps the intent
    # obvious.
    c_i = pred[:-1, 2]  # C of residue i
    n_next = pred[1:, 0]  # N of residue i+1
    peptide_bonds = torch.linalg.norm(c_i - n_next, dim=-1)
    return {
        "backbone_rmsd_after_alignment": rmsd(aligned[:, bb_atoms][bb_common], truth_bb),
        "ca_rmsd_after_alignment": rmsd(aligned[ca_valid, 1], truth[ca_valid, 1]),
        "all_atom_rmsd_after_alignment": rmsd(aligned[common], truth[common]),
        "peptide_bond_mean": float(peptide_bonds.mean().item()),
        "peptide_bond_max": float(peptide_bonds.max().item()),
        "peptide_bond_min": float(peptide_bonds.min().item()),
    }


def apply_kabsch_alignment_to_outputs(outputs: dict, batch: dict) -> dict:
    """Return a copy of ``outputs`` with ``atom14_coords`` Kabsch-aligned to the ground truth.

    Writing the aligned structure to disk (instead of the raw model output)
    makes the predicted / ground-truth pair overlay nicely in PyMOL without
    the viewer having to re-align them.
    """
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


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True, help="Ground-truth PDB file")
    parser.add_argument("--chain-id", type=str, default=None, help="Name to use for artefacts (default: PDB stem)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of optimiser steps")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=0.1)
    parser.add_argument(
        "--model-profile",
        choices=["tiny", "medium", "alphafold2"],
        default="medium",
        help="tiny (fast smoke test) / medium (good default for <50 residues on CPU) / alphafold2 (full-size)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-cycles", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--use-clamped-fape", type=float, default=None,
                        help="Mix-weight of clamped FAPE; None = fully clamped (supplement 1.11.5).")
    parser.add_argument("--violations-after-step", type=int, default=None,
                        help="Enable the structural-violation loss (supplement 1.9.11, eqs 44-47) "
                             "after this step. OFF by default — the supplement is explicit on "
                             "page 40: 'We apply this violation loss only during the fine-tuning "
                             "training phase. Switching it on in the early training leads to "
                             "strong instabilities in the training dynamics.' AF2's fine-tuning "
                             "(table 4) starts from fully-converged initial-training weights "
                             "(~10^7 samples) with the LR halved; our ~10^3-step overfit matches "
                             "neither condition. For single-protein overfit, ~1000 FAPE-only "
                             "steps is enough to drive peptide bonds into the 0.9-1.5 Å range "
                             "that PyMOL will draw. Pass a step number here only if you want to "
                             "reproduce the instability the supplement documents.")
    args = parser.parse_args(argv)

    chain_id = args.chain_id or args.pdb.stem
    out_dir = args.out_dir or (ARTIFACT_ROOT / chain_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"[overfit] parsing PDB: {args.pdb}")
    parsed = parse_pdb(args.pdb)
    n_res = parsed["aatype"].shape[0]
    print(f"[overfit] sequence ({n_res} residues): {parsed['sequence']}")

    example = build_minimal_example(chain_id, parsed, resolution=args.resolution)

    # Build a batch via the same collation path the regular training loop uses.
    # crop_size = n_res means no cropping; msa_depth = 1 keeps the single query row;
    # extra_msa_depth = 0 skips the extra MSA stack entirely.
    batch = collate_batch(
        [example],
        crop_size=n_res,
        msa_depth=1,
        extra_msa_depth=0,
        max_templates=0,
        training=True,
        block_delete_training_msa=False,
        masked_msa_probability=0.0,
        random_seed=args.seed,
        num_recycling_samples=1,
        num_ensemble_samples=1,
    )
    batch = move_to_device(batch, device)

    profile_builders = {
        "tiny": default_model_config,
        "medium": medium_model_config,
        "alphafold2": alphafold2_model_config,
    }
    model_config = profile_builders[args.model_profile]()
    model_config = zero_dropout_model_config(model_config)
    model = AlphaFold2(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[overfit] model profile: {args.model_profile} ({n_params / 1e6:.1f}M params)")

    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        grad_clip_norm=args.grad_clip_norm,
        device=str(device),
        seed=args.seed,
        n_cycles=args.n_cycles,
        n_ensemble=1,
    )
    optimizer = build_optimizer(model, training_config)
    # finetune is toggled on in the training loop at `violations_after_step`.
    loss_fn = AlphaFoldLoss(finetune=False, use_clamped_fape=args.use_clamped_fape).to(device)
    violations_enabled_at_step: int | None = None

    model.train()
    set_optimizer_learning_rate(optimizer, args.learning_rate)

    history: list[dict] = []
    # `best` ranks by Cα RMSD (computed at every logged eval step) because the
    # user actually cares about geometry; training loss is a loose proxy and
    # oscillates around convergence. We save the best model state dict so the
    # output PDB is the best-geometry checkpoint, not whichever step happened
    # to be last.
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
            violations_enabled_at_step = step
            print(f"[overfit] enabling structural-violation loss at step {step}")
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
            with torch.no_grad():
                eval_outputs = model(**model_inputs_from_batch(batch, training_config))
            metrics = structure_metrics(eval_outputs, batch)
            entry.update(metrics)
            elapsed = time.time() - start_time
            print(
                f"[overfit] step {step:4d}/{args.steps} "
                f"loss={loss_value:.4f}  "
                f"bb_rmsd={metrics['backbone_rmsd_after_alignment']:.3f}  "
                f"ca_rmsd={metrics['ca_rmsd_after_alignment']:.3f}  "
                f"aa_rmsd={metrics['all_atom_rmsd_after_alignment']:.3f}  "
                f"pep={metrics['peptide_bond_mean']:.2f}"
                f"[{metrics['peptide_bond_min']:.2f},{metrics['peptide_bond_max']:.2f}]  "
                f"({elapsed:.0f}s)"
            )
            if metrics["ca_rmsd_after_alignment"] < best["ca_rmsd_after_alignment"]:
                best = {"loss": loss_value, "step": step, **metrics}
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(entry)

    # Restore the best-by-Cα-RMSD checkpoint for the output PDB so the user
    # sees the best learned structure, not whichever step the loop stopped on.
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[overfit] restoring best-by-ca-rmsd checkpoint: step {best['step']}")
    model.eval()
    with torch.no_grad():
        final_outputs = model(**model_inputs_from_batch(batch, training_config))
    final_metrics = structure_metrics(final_outputs, batch)

    aligned_outputs = apply_kabsch_alignment_to_outputs(final_outputs, batch)
    predicted_pdb = out_dir / f"predicted_{chain_id}.pdb"
    write_model_output_pdb(predicted_pdb, aligned_outputs, batch, example_index=0)

    truth_pdb = out_dir / f"ground_truth_{chain_id}.pdb"
    write_atom14_pdb(
        truth_pdb,
        batch["aatype"][0].detach().cpu(),
        batch["true_atom_positions"][0].detach().cpu(),
        batch["true_atom_mask"][0].detach().cpu(),
        residue_index=batch["residue_index"][0].detach().cpu(),
    )
    write_pymol_script(out_dir / "view_in_pymol.pml", predicted_pdb, truth_pdb)

    metrics_payload = {
        "chain_id": chain_id,
        "num_residues": n_res,
        "steps": args.steps,
        "model_profile": args.model_profile,
        "violations_after_step": args.violations_after_step,
        "violations_enabled_at_step": violations_enabled_at_step,
        "final_loss": float(history[-1]["loss"]),
        "best": best,
        "final_metrics": final_metrics,
        "predicted_pdb": str(predicted_pdb),
        "ground_truth_pdb": str(truth_pdb),
        "pymol_script": str(out_dir / "view_in_pymol.pml"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    (out_dir / "losses.json").write_text(json.dumps(history, indent=2))

    print()
    print(f"[overfit] final loss={final_metrics}")
    print(f"[overfit] best step: {best.get('step', 'n/a')}  best loss: {best['loss']:.4f}")
    print(f"[overfit] artefacts written to {out_dir}")
    print(f"[overfit] view in PyMOL: pymol {out_dir / 'view_in_pymol.pml'}")
    return metrics_payload


if __name__ == "__main__":
    main()
