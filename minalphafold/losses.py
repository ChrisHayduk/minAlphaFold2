import torch
from typing import Optional

try:
    from .utils import distance_bin
    from .residue_constants import (
        between_res_bond_length_c_n, between_res_bond_length_stddev_c_n,
        between_res_cos_angles_c_n_ca, between_res_cos_angles_ca_c_n,
        make_atom14_dists_bounds,
        restype_atom14_vdw_radius,
    )
except ImportError:  # pragma: no cover - compatibility for direct module imports in tests/scripts.
    from utils import distance_bin
    from residue_constants import (
        between_res_bond_length_c_n, between_res_bond_length_stddev_c_n,
        between_res_cos_angles_c_n_ca, between_res_cos_angles_ca_c_n,
        make_atom14_dists_bounds,
        restype_atom14_vdw_radius,
    )


def frame_aligned_point_error(
    predicted_rotations: torch.Tensor,
    predicted_translations: torch.Tensor,
    true_rotations: torch.Tensor,
    true_translations: torch.Tensor,
    predicted_positions: torch.Tensor,
    true_positions: torch.Tensor,
    frames_mask: torch.Tensor,
    positions_mask: torch.Tensor,
    *,
    length_scale: float,
    pair_mask: Optional[torch.Tensor] = None,
    l1_clamp_distance: Optional[float] = None,
    eps: float,
) -> torch.Tensor:
    """Frame aligned point error (Algorithm 28, supplement 1.9.2).

    Computes `L_FAPE = (1/Z) mean_{i,j}(min(d_clamp, d_ij))` where
    `d_ij = sqrt(||T_i^{-1} ∘ x_j - T_i^{true -1} ∘ x_j^{true}||^2 + eps)`.

    ε is a caller-supplied smoothing constant. Algorithm 20 uses ε = 10⁻¹²
    for the per-layer auxiliary FAPE on Cα atoms (line 17) and ε = 10⁻⁴ for
    the final all-atom FAPE (line 28); the paper notes the exact value does
    not matter as long as it is small enough. `length_scale` is Z = 10 Å
    (supplement 1.9.2). `l1_clamp_distance` is d_clamp = 10 Å when clamping
    is requested (supplement 1.11.5).

    Masks extend the paper to accommodate padded/variable-length batches:
    `frames_mask` and `positions_mask` mark valid frames i and atoms j; the
    optional `pair_mask` also masks specific (i, j) pairs. The denominator
    is the number of valid (i, j) pairs — equal to `N_res^2` when nothing
    is masked, matching the supplement's `mean_{i,j}`.
    """

    # Algorithm 28 lines 1-2: x_ij = T_i^{-1} ∘ x_j, with (R, t)^{-1} = (R^T, -R^T t).
    predicted_rotations_inv = predicted_rotations.transpose(-1, -2)
    predicted_translations_inv = -torch.einsum(
        "...ij,...j->...i",
        predicted_rotations_inv,
        predicted_translations,
    )
    true_rotations_inv = true_rotations.transpose(-1, -2)
    true_translations_inv = -torch.einsum(
        "...ij,...j->...i",
        true_rotations_inv,
        true_translations,
    )

    local_predicted_positions = (
        torch.einsum("...fij,...aj->...fai", predicted_rotations_inv, predicted_positions)
        + predicted_translations_inv[..., :, None, :]
    )
    local_true_positions = (
        torch.einsum("...fij,...aj->...fai", true_rotations_inv, true_positions)
        + true_translations_inv[..., :, None, :]
    )

    # Algorithm 28 line 3: d_ij = sqrt(||Δx||^2 + ε).
    error_distance = torch.sqrt(
        torch.sum((local_predicted_positions - local_true_positions) ** 2, dim=-1) + eps
    )

    # Algorithm 28 line 4: min(d_clamp, d_ij) is equivalent to clamp(max=d_clamp)
    # since d_ij ≥ 0 by construction.
    if l1_clamp_distance is not None:
        error_distance = error_distance.clamp(max=l1_clamp_distance)

    # mean_{i,j}(...) divided by length_scale Z. The mask zeros out (i, j)
    # pairs where either the frame or atom is invalid (or the caller-provided
    # pair_mask rejects the pair); the denominator counts just the surviving
    # pairs, so un-masked inputs recover the paper's 1/N_res^2 normalisation.
    mask = frames_mask[..., :, None] * positions_mask[..., None, :]
    if pair_mask is not None:
        mask = mask * pair_mask

    numerator = (error_distance * mask).sum(dim=(-1, -2))
    denominator = mask.sum(dim=(-1, -2)).clamp(min=1.0)
    return numerator / (denominator * length_scale)


class AlphaFoldLoss(torch.nn.Module):
    """Combined AlphaFold loss (supplement 1.9 equation 7).

    Training:
        L = 0.5 L_FAPE + 0.5 L_aux + 0.3 L_dist + 2.0 L_msa + 0.01 L_conf
    Fine-tuning:
        L += 0.01 L_exp_resolved + 1.0 L_viol

    Term-by-term mapping (all weights below are *absolute* eq 7 weights, not
    relative fractions):

    * `sidechain_weight_frac = 0.5` doubles as the weight of L_FAPE (final
      all-atom FAPE, Algorithm 20 line 28) AND the weight on the FAPE half
      of L_aux (backbone FAPE averaged over iterations, Algorithm 20 line 17).
      We decompose `0.5 L_aux = 0.5 L_aux^{FAPE} + 0.5 L_aux^{torsion}` and
      sum `0.5 L_FAPE + 0.5 L_aux^{FAPE}` directly.
    * `distogram_weight = 0.3`   → 0.3 L_dist (supplement 1.9.8, eq 41).
    * `msa_weight = 2.0`         → 2.0 L_msa  (supplement 1.9.9, eq 42).
    * `confidence_weight = 0.01` → 0.01 L_conf (supplement 1.9.6, Alg 29).
    * `experimentally_resolved_weight = 0.01` → 0.01 L_exp_resolved (1.9.10).
    * `structural_violation_weight = 1.0` → 1.0 L_viol (1.9.11, eq 47).

    `TorsionAngleLoss` pre-applies the outer 0.5 factor on L_aux^{torsion}
    and the 0.02 coefficient on L_anglenorm from Algorithm 27, so
    `weighted_torsion_loss = torsion_loss` without an additional weight.

    `use_clamped_fape` implements supplement 1.11.5: in 90% of training
    mini-batches the backbone FAPE is clamped by 10 Å, and unclamped in the
    remaining 10%. Passing a float in [0, 1] mixes the two versions with
    the given clamped weight; `None` defaults to fully clamped. The side-
    chain FAPE is always clamped regardless, so this knob never reaches it.
    """

    def __init__(self, finetune: bool = False, use_clamped_fape: Optional[float] = None):
        super().__init__()
        self.torsion_angle_loss = TorsionAngleLoss()
        self.plddt_loss = PLDDTLoss(
            filter_by_resolution=True,
            min_resolution=0.1,
            max_resolution=3.0,
        )
        self.distogram_loss = DistogramLoss()
        self.msa_loss = MSALoss()
        self.experimentally_resolved_loss = ExperimentallyResolvedLoss(
            filter_by_resolution=True,
            min_resolution=0.1,
            max_resolution=3.0,
        )
        self.structural_violation_loss = StructuralViolationLoss()
        self.backbone_loss = BackboneTrajectoryLoss()
        self.sidechain_fape_loss = AllAtomFAPE()

        # Equation 7 weights, all absolute (not relative).
        self.sidechain_weight_frac = 0.5
        self.distogram_weight = 0.3
        self.msa_weight = 2.0
        self.confidence_weight = 0.01
        self.experimentally_resolved_weight = 0.01
        self.structural_violation_weight = 1.0

        self.finetune = finetune
        self.use_clamped_fape = use_clamped_fape

    def forward(
            self,
            structure_model_prediction: dict,
            true_rotations: torch.Tensor,           # (b, N_res, 3, 3)
            true_translations: torch.Tensor,        # (b, N_res, 3)
            true_atom_positions: torch.Tensor,      # (b, N_res, 14, 3)
            true_atom_mask: torch.Tensor,           # (b, N_res, 14)
            true_atom_positions_alt: torch.Tensor,
            true_atom_mask_alt: torch.Tensor,
            true_atom_is_ambiguous: torch.Tensor,
            true_torsion_angles: torch.Tensor,      # (b, N_res, 7, 2)
            true_torsion_angles_alt: torch.Tensor,  # (b, N_res, 7, 2)
            true_torsion_mask: torch.Tensor,        # (b, N_res, 7)
            true_rigid_group_frames_R: torch.Tensor,
            true_rigid_group_frames_t: torch.Tensor,
            true_rigid_group_frames_R_alt: torch.Tensor,
            true_rigid_group_frames_t_alt: torch.Tensor,
            true_rigid_group_exists: torch.Tensor,
            experimentally_resolved_pred: torch.Tensor,
            experimentally_resolved_true: torch.Tensor,
            experimentally_resolved_exists: torch.Tensor,
            masked_msa_pred: torch.Tensor,
            masked_msa_target: torch.Tensor,
            masked_msa_mask: torch.Tensor,
            plddt_pred: torch.Tensor,
            distogram_pred: torch.Tensor,
            res_types: torch.Tensor,                # (b, N_res) integer 0-20
            residue_index: torch.Tensor,
            seq_mask: Optional[torch.Tensor] = None,  # (b, N_res) 1=valid, 0=padding
            return_breakdown: bool = False,
            resolution: Optional[torch.Tensor] = None,
        ):
        loss_terms = self.compute_loss_terms(
            structure_model_prediction=structure_model_prediction,
            true_rotations=true_rotations,
            true_translations=true_translations,
            true_atom_positions=true_atom_positions,
            true_atom_mask=true_atom_mask,
            true_atom_positions_alt=true_atom_positions_alt,
            true_atom_mask_alt=true_atom_mask_alt,
            true_atom_is_ambiguous=true_atom_is_ambiguous,
            true_torsion_angles=true_torsion_angles,
            true_torsion_angles_alt=true_torsion_angles_alt,
            true_torsion_mask=true_torsion_mask,
            true_rigid_group_frames_R=true_rigid_group_frames_R,
            true_rigid_group_frames_t=true_rigid_group_frames_t,
            true_rigid_group_frames_R_alt=true_rigid_group_frames_R_alt,
            true_rigid_group_frames_t_alt=true_rigid_group_frames_t_alt,
            true_rigid_group_exists=true_rigid_group_exists,
            experimentally_resolved_pred=experimentally_resolved_pred,
            experimentally_resolved_true=experimentally_resolved_true,
            experimentally_resolved_exists=experimentally_resolved_exists,
            resolution=resolution,
            masked_msa_pred=masked_msa_pred,
            masked_msa_target=masked_msa_target,
            masked_msa_mask=masked_msa_mask,
            plddt_pred=plddt_pred,
            distogram_pred=distogram_pred,
            res_types=res_types,
            residue_index=residue_index,
            seq_mask=seq_mask,
        )
        if return_breakdown:
            return loss_terms["loss"], loss_terms
        return loss_terms["loss"]

    def compute_loss_terms(
            self,
            structure_model_prediction: dict,
            true_rotations: torch.Tensor,
            true_translations: torch.Tensor,
            true_atom_positions: torch.Tensor,
            true_atom_mask: torch.Tensor,
            true_atom_positions_alt: torch.Tensor,
            true_atom_mask_alt: torch.Tensor,
            true_atom_is_ambiguous: torch.Tensor,
            true_torsion_angles: torch.Tensor,
            true_torsion_angles_alt: torch.Tensor,
            true_torsion_mask: torch.Tensor,
            true_rigid_group_frames_R: torch.Tensor,
            true_rigid_group_frames_t: torch.Tensor,
            true_rigid_group_frames_R_alt: torch.Tensor,
            true_rigid_group_frames_t_alt: torch.Tensor,
            true_rigid_group_exists: torch.Tensor,
            experimentally_resolved_pred: torch.Tensor,
            experimentally_resolved_true: torch.Tensor,
            experimentally_resolved_exists: torch.Tensor,
            masked_msa_pred: torch.Tensor,
            masked_msa_target: torch.Tensor,
            masked_msa_mask: torch.Tensor,
            plddt_pred: torch.Tensor,
            distogram_pred: torch.Tensor,
            res_types: torch.Tensor,
            residue_index: torch.Tensor,
            seq_mask: Optional[torch.Tensor] = None,
            resolution: Optional[torch.Tensor] = None,
        ) -> dict[str, torch.Tensor]:
        pred_all_frames_R = structure_model_prediction["all_frames_R"]  # (batch, N_res, 8, 3, 3)
        pred_all_frames_t = structure_model_prediction["all_frames_t"]  # (batch, N_res, 8, 3)
        atom_coords = structure_model_prediction["atom14_coords"]   # (batch, N_res, 14, 3)
        atom_mask = structure_model_prediction["atom14_mask"]       # (batch, N_res, 14)
        # Canonically rename ambiguous sidechains before any atom-derived supervision.
        true_atom_positions, true_atom_mask, alt_naming_is_better = select_best_atom14_ground_truth(
            atom_coords,
            true_atom_positions,
            true_atom_mask,
            true_atom_positions_alt,
            true_atom_mask_alt,
            true_atom_is_ambiguous,
        )

        renamed_rigid_group_frames_R = torch.where(
            alt_naming_is_better[:, :, None, None, None] > 0,
            true_rigid_group_frames_R_alt,
            true_rigid_group_frames_R,
        )
        renamed_rigid_group_frames_t = torch.where(
            alt_naming_is_better[:, :, None, None] > 0,
            true_rigid_group_frames_t_alt,
            true_rigid_group_frames_t,
        )

        backbone_mask = true_atom_mask[:, :, 0] * true_atom_mask[:, :, 1] * true_atom_mask[:, :, 2]
        backbone_loss = self.backbone_loss(
            structure_model_prediction,
            true_rotations,
            true_translations,
            backbone_mask=backbone_mask,
            seq_mask=seq_mask,
            use_clamped_fape=self.use_clamped_fape,
        )
        # Side-chain FAPE is always clamped (supplement 1.11.5); use_clamped_fape
        # controls only the backbone FAPE trajectory loss above.
        sidechain_loss = self.sidechain_fape_loss(
            pred_all_frames_R,
            pred_all_frames_t,
            atom_coords,
            atom_mask,
            renamed_rigid_group_frames_R,
            renamed_rigid_group_frames_t,
            true_atom_positions,
            true_atom_mask=true_atom_mask,
            seq_mask=seq_mask,
            frame_mask=true_rigid_group_exists,
        )
        torsion_loss = self.torsion_angle_loss(
            structure_model_prediction["traj_torsion_angles"],
            structure_model_prediction["traj_torsion_angles_unnormalized"],
            true_torsion_angles,
            true_torsion_angles_alt,
            true_torsion_mask,
            seq_mask=seq_mask,
        )

        # --- Derive distogram target (supplement 1.9.8) ---
        # Targets are the one-hot encoding of Cβ-Cβ distances (Cα for GLY).
        is_gly = (res_types == 7)                       # (batch, N_res)
        cb_idx = torch.where(is_gly, 1, 4)              # atom14 slots: CA=1, CB=4
        cb_pos = torch.gather(
            true_atom_positions, 2,
            cb_idx[:, :, None, None].expand(-1, -1, 1, 3),
        ).squeeze(2)
        n_dist_bins = distogram_pred.shape[-1]
        cb_mask = torch.gather(true_atom_mask, 2, cb_idx[:, :, None]).squeeze(-1)
        distogram_true = distance_bin(cb_pos, n_dist_bins)
        dist_pair_mask = cb_mask[:, :, None] * cb_mask[:, None, :]
        if seq_mask is not None:
            dist_pair_mask = dist_pair_mask * (seq_mask[:, :, None] * seq_mask[:, None, :])

        dist_loss = self.distogram_loss(distogram_pred, distogram_true, pair_mask=dist_pair_mask)
        msa_loss = self.msa_loss(masked_msa_pred, masked_msa_target, masked_msa_mask)

        # --- Derive pLDDT target (supplement 1.9.6) ---
        # Compute per-residue lDDT-Cα of the prediction against ground truth,
        # then discretise into 50 bins of width 2 (v_bins in Algorithm 29).
        # lDDT-Cα is the mean over 4 thresholds (0.5, 1, 2, 4 Å) of the fraction
        # of included Cα-Cα distance pairs that are preserved within tolerance.
        # "Included" = pairs with d_true < 15 Å, excluding self.
        N_res = atom_coords.shape[1]
        with torch.no_grad():
            pred_ca = atom_coords[:, :, 1, :]                 # (batch, N_res, 3)
            true_ca = true_atom_positions[:, :, 1, :]
            true_ca_mask = true_atom_mask[:, :, 1]
            true_ca_dists = torch.cdist(true_ca, true_ca)     # (batch, N_res, N_res)
            pred_ca_dists = torch.cdist(pred_ca, pred_ca)
            inclusion = (true_ca_dists < 15.0).float() * (
                1.0 - torch.eye(N_res, device=pred_ca.device).unsqueeze(0))
            inclusion = inclusion * (true_ca_mask[:, :, None] * true_ca_mask[:, None, :])
            if seq_mask is not None:
                pair_valid = seq_mask[:, :, None] * seq_mask[:, None, :]  # (batch, N_res, N_res)
                inclusion = inclusion * pair_valid
            dist_error = torch.abs(pred_ca_dists - true_ca_dists)
            # Average fraction of preserved distances across four thresholds
            lddt = torch.zeros(pred_ca.shape[:2], device=pred_ca.device)  # (batch, N_res)
            n_included = inclusion.sum(dim=-1).clamp(min=1)
            for thresh in [0.5, 1.0, 2.0, 4.0]:
                lddt = lddt + ((dist_error < thresh).float() * inclusion).sum(dim=-1) / n_included
            lddt = lddt / 4.0  # (batch, N_res) in [0, 1]
            lddt_mask = true_ca_mask if seq_mask is None else true_ca_mask * seq_mask
            n_plddt_bins = plddt_pred.shape[-1]
            plddt_edges = torch.arange(1, n_plddt_bins, device=pred_ca.device).float() / n_plddt_bins
            plddt_bin_idx = torch.bucketize(lddt, plddt_edges)
            plddt_true = torch.nn.functional.one_hot(plddt_bin_idx, n_plddt_bins).float()

        plddt_loss = self.plddt_loss(
            plddt_pred,
            plddt_true,
            seq_mask=lddt_mask,
            resolution=resolution,
        )

        # Equation 7, training row. `backbone_loss` is L_aux^{FAPE} = mean_l(FAPE^l);
        # `sidechain_loss` is L_FAPE (all-atom, final layer); `torsion_loss` already
        # bundles 0.5 * L_aux^{torsion} + 0.01 * L_aux^{anglenorm} per Algorithm 27
        # and the L_aux factor from equation 7.
        weighted_backbone_loss = (1.0 - self.sidechain_weight_frac) * backbone_loss
        weighted_sidechain_fape_loss = self.sidechain_weight_frac * sidechain_loss
        weighted_torsion_loss = torsion_loss
        fape_loss = weighted_backbone_loss + weighted_sidechain_fape_loss
        structure_loss = fape_loss + weighted_torsion_loss
        weighted_distogram_loss = self.distogram_weight * dist_loss
        weighted_msa_loss = self.msa_weight * msa_loss
        weighted_plddt_loss = self.confidence_weight * plddt_loss
        loss = structure_loss + weighted_distogram_loss + weighted_msa_loss + weighted_plddt_loss

        loss_terms = {
            "loss": loss,
            "structure_loss": structure_loss,
            "fape_loss": fape_loss,
            "backbone_loss": backbone_loss,
            "sidechain_fape_loss": sidechain_loss,
            "torsion_loss": torsion_loss,
            "distogram_loss": dist_loss,
            "msa_loss": msa_loss,
            "plddt_loss": plddt_loss,
            "weighted_backbone_loss": weighted_backbone_loss,
            "weighted_sidechain_fape_loss": weighted_sidechain_fape_loss,
            "weighted_torsion_loss": weighted_torsion_loss,
            "weighted_distogram_loss": weighted_distogram_loss,
            "weighted_msa_loss": weighted_msa_loss,
            "weighted_plddt_loss": weighted_plddt_loss,
        }

        if self.finetune:
            structural_violation_loss = self.structural_violation_loss(
                atom_coords,
                atom_mask,
                res_types,
                residue_index,
            )
            weighted_structural_violation_loss = self.structural_violation_weight * structural_violation_loss
            loss = loss + weighted_structural_violation_loss
            loss_terms["structural_violation_loss"] = structural_violation_loss
            loss_terms["weighted_structural_violation_loss"] = weighted_structural_violation_loss

            exp_resolved_loss = self.experimentally_resolved_loss(
                experimentally_resolved_pred,
                experimentally_resolved_true,
                experimentally_resolved_exists,
                resolution=resolution,
            )
            weighted_exp_resolved_loss = self.experimentally_resolved_weight * exp_resolved_loss
            loss = loss + weighted_exp_resolved_loss
            loss_terms["experimentally_resolved_loss"] = exp_resolved_loss
            loss_terms["weighted_experimentally_resolved_loss"] = weighted_exp_resolved_loss

        loss_terms["loss"] = loss
        return loss_terms
    
class BackboneTrajectoryLoss(torch.nn.Module):
    """Per-iteration backbone FAPE averaged over layers (L_aux^{FAPE}).

    Algorithm 20 emits backbone frames at every iteration l ∈ [1, N_layer];
    line 17 computes a Cα-only FAPE against ground truth on each iteration
    and line 23 averages them to yield the FAPE component of L_aux.

    `use_clamped_fape ∈ [0, 1]` (supplement 1.11.5): weight of the clamped
    FAPE in a soft mix with the unclamped version. `None` ≡ 1.0 (fully
    clamped). AlphaFold samples 10% of mini-batches to be fully unclamped,
    so the expected loss per batch has ≈0.9 weight on the clamped form;
    passing `use_clamped_fape=0.9` reproduces that expectation directly.
    """

    def __init__(self):
        super().__init__()
        self.fape_loss = BackboneFAPE()

    def forward(
            self,
            structure_model_prediction: dict,
            true_rotations: torch.Tensor,          # (b, N_res, 3, 3)
            true_translations: torch.Tensor,        # (b, N_res, 3)
            backbone_mask: Optional[torch.Tensor] = None,
            seq_mask: Optional[torch.Tensor] = None,
            use_clamped_fape: Optional[torch.Tensor] = None,
        ):
        traj_R = structure_model_prediction["traj_rotations"]          # (L, b, N_res, 3, 3)
        traj_t = structure_model_prediction["traj_translations"]       # (L, b, N_res, 3)

        num_layers = traj_R.shape[0]
        total_loss = torch.zeros(traj_R.shape[1], device=traj_R.device, dtype=traj_R.dtype)
        valid_mask = backbone_mask if seq_mask is None else backbone_mask * seq_mask

        for l in range(num_layers):
            clamped_fape = self.fape_loss(
                traj_R[l],
                traj_t[l],
                true_rotations,
                true_translations,
                frame_mask=valid_mask,
                position_mask=valid_mask,
                l1_clamp_distance=self.fape_loss.d_clamp_val,
            )
            if use_clamped_fape is None:
                total_loss = total_loss + clamped_fape
            else:
                unclamped_fape = self.fape_loss(
                    traj_R[l],
                    traj_t[l],
                    true_rotations,
                    true_translations,
                    frame_mask=valid_mask,
                    position_mask=valid_mask,
                    l1_clamp_distance=None,
                )
                total_loss = total_loss + (
                    clamped_fape * use_clamped_fape + unclamped_fape * (1.0 - use_clamped_fape)
                )

        # Algorithm 20 line 23: L_aux = mean_l(L_aux^l). This returns just the
        # FAPE component; the torsion component is added separately by the
        # caller as TorsionAngleLoss.
        return total_loss / num_layers


def select_best_atom14_ground_truth(
    predicted_atom_positions: torch.Tensor,
    true_atom_positions: torch.Tensor,
    true_atom_mask: torch.Tensor,
    true_atom_positions_alt: torch.Tensor,
    true_atom_mask_alt: torch.Tensor,
    true_atom_is_ambiguous: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rename 180°-symmetric ground-truth atoms (Algorithm 26).

    Some residues (ASP/GLU/PHE/TYR — see Table 3) are symmetric under a 180°
    flip around their side-chain torsion axis, which makes the naming of
    their terminal atoms ambiguous. For each residue i we pick the naming
    (either "true" or "alt truth") that minimises the |pred - true| distance
    sum over pairs (a, (j, b)) where a is ambiguous in residue i and (j, b)
    is a non-ambiguous atom elsewhere — matching Algorithm 26 line 5.

    Paper asymmetry: `d^{alt truth}_{(i,a),(j,b)} = ||x_i^{alt truth,a} -
    x_j^{true,b}||` uses alt positions only on the i side. We compute
    pairwise distances from `true_atom_positions_alt` on both sides, which
    agrees with the paper because the j-side (b) is masked to non-ambiguous
    atoms where `alt == true` anyway.
    """
    batch_size, num_residues, num_atoms = predicted_atom_positions.shape[:3]

    def pairwise_distances(atom14_positions: torch.Tensor) -> torch.Tensor:
        flat_positions = atom14_positions.reshape(batch_size, num_residues * num_atoms, 3)
        flat_distances = torch.cdist(flat_positions, flat_positions)
        return flat_distances.reshape(batch_size, num_residues, num_atoms, num_residues, num_atoms).permute(0, 1, 3, 2, 4)

    pred_dists = pairwise_distances(predicted_atom_positions)
    true_dists = pairwise_distances(true_atom_positions)
    alt_true_dists = pairwise_distances(true_atom_positions_alt)

    # |d - d^true| vs |d - d^{alt truth}| per Algorithm 26 line 5.
    error = torch.sqrt(1e-10 + (pred_dists - true_dists) ** 2)
    alt_error = torch.sqrt(1e-10 + (pred_dists - alt_true_dists) ** 2)

    # Restrict sums to pairs with (i, a) ambiguous and (j, b) non-ambiguous
    # (S_non-ambiguous atoms in the paper), per Algorithm 26 line 5's quantifier.
    ambiguity_mask = (
        true_atom_mask[:, :, None, :, None]
        * true_atom_is_ambiguous[:, :, None, :, None]
        * true_atom_mask[:, None, :, None, :]
        * (1.0 - true_atom_is_ambiguous[:, None, :, None, :])
    )

    per_res_error = torch.sum(ambiguity_mask * error, dim=(2, 3, 4))
    per_res_alt_error = torch.sum(ambiguity_mask * alt_error, dim=(2, 3, 4))

    # Algorithm 26 lines 5-7: swap truth ↔ alt for residues where alt fits better.
    use_alt = (per_res_alt_error < per_res_error).to(true_atom_positions.dtype)
    chosen_positions = torch.where(
        use_alt[..., None, None] > 0,
        true_atom_positions_alt,
        true_atom_positions,
    )
    chosen_mask = torch.where(
        use_alt[..., None] > 0,
        true_atom_mask_alt,
        true_atom_mask,
    )
    return chosen_positions, chosen_mask, use_alt

class TorsionAngleLoss(torch.nn.Module):
    """Side-chain and backbone torsion angle loss (Algorithm 27).

    Scores all 7 torsions f ∈ {ω, φ, ψ, χ1, χ2, χ3, χ4} per the supplement.
    Validity of each torsion is carried by `torsion_mask_true` (shape
    [..., 7]): ω/φ are undefined for the first residue, χ1..χ4 existence
    depends on residue type, and any torsion whose atoms are missing in the
    ground truth is masked out. See `geometry.torsion_angles` for how the
    mask is built from atom14 data.

    The min-of-(true, alt_true) term in line 3 handles 180°-rotation symmetry
    for ASP/GLU/PHE/TYR; for all other torsions alt_true == true, so the
    minimum reduces to the plain L2 term.

    Both `torsion_weight` and `angle_norm_weight` pre-apply the outer 0.5
    factor that equation (7) multiplies L_aux by: they are (1.0, 0.02) from
    Algorithm 27 times 0.5, i.e. (0.5, 0.01). Keeping the pre-multiplication
    here lets `AlphaFoldLoss` add the returned value directly.
    """

    def __init__(self):
        super().__init__()
        self.torsion_weight = 0.5
        self.angle_norm_weight = 0.01

    def forward(
        self,
        torsion_angles: torch.Tensor,
        unnormalized_torsion_angles: torch.Tensor,
        torsion_angles_true: torch.Tensor,
        torsion_angles_true_alt: torch.Tensor,
        torsion_mask_true: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ):
        # Prepend the per-layer trajectory dim if only the final iteration was
        # passed (Algorithm 20 averages L_aux over layers; the normalization
        # below sums across L so a single layer broadcasts as L=1).
        if torsion_angles.ndim == 4:
            torsion_angles = torsion_angles.unsqueeze(0)
            unnormalized_torsion_angles = unnormalized_torsion_angles.unsqueeze(0)

        true_angles = torsion_angles_true.unsqueeze(0)
        true_alt = torsion_angles_true_alt.unsqueeze(0)
        mask = torsion_mask_true.unsqueeze(0)  # (1, b, N_res, 7)
        if seq_mask is not None:
            mask = mask * seq_mask.unsqueeze(0).unsqueeze(-1)

        # Algorithm 27 line 3: L_torsion = mean_{i,f} min(||α̂ - α_true||², ||α̂ - α_alt||²).
        true_dist_sq = torch.sum((true_angles - torsion_angles) ** 2, dim=-1)
        alt_dist_sq = torch.sum((true_alt - torsion_angles) ** 2, dim=-1)
        torsion_dist_sq = torch.minimum(true_dist_sq, alt_dist_sq)

        torsion_normalizer = mask.sum(dim=(0, 2, 3)).clamp(min=1.0)
        torsion_loss = torch.sum(torsion_dist_sq * mask, dim=(0, 2, 3)) / torsion_normalizer

        # Algorithm 27 line 4: L_anglenorm = mean_{i,f} |||α_tilde_i^f|| - 1|.
        # Covers all 7 torsions regardless of mask — this regulariser keeps the
        # raw network output close to unit norm before the α̂ = α̃ / ||α̃||
        # normalization, independent of whether the torsion is supervised.
        angle_norm = torch.sqrt(torch.sum(unnormalized_torsion_angles ** 2, dim=-1) + 1e-8)
        if seq_mask is not None:
            angle_norm_mask = seq_mask.unsqueeze(0).unsqueeze(-1).expand_as(angle_norm)
        else:
            angle_norm_mask = torch.ones_like(angle_norm)
        norm_normalizer = angle_norm_mask.sum(dim=(0, 2, 3)).clamp(min=1.0)
        angle_norm_loss = torch.sum(torch.abs(angle_norm - 1.0) * angle_norm_mask, dim=(0, 2, 3)) / norm_normalizer

        return self.torsion_weight * torsion_loss + self.angle_norm_weight * angle_norm_loss
    
class BackboneFAPE(torch.nn.Module):
    """Auxiliary backbone FAPE (Algorithm 20 line 17).

    Scores per-iteration backbone frames against the ground-truth backbone
    frames, using the frame translations (Cα positions) as the atoms. Runs
    inside `BackboneTrajectoryLoss`, which averages over Structure Module
    iterations to realise `L_aux^{FAPE} = mean_l(L_FAPE^l)` from Algorithm
    20 lines 17 and 23.

    Uses ε = 10⁻¹² per Algorithm 20 line 17; Z = d_clamp = 10 Å.
    """

    def __init__(self, d_clamp: float = 10.0, eps: float = 1e-12, Z: float = 10.0):
        super().__init__()
        self.eps = eps
        self.d_clamp_val = d_clamp
        self.Z = Z

    def forward(self,
                predicted_rotations,      # (b, N_res, 3, 3)
                predicted_translations,   # (b, N_res, 3)
                true_rotations,           # (b, N_res, 3, 3)
                true_translations,        # (b, N_res, 3)
                frame_mask: torch.Tensor,  # (b, N_res)
                position_mask: torch.Tensor,  # (b, N_res)
                pair_mask: Optional[torch.Tensor] = None,  # (b, N_res, N_res)
                l1_clamp_distance: Optional[float] = None,
    ):
        # The frames and the atoms are the same objects: backbone frames'
        # translations *are* the Cα positions (Algorithm 20 lines 15-16).
        return frame_aligned_point_error(
            predicted_rotations,
            predicted_translations,
            true_rotations,
            true_translations,
            predicted_translations,
            true_translations,
            frame_mask,
            position_mask,
            length_scale=self.Z,
            pair_mask=pair_mask,
            l1_clamp_distance=l1_clamp_distance,
            eps=self.eps,
        )


class AllAtomFAPE(torch.nn.Module):
    """Final all-atom FAPE (Algorithm 20 line 28).

    Scores the 8 per-residue rigid-group frames (3 backbone + 4 side-chain
    torsion frames + ψ frame, see Table 2) against all 14 atom positions of
    every residue after the symmetric-ground-truth renaming of Algorithm 26.

    Per supplement 1.11.5, the side-chain FAPE is *always* clamped by
    d_clamp = 10 Å (unlike the backbone FAPE, which is unclamped in 10% of
    mini-batches). Uses ε = 10⁻⁴ per Algorithm 20 line 28.
    """

    def __init__(self, d_clamp: float = 10.0, eps: float = 1e-4, Z: float = 10.0):
        super().__init__()
        self.eps = eps
        self.d_clamp_val = d_clamp
        self.Z = Z

    def forward(self,
                predicted_frames_R,       # (b, N_res, 8, 3, 3)
                predicted_frames_t,       # (b, N_res, 8, 3)
                predicted_atom_positions, # (b, N_res, 14, 3)
                atom_mask,                # (b, N_res, 14) — predicted atom existence
                true_frames_R,            # (b, N_res, 8, 3, 3)
                true_frames_t,            # (b, N_res, 8, 3)
                true_atom_positions,      # (b, N_res, 14, 3)
                true_atom_mask: Optional[torch.Tensor] = None,  # (b, N_res, 14)
                seq_mask: Optional[torch.Tensor] = None,       # (b, N_res)
                frame_mask: Optional[torch.Tensor] = None,     # (b, N_res, 8)
    ):
        b, N_res, n_frames = predicted_frames_R.shape[:3]
        n_atoms = predicted_atom_positions.shape[2]

        # Flatten per-residue rigid groups and atoms into a single list of
        # frames / atoms so FAPE scores every (frame, atom) pair, as required
        # by Algorithm 20 line 28's `mean_{i,j}` over the full structure.
        pred_R = predicted_frames_R.reshape(b, N_res * n_frames, 3, 3)
        pred_t = predicted_frames_t.reshape(b, N_res * n_frames, 3)
        true_R = true_frames_R.reshape(b, N_res * n_frames, 3, 3)
        true_t = true_frames_t.reshape(b, N_res * n_frames, 3)
        pred_pos = predicted_atom_positions.reshape(b, N_res * n_atoms, 3)
        true_pos = true_atom_positions.reshape(b, N_res * n_atoms, 3)

        flat_atom_mask = atom_mask.reshape(b, N_res * n_atoms)
        if true_atom_mask is not None:
            flat_atom_mask = flat_atom_mask * true_atom_mask.reshape(b, N_res * n_atoms)

        group_mask = (
            frame_mask.to(predicted_frames_R.dtype)
            if frame_mask is not None
            else predicted_frames_R.new_ones(b, N_res, n_frames)
        )

        # Fold the residue-level seq_mask into both the per-atom mask and
        # per-frame mask so padded residues do not contribute.
        if seq_mask is not None:
            seq_atom_mask = seq_mask[:, :, None].expand(-1, -1, n_atoms).reshape(b, N_res * n_atoms)
            flat_atom_mask = flat_atom_mask * seq_atom_mask
            flat_frame_mask = (seq_mask[:, :, None] * group_mask).reshape(b, N_res * n_frames)
        else:
            flat_frame_mask = group_mask.reshape(b, N_res * n_frames)

        return frame_aligned_point_error(
            predicted_rotations=pred_R,
            predicted_translations=pred_t,
            true_rotations=true_R,
            true_translations=true_t,
            predicted_positions=pred_pos,
            true_positions=true_pos,
            frames_mask=flat_frame_mask,
            positions_mask=flat_atom_mask,
            length_scale=self.Z,
            # Supplement 1.11.5: side-chain FAPE is always clamped.
            l1_clamp_distance=self.d_clamp_val,
            eps=self.eps,
        )

class PLDDTLoss(torch.nn.Module):
    """Model-confidence loss L_conf (supplement 1.9.6, Algorithm 29 line 4).

    Cross-entropy between the predicted pLDDT distribution (from Algorithm 29
    lines 1-2) and the one-hot discretisation of the per-residue true lDDT-Cα
    score into 50 bins of width 2. The true lDDT-Cα is computed in
    `AlphaFoldLoss.compute_loss_terms`, this module just performs the CE.

    `filter_by_resolution` zeros the loss on examples whose crystal-structure
    resolution falls outside [0.1 Å, 3.0 Å] per supplement 1.9.6.
    """

    def __init__(
        self,
        *,
        filter_by_resolution: bool = False,
        min_resolution: float = 0.1,
        max_resolution: float = 3.0,
    ):
        super().__init__()
        self.filter_by_resolution = filter_by_resolution
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution

    def forward(
        self,
        pred_plddt: torch.Tensor,
        true_plddt: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        resolution: Optional[torch.Tensor] = None,
    ):
        # pred_plddt, true_plddt: (batch, N_res, n_plddt_bins), latter one-hot.
        log_pred = torch.log_softmax(pred_plddt, dim=-1)

        # Algorithm 29 line 4: L_conf = mean_i(p_i^{true LDDT T} · log p_i^pLDDT).
        # (The published formula omits the minus sign; cross-entropy is a
        # negative log-likelihood, hence the sign here.)
        conf_loss = -torch.einsum('bic, bic -> bi', true_plddt, log_pred)

        if seq_mask is not None:
            conf_loss = conf_loss * seq_mask
            conf_loss = conf_loss.sum(dim=-1) / seq_mask.sum(dim=-1).clamp(min=1)
        else:
            conf_loss = torch.mean(conf_loss, dim=-1)

        if self.filter_by_resolution and resolution is not None:
            resolution = resolution.to(conf_loss.device, dtype=conf_loss.dtype).reshape(-1)
            if resolution.numel() == 1 and conf_loss.numel() != 1:
                resolution = resolution.expand_as(conf_loss)
            else:
                resolution = resolution.reshape(conf_loss.shape)
            in_range = (
                (resolution >= self.min_resolution)
                & (resolution <= self.max_resolution)
            ).to(conf_loss.dtype)
            conf_loss = conf_loss * in_range

        return conf_loss


class DistogramLoss(torch.nn.Module):
    """Distogram cross-entropy L_dist (supplement 1.9.8 equation 41).

        L_dist = -1/N_res^2 Σ_{i,j} Σ_b y_{ij}^b log p_{ij}^b

    where p_{ij}^b comes from the distogram head applied to the symmetrised
    pair representation and y_{ij}^b is the one-hot encoding of the true
    Cβ-Cβ distance (Cα for glycine) into 64 equal-width bins covering 2-22 Å,
    with the final bin catching anything more distant. Target construction
    lives in `AlphaFoldLoss.compute_loss_terms`.

    When `pair_mask` masks padded residues, the denominator is the number of
    *valid* pairs rather than `N_res^2` so variable-length batches behave
    sensibly; on a full un-padded crop the two agree.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_distograms: torch.Tensor, true_distograms: torch.Tensor,
                pair_mask: Optional[torch.Tensor] = None):
        # input shapes: (batch, N_res, N_res, num_dist_buckets)
        log_pred = torch.log_softmax(pred_distograms, dim=-1)
        vals = torch.einsum('bijc, bijc -> bij', true_distograms, log_pred)
        if pair_mask is not None:
            vals = vals * pair_mask
            dist_loss = -vals.sum(dim=(1, 2)) / pair_mask.sum(dim=(1, 2)).clamp(min=1)
        else:
            dist_loss = -torch.mean(vals, dim=(1, 2))
        return dist_loss


class MSALoss(torch.nn.Module):
    """Masked MSA cross-entropy L_msa (supplement 1.9.9 equation 42).

        L_msa = -1/N_mask Σ_{(s,i)∈mask} Σ_{c=1}^{23} y_{si}^c log p_{si}^c

    23 classes = 20 amino acids + unknown + gap + mask token (supplement
    1.9.9). Only positions selected by the BERT-style MSA masking
    contribute, and the sum is divided by the number of masked positions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, msa_preds: torch.Tensor, msa_true: torch.Tensor, masked_msa_mask: torch.Tensor):
        # msa_preds: (batch, N_seq, N_res, 23) logits; msa_true: one-hot targets.
        log_pred = torch.log_softmax(msa_preds, dim=-1)

        # Per-position cross-entropy: (batch, N_seq, N_res)
        ce = -torch.einsum('bsic, bsic -> bsi', msa_true, log_pred)

        mask = masked_msa_mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        ce = ce * mask

        # Mean over masked positions only (eq 42: 1/N_mask).
        N_mask = torch.sum(mask, dim=(1, 2)).clamp(min=1)
        msa_loss = torch.sum(ce, dim=(1, 2)) / N_mask
        return msa_loss


class ExperimentallyResolvedLoss(torch.nn.Module):
    """Experimentally-resolved loss L_exp_resolved (supplement 1.9.10 eq 43).

        L_exp_resolved = mean_{(i,a)} (
            -y_i^a log p_i^{exp resolved,a}
            -(1 - y_i^a) log(1 - p_i^{exp resolved,a}))

    Binary cross-entropy per (residue, atom37) slot predicting whether that
    atom was resolved in the crystal structure. Only used during fine-tuning
    (eq 7 fine-tuning row) and only on examples with resolution in
    [0.1 Å, 3.0 Å] (1.9.10).

    The `atom37_exists` mask restricts the mean to atom slots that are
    defined for the residue type — slots that do not exist (e.g. χ atoms on
    ALA) would otherwise inject meaningless BCE terms.
    """

    def __init__(
        self,
        *,
        filter_by_resolution: bool = False,
        min_resolution: float = 0.1,
        max_resolution: float = 3.0,
    ):
        super().__init__()
        self.filter_by_resolution = filter_by_resolution
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution

    def forward(
        self,
        exp_resolved_preds: torch.Tensor,
        exp_resolved_true: torch.Tensor,
        atom37_exists: torch.Tensor,
        resolution: Optional[torch.Tensor] = None,
    ):
        xent = torch.nn.functional.binary_cross_entropy_with_logits(
            exp_resolved_preds,
            exp_resolved_true,
            reduction="none",
        )
        weighted_xent = xent * atom37_exists
        normalizer = atom37_exists.sum(dim=(1, 2)).clamp(min=1.0)
        loss = weighted_xent.sum(dim=(1, 2)) / normalizer

        if self.filter_by_resolution and resolution is not None:
            resolution = resolution.to(loss.device, dtype=loss.dtype).reshape(-1)
            if resolution.numel() == 1 and loss.numel() != 1:
                resolution = resolution.expand_as(loss)
            else:
                resolution = resolution.reshape(loss.shape)
            in_range = (
                (resolution >= self.min_resolution)
                & (resolution <= self.max_resolution)
            ).to(loss.dtype)
            loss = loss * in_range

        return loss



class StructuralViolationLoss(torch.nn.Module):
    """Structural-violation loss L_viol (supplement 1.9.11 equations 44-47).

        L_viol = L_bondlength + L_bondangle + L_clash        (eq 47)

    * `L_bondlength` (eq 44): flat-bottom L1 on inter-residue C-N peptide
      bond lengths relative to literature values, tolerance 12 σ_lit.
    * `L_bondangle` (eq 45): flat-bottom L1 on the cosine of each peptide
      bond angle (CA-C-N and C-N-CA) against literature, tolerance 12 σ_lit.
      Supplement 1.9.11 describes a single bond-angle term; we sum the two
      peptide-bond angles, which is a sum of two paper-faithful flat-bottom
      L1 terms and matches what the DeepMind reference releases compute.
    * `L_clash` (eq 46): one-sided flat-bottom on VDW overlaps between
      non-bonded heavy atoms with tolerance 1.5 Å. Split into between- and
      within-residue halves for memory, averaged per atom so the term sits
      at a sane scale relative to L_bondlength / L_bondangle (eq 7 applies
      an absolute weight of 1.0 to L_viol).

    Used only during fine-tuning (eq 7 fine-tuning row).
    """

    vdw_table: torch.Tensor

    def __init__(
        self,
        violation_tolerance_factor: float = 12.0,
        clash_overlap_tolerance: float = 1.5,
    ):
        super().__init__()
        bounds = make_atom14_dists_bounds(
            overlap_tolerance=clash_overlap_tolerance,
            bond_length_tolerance_factor=violation_tolerance_factor,
        )
        self.register_buffer('vdw_table', torch.tensor(restype_atom14_vdw_radius))
        self.register_buffer('distance_lower_bound_table', torch.tensor(bounds["lower_bound"]))
        self.register_buffer('distance_upper_bound_table', torch.tensor(bounds["upper_bound"]))
        self.violation_tolerance_factor = violation_tolerance_factor
        self.clash_overlap_tolerance = clash_overlap_tolerance

    def forward(
        self,
        predicted_positions,  # (batch, N_res, 14, 3) — all-atom coordinates
        atom_mask,            # (batch, N_res, 14)    — 1 if atom exists, 0 otherwise
        residue_types,        # (batch, N_res)         — integer residue type index (0–20)
        residue_index,        # (batch, N_res)
    ):
        connection_violations = self.between_residue_bond_and_angle_loss(
            predicted_positions,
            atom_mask,
            residue_types,
            residue_index,
        )
        between_residue_clashes = self.between_residue_clash_loss(
            predicted_positions,
            atom_mask,
            residue_types,
            residue_index,
        )
        within_residue_violations = self.within_residue_violation_loss(
            predicted_positions,
            atom_mask,
            residue_types,
        )
        num_atoms = torch.sum(atom_mask, dim=(1, 2)).clamp(min=1e-6)
        per_atom_clash = (
            between_residue_clashes["per_atom_loss_sum"]
            + within_residue_violations["per_atom_loss_sum"]
        )
        clash_loss = torch.sum(per_atom_clash, dim=(1, 2)) / num_atoms
        return (
            connection_violations["c_n_loss_mean"]
            + connection_violations["ca_c_n_loss_mean"]
            + connection_violations["c_n_ca_loss_mean"]
            + clash_loss
        )

    def between_residue_bond_and_angle_loss(
        self,
        predicted_positions,  # (batch, N_res, 14, 3) — all-atom coordinates
        atom_mask,            # (batch, N_res, 14)    — 1 if atom exists, 0 otherwise
        residue_types,        # (batch, N_res)         — integer residue type index (0–20)
        residue_index,        # (batch, N_res)
    ):
        eps = 1e-6
        this_ca_pos = predicted_positions[:, :-1, 1, :]
        this_ca_mask = atom_mask[:, :-1, 1]
        this_c_pos = predicted_positions[:, :-1, 2, :]
        this_c_mask = atom_mask[:, :-1, 2]
        next_n_pos = predicted_positions[:, 1:, 0, :]
        next_n_mask = atom_mask[:, 1:, 0]
        next_ca_pos = predicted_positions[:, 1:, 1, :]
        next_ca_mask = atom_mask[:, 1:, 1]
        has_no_gap_mask = (residue_index[:, 1:] - residue_index[:, :-1] == 1).to(predicted_positions.dtype)

        c_n_bond_length = torch.sqrt(torch.sum((this_c_pos - next_n_pos) ** 2, dim=-1) + eps)
        next_is_proline = (residue_types[:, 1:] == 14).to(predicted_positions.dtype)
        gt_length = (
            (1.0 - next_is_proline) * between_res_bond_length_c_n[0]
            + next_is_proline * between_res_bond_length_c_n[1]
        )
        gt_stddev = (
            (1.0 - next_is_proline) * between_res_bond_length_stddev_c_n[0]
            + next_is_proline * between_res_bond_length_stddev_c_n[1]
        )
        c_n_mask = this_c_mask * next_n_mask * has_no_gap_mask
        c_n_error = torch.sqrt((c_n_bond_length - gt_length) ** 2 + eps)
        c_n_loss_per_residue = torch.relu(c_n_error - self.violation_tolerance_factor * gt_stddev)
        c_n_loss = torch.sum(c_n_mask * c_n_loss_per_residue, dim=-1) / (torch.sum(c_n_mask, dim=-1) + eps)
        c_n_violation_mask = c_n_mask * (c_n_error > (self.violation_tolerance_factor * gt_stddev))

        ca_c_bond_length = torch.sqrt(torch.sum((this_ca_pos - this_c_pos) ** 2, dim=-1) + eps)
        n_ca_bond_length = torch.sqrt(torch.sum((next_n_pos - next_ca_pos) ** 2, dim=-1) + eps)
        c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[..., None]
        c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[..., None]
        n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[..., None]

        ca_c_n_metric = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
        ca_c_n_mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
        ca_c_n_error = torch.sqrt((ca_c_n_metric - between_res_cos_angles_ca_c_n[0]) ** 2 + eps)
        ca_c_n_loss_per_residue = torch.relu(
            ca_c_n_error - self.violation_tolerance_factor * between_res_cos_angles_ca_c_n[1]
        )
        ca_c_n_loss = torch.sum(ca_c_n_mask * ca_c_n_loss_per_residue, dim=-1) / (
            torch.sum(ca_c_n_mask, dim=-1) + eps
        )
        ca_c_n_violation_mask = ca_c_n_mask * (
            ca_c_n_error > (self.violation_tolerance_factor * between_res_cos_angles_ca_c_n[1])
        )

        c_n_ca_metric = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
        c_n_ca_mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
        c_n_ca_error = torch.sqrt((c_n_ca_metric - between_res_cos_angles_c_n_ca[0]) ** 2 + eps)
        c_n_ca_loss_per_residue = torch.relu(
            c_n_ca_error - self.violation_tolerance_factor * between_res_cos_angles_c_n_ca[1]
        )
        c_n_ca_loss = torch.sum(c_n_ca_mask * c_n_ca_loss_per_residue, dim=-1) / (
            torch.sum(c_n_ca_mask, dim=-1) + eps
        )
        c_n_ca_violation_mask = c_n_ca_mask * (
            c_n_ca_error > (self.violation_tolerance_factor * between_res_cos_angles_c_n_ca[1])
        )

        per_residue_loss_sum = c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
        per_residue_loss_sum = 0.5 * (
            torch.nn.functional.pad(per_residue_loss_sum, (0, 1))
            + torch.nn.functional.pad(per_residue_loss_sum, (1, 0))
        )
        violation_mask = torch.max(
            torch.stack(
                [c_n_violation_mask, ca_c_n_violation_mask, c_n_ca_violation_mask],
                dim=-1,
            ),
            dim=-1,
        ).values
        violation_mask = torch.maximum(
            torch.nn.functional.pad(violation_mask, (0, 1)),
            torch.nn.functional.pad(violation_mask, (1, 0)),
        )
        return {
            "c_n_loss_mean": c_n_loss,
            "ca_c_n_loss_mean": ca_c_n_loss,
            "c_n_ca_loss_mean": c_n_ca_loss,
            "per_residue_loss_sum": per_residue_loss_sum,
            "per_residue_violation_mask": violation_mask,
        }

    def between_residue_clash_loss(
        self,
        predicted_positions,  # (batch, N_res, 14, 3) — all-atom coordinates
        atom_mask,            # (batch, N_res, 14)    — 1 if atom exists, 0 otherwise
        residue_types,        # (batch, N_res)         — integer residue type index (0–20)
        residue_index,        # (batch, N_res)
    ):
        batch, N_res = predicted_positions.shape[:2]
        overlap_tolerance = self.clash_overlap_tolerance

        # Flatten atoms: (batch, N_res*14, 3) and (batch, N_res*14)
        pos_flat = predicted_positions.reshape(batch, N_res * 14, 3)
        mask_flat = atom_mask.reshape(batch, N_res * 14)

        # VDW radii per atom: look up from registered buffer
        # residue_types: (batch, N_res) -> vdw: (batch, N_res, 14)
        residue_types_clamped = residue_types.clamp(max=20)
        vdw = self.vdw_table[residue_types_clamped]  # (batch, N_res, 14)
        vdw_flat = vdw.reshape(batch, N_res * 14)  # (batch, N_res*14)

        # Pairwise distances: (batch, N_res*14, N_res*14)
        diff = pos_flat[:, :, None, :] - pos_flat[:, None, :, :]  # (batch, M, M, 3)
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)   # (batch, M, M)

        # Pair mask: both atoms valid
        pair_mask = mask_flat[:, :, None] * mask_flat[:, None, :]  # (batch, M, M)

        atom_residue_index = residue_index.repeat_interleave(14, dim=1)
        unique_residue_pairs = (atom_residue_index[:, :, None] < atom_residue_index[:, None, :]).to(
            predicted_positions.dtype
        )
        pair_mask = pair_mask * unique_residue_pairs

        atom_slot_index = torch.arange(14, device=predicted_positions.device).repeat(N_res)
        atom_slot_index = atom_slot_index.unsqueeze(0).expand(batch, -1)
        residue_type_flat = residue_types.repeat_interleave(14, dim=1)

        c_n_bond = (
            (atom_slot_index[:, :, None] == 2)
            & (atom_slot_index[:, None, :] == 0)
            & (atom_residue_index[:, None, :] - atom_residue_index[:, :, None] == 1)
        ) | (
            (atom_slot_index[:, :, None] == 0)
            & (atom_slot_index[:, None, :] == 2)
            & (atom_residue_index[:, :, None] - atom_residue_index[:, None, :] == 1)
        )
        disulfide_bond = (
            (residue_type_flat[:, :, None] == 4)
            & (residue_type_flat[:, None, :] == 4)
            & (atom_slot_index[:, :, None] == 5)
            & (atom_slot_index[:, None, :] == 5)
        )
        pair_mask = pair_mask * (~c_n_bond).to(predicted_positions.dtype) * (~disulfide_bond).to(predicted_positions.dtype)

        # Overlap: vdw_i + vdw_j - tolerance - dist
        vdw_sum = vdw_flat[:, :, None] + vdw_flat[:, None, :]  # (batch, M, M)
        overlap = vdw_sum - overlap_tolerance - dist

        clash = torch.clamp(overlap, min=0) * pair_mask
        mean_loss = torch.sum(clash, dim=(1, 2)) / torch.sum(pair_mask, dim=(1, 2)).clamp(min=1e-6)
        per_atom_loss_sum = (torch.sum(clash, dim=1) + torch.sum(clash, dim=2)).reshape(batch, N_res, 14)
        clash_mask = pair_mask * (dist < (vdw_sum - overlap_tolerance))
        per_atom_clash_mask = torch.maximum(
            torch.amax(clash_mask, dim=1),
            torch.amax(clash_mask, dim=2),
        ).reshape(batch, N_res, 14)
        per_atom_num_clash = (torch.sum(clash_mask, dim=1) + torch.sum(clash_mask, dim=2)).reshape(batch, N_res, 14)
        return {
            "mean_loss": mean_loss,
            "per_atom_loss_sum": per_atom_loss_sum,
            "per_atom_clash_mask": per_atom_clash_mask,
            "per_atom_num_clash": per_atom_num_clash,
        }

    def within_residue_violation_loss(
        self,
        predicted_positions,  # (batch, N_res, 14, 3)
        atom_mask,            # (batch, N_res, 14)
        residue_types,        # (batch, N_res)
    ):
        residue_types_clamped = residue_types.clamp(max=20)
        lower_bound = self.distance_lower_bound_table[residue_types_clamped]
        upper_bound = self.distance_upper_bound_table[residue_types_clamped]

        distances = torch.sqrt(
            torch.sum(
                (predicted_positions[:, :, :, None, :] - predicted_positions[:, :, None, :, :]) ** 2,
                dim=-1,
            ) + 1e-8
        )
        pair_mask = atom_mask[:, :, :, None] * atom_mask[:, :, None, :]
        eye = torch.eye(14, device=predicted_positions.device, dtype=predicted_positions.dtype)
        pair_mask = pair_mask * (1.0 - eye.view(1, 1, 14, 14))
        bound_mask = ((lower_bound > 0.0) | (upper_bound > 0.0)).to(predicted_positions.dtype)
        pair_mask = pair_mask * bound_mask

        lower_violation = torch.clamp(lower_bound - distances, min=0.0)
        upper_violation = torch.clamp(distances - upper_bound, min=0.0)
        loss = (lower_violation + upper_violation) * pair_mask
        per_atom_loss_sum = torch.sum(loss, dim=-2) + torch.sum(loss, dim=-1)

        violations = pair_mask * ((distances < lower_bound) | (distances > upper_bound)).to(predicted_positions.dtype)
        per_atom_violations = torch.maximum(
            torch.amax(violations, dim=-2),
            torch.amax(violations, dim=-1),
        )
        per_atom_num_clash = torch.sum(violations, dim=-2) + torch.sum(violations, dim=-1)
        return {
            "per_atom_loss_sum": per_atom_loss_sum,
            "per_atom_violations": per_atom_violations,
            "per_atom_num_clash": per_atom_num_clash,
        }
