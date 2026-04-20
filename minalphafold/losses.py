import torch
from typing import Optional

try:
    from .utils import distance_bin
    from .residue_constants import (
        between_res_bond_length_c_n, between_res_bond_length_stddev_c_n,
        between_res_cos_angles_c_n_ca, between_res_cos_angles_ca_c_n,
        chi_angles_mask,
        make_atom14_dists_bounds,
        restype_atom14_vdw_radius,
    )
except ImportError:  # pragma: no cover - compatibility for direct module imports in tests/scripts.
    from utils import distance_bin
    from residue_constants import (
        between_res_bond_length_c_n, between_res_bond_length_stddev_c_n,
        between_res_cos_angles_c_n_ca, between_res_cos_angles_ca_c_n,
        chi_angles_mask,
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
    eps: float = 1e-8,
) -> torch.Tensor:
    """AF2/OpenFold-style frame aligned point error."""
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

    error_distance = torch.sqrt(
        torch.sum((local_predicted_positions - local_true_positions) ** 2, dim=-1) + eps
    )
    if l1_clamp_distance is not None:
        error_distance = torch.clamp(error_distance, min=0.0, max=l1_clamp_distance)

    normalized_error = error_distance / length_scale
    normalized_error = normalized_error * frames_mask[..., :, None]
    normalized_error = normalized_error * positions_mask[..., None, :]

    if pair_mask is not None:
        normalized_error = normalized_error * pair_mask
        mask = frames_mask[..., :, None] * positions_mask[..., None, :] * pair_mask
        return torch.sum(normalized_error, dim=(-1, -2)) / (eps + torch.sum(mask, dim=(-1, -2)))

    normalized_error = torch.sum(normalized_error, dim=-1)
    normalized_error = normalized_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    normalized_error = torch.sum(normalized_error, dim=-1)
    normalized_error = normalized_error / (eps + torch.sum(positions_mask, dim=-1))
    return normalized_error


class AlphaFoldLoss(torch.nn.Module):
    def __init__(self, finetune = False, use_clamped_fape: Optional[float] = None):
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

        # AF2 monomer head weights:
        # training   = 0.5 L_FAPE + 0.5 L_aux + 0.3 L_dist + 2.0 L_msa + 0.01 L_conf
        # finetuning = training + 0.01 L_exp_resolved + 1.0 L_viol
        self.sidechain_weight_frac = 0.5
        self.distogram_weight = 0.3
        self.msa_weight = 2.0
        self.confidence_weight = 0.01
        self.experimentally_resolved_weight = 0.01
        self.structural_violation_weight = 1.0

        self.finetune = finetune
        # use_clamped_fape: None = clamped only (AF2 initial training default)
        # 0.0 = fully unclamped, 0.9 = 90% clamped + 10% unclamped (AF2 fine-tuning default)
        self.use_clamped_fape = use_clamped_fape
        # coordinate_loss_weight: direct Cartesian MSE on atom14 positions (bypasses FAPE indirection)
        self.coordinate_loss_weight = 0.0

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
            use_clamped_fape=self.use_clamped_fape,
        )
        chi_loss = self.torsion_angle_loss(
            structure_model_prediction["traj_torsion_angles"],
            structure_model_prediction["traj_torsion_angles_unnormalized"],
            true_torsion_angles,
            true_torsion_angles_alt,
            true_torsion_mask,
            res_types,
            seq_mask=seq_mask,
        )

        # --- Derive distogram_true ---
        # CB-CB distances (CA for GLY) binned into distance buckets
        is_gly = (res_types == 7)  # (batch, N_res)
        cb_idx = torch.where(is_gly, 1, 4)  # CA=1 for GLY, CB=4 otherwise
        cb_pos = torch.gather(
            true_atom_positions, 2,
            cb_idx[:, :, None, None].expand(-1, -1, 1, 3),
        ).squeeze(2)  # (batch, N_res, 3)
        n_dist_bins = distogram_pred.shape[-1]
        cb_mask = torch.gather(
            true_atom_mask, 2,
            cb_idx[:, :, None],
        ).squeeze(-1)
        distogram_true = distance_bin(cb_pos, n_dist_bins)
        dist_pair_mask = cb_mask[:, :, None] * cb_mask[:, None, :]
        if seq_mask is not None:
            dist_pair_mask = dist_pair_mask * (seq_mask[:, :, None] * seq_mask[:, None, :])

        dist_loss = self.distogram_loss(distogram_pred, distogram_true, pair_mask=dist_pair_mask)

        msa_loss = self.msa_loss(masked_msa_pred, masked_msa_target, masked_msa_mask)

        # --- Derive plddt_true ---
        # Per-residue lDDT between predicted and true CA positions, then binned
        N_res = atom_coords.shape[1]
        with torch.no_grad():
            pred_ca = atom_coords[:, :, 1, :]       # (batch, N_res, 3)
            true_ca = true_atom_positions[:, :, 1, :]
            true_ca_mask = true_atom_mask[:, :, 1]
            true_ca_dists = torch.cdist(true_ca, true_ca)  # (batch, N_res, N_res)
            pred_ca_dists = torch.cdist(pred_ca, pred_ca)
            # Include pairs within 15 Å in the true structure, exclude self
            inclusion = (true_ca_dists < 15.0).float() * (
                1.0 - torch.eye(N_res, device=pred_ca.device).unsqueeze(0))
            inclusion = inclusion * (true_ca_mask[:, :, None] * true_ca_mask[:, None, :])
            # Mask out padded residues from lDDT inclusion set
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

        weighted_backbone_loss = (1.0 - self.sidechain_weight_frac) * backbone_loss
        weighted_sidechain_fape_loss = self.sidechain_weight_frac * sidechain_loss
        weighted_chi_loss = chi_loss
        fape_loss = weighted_backbone_loss + weighted_sidechain_fape_loss
        structure_loss = fape_loss + weighted_chi_loss
        weighted_distogram_loss = self.distogram_weight * dist_loss
        weighted_msa_loss = self.msa_weight * msa_loss
        weighted_plddt_loss = self.confidence_weight * plddt_loss
        loss = structure_loss + weighted_distogram_loss + weighted_msa_loss + weighted_plddt_loss

        # Direct Cartesian coordinate loss — bypasses FAPE frame indirection
        if self.coordinate_loss_weight > 0:
            combined_mask = atom_mask
            if true_atom_mask is not None:
                combined_mask = combined_mask * true_atom_mask
            if seq_mask is not None:
                combined_mask = combined_mask * seq_mask[:, :, None]
            coord_error = torch.sum((atom_coords - true_atom_positions) ** 2, dim=-1)  # (b, N, 14)
            coord_loss = (coord_error * combined_mask).sum(dim=(1, 2)) / combined_mask.sum(dim=(1, 2)).clamp(min=1)
            weighted_coord_loss = self.coordinate_loss_weight * coord_loss
            loss = loss + weighted_coord_loss
        else:
            coord_loss = atom_coords.new_zeros(atom_coords.shape[0])
            weighted_coord_loss = coord_loss

        loss_terms = {
            "loss": loss,
            "structure_loss": structure_loss,
            "fape_loss": fape_loss,
            "backbone_loss": backbone_loss,
            "sidechain_fape_loss": sidechain_loss,
            "chi_loss": chi_loss,
            "distogram_loss": dist_loss,
            "msa_loss": msa_loss,
            "plddt_loss": plddt_loss,
            "coordinate_loss": coord_loss,
            "weighted_backbone_loss": weighted_backbone_loss,
            "weighted_sidechain_fape_loss": weighted_sidechain_fape_loss,
            "weighted_chi_loss": weighted_chi_loss,
            "weighted_distogram_loss": weighted_distogram_loss,
            "weighted_msa_loss": weighted_msa_loss,
            "weighted_plddt_loss": weighted_plddt_loss,
            "weighted_coordinate_loss": weighted_coord_loss,
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

        return total_loss / num_layers


def select_best_atom14_ground_truth(
    predicted_atom_positions: torch.Tensor,
    true_atom_positions: torch.Tensor,
    true_atom_mask: torch.Tensor,
    true_atom_positions_alt: torch.Tensor,
    true_atom_mask_alt: torch.Tensor,
    true_atom_is_ambiguous: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_residues, num_atoms = predicted_atom_positions.shape[:3]

    def pairwise_distances(atom14_positions: torch.Tensor) -> torch.Tensor:
        flat_positions = atom14_positions.reshape(batch_size, num_residues * num_atoms, 3)
        flat_distances = torch.cdist(flat_positions, flat_positions)
        return flat_distances.reshape(batch_size, num_residues, num_atoms, num_residues, num_atoms).permute(0, 1, 3, 2, 4)

    pred_dists = pairwise_distances(predicted_atom_positions)
    true_dists = pairwise_distances(true_atom_positions)
    alt_true_dists = pairwise_distances(true_atom_positions_alt)

    error = torch.sqrt(1e-10 + (pred_dists - true_dists) ** 2)
    alt_error = torch.sqrt(1e-10 + (pred_dists - alt_true_dists) ** 2)

    ambiguity_mask = (
        true_atom_mask[:, :, None, :, None]
        * true_atom_is_ambiguous[:, :, None, :, None]
        * true_atom_mask[:, None, :, None, :]
        * (1.0 - true_atom_is_ambiguous[:, None, :, None, :])
    )

    per_res_error = torch.sum(ambiguity_mask * error, dim=(2, 3, 4))
    per_res_alt_error = torch.sum(ambiguity_mask * alt_error, dim=(2, 3, 4))
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
    chi_mask_table: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "chi_mask_table",
            torch.tensor(chi_angles_mask + [[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        )
        self.chi_weight = 0.5
        self.angle_norm_weight = 0.01

    def forward(
        self,
        torsion_angles: torch.Tensor,
        unnormalized_torsion_angles: torch.Tensor,
        torsion_angles_true: torch.Tensor,
        torsion_angles_true_alt: torch.Tensor,
        torsion_mask_true: torch.Tensor,
        res_types: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ):
        if torsion_angles.ndim == 4:
            torsion_angles = torsion_angles.unsqueeze(0)
            unnormalized_torsion_angles = unnormalized_torsion_angles.unsqueeze(0)

        pred_chi = torsion_angles[..., 3:, :]
        true_chi = torsion_angles_true.unsqueeze(0)[..., 3:, :]
        true_chi_alt = torsion_angles_true_alt.unsqueeze(0)[..., 3:, :]
        chi_mask = self.chi_mask_table[res_types.long()].to(torsion_angles.dtype).unsqueeze(0)
        torsion_mask = chi_mask * torsion_mask_true.unsqueeze(0)[..., 3:]

        if seq_mask is not None:
            torsion_mask = torsion_mask * seq_mask.unsqueeze(0).unsqueeze(-1)

        true_dist_sq = torch.sum((true_chi - pred_chi) ** 2, dim=-1)
        alt_true_dist_sq = torch.sum((true_chi_alt - pred_chi) ** 2, dim=-1)
        torsion_dist_sq = torch.minimum(true_dist_sq, alt_true_dist_sq)

        torsion_normalizer = torsion_mask.sum(dim=(0, 2, 3)).clamp(min=1.0)
        torsion_loss = torch.sum(torsion_dist_sq * torsion_mask, dim=(0, 2, 3)) / torsion_normalizer

        angle_norm = torch.sqrt(torch.sum(unnormalized_torsion_angles ** 2, dim=-1) + 1e-8)
        if seq_mask is not None:
            angle_norm_mask = seq_mask.unsqueeze(0).unsqueeze(-1).expand_as(angle_norm)
        else:
            angle_norm_mask = torch.ones_like(angle_norm)
        norm_normalizer = angle_norm_mask.sum(dim=(0, 2, 3)).clamp(min=1.0)
        angle_norm_loss = torch.sum(torch.abs(angle_norm - 1.0) * angle_norm_mask, dim=(0, 2, 3)) / norm_normalizer

        return self.chi_weight * torsion_loss + self.angle_norm_weight * angle_norm_loss
    
class BackboneFAPE(torch.nn.Module):
    def __init__(self, d_clamp=10.0, eps=1e-4, Z=10.0):
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
    def __init__(self, d_clamp=10.0, eps=1e-4, Z=10.0):
        super().__init__()
        self.eps = eps
        self.d_clamp_val = d_clamp
        self.Z = Z

    def forward(self,
                predicted_frames_R,       # (b, N_res, 8, 3, 3)
                predicted_frames_t,       # (b, N_res, 8, 3)
                predicted_atom_positions, # (b, N_res, 14, 3)
                atom_mask,                # (b, N_res, 14)
                true_frames_R,            # (b, N_res, 8, 3, 3)
                true_frames_t,            # (b, N_res, 8, 3)
                true_atom_positions,      # (b, N_res, 14, 3)
                true_atom_mask: Optional[torch.Tensor] = None,  # (b, N_res, 14)
                seq_mask: Optional[torch.Tensor] = None,  # (b, N_res)
                frame_mask: Optional[torch.Tensor] = None,  # (b, N_res, 8)
                rigid_group_mask: Optional[torch.Tensor] = None,  # (21, 8)
                aatype: Optional[torch.Tensor] = None,  # (b, N_res)
                use_clamped_fape: Optional[float] = None,
    ):
        b, N_res, n_frames = predicted_frames_R.shape[:3]
        n_atoms = predicted_atom_positions.shape[2]

        # Flatten rigid-group frames and atoms.
        pred_R = predicted_frames_R.reshape(b, N_res * n_frames, 3, 3)
        pred_t = predicted_frames_t.reshape(b, N_res * n_frames, 3)
        true_R = true_frames_R.reshape(b, N_res * n_frames, 3, 3)
        true_t = true_frames_t.reshape(b, N_res * n_frames, 3)

        # Flatten atoms: (b, N_res*14, 3) and mask: (b, N_res*14)
        pred_pos = predicted_atom_positions.reshape(b, N_res * n_atoms, 3)
        true_pos = true_atom_positions.reshape(b, N_res * n_atoms, 3)
        flat_atom_mask = atom_mask.reshape(b, N_res * n_atoms)
        if true_atom_mask is not None:
            flat_atom_mask = flat_atom_mask * true_atom_mask.reshape(b, N_res * n_atoms)

        # Per-residue-type rigid group existence mask
        if frame_mask is not None:
            group_mask = frame_mask.to(predicted_frames_R.dtype)
        elif rigid_group_mask is not None and aatype is not None:
            group_mask = rigid_group_mask[aatype.long()]  # (b, N_res, 8)
        else:
            group_mask = predicted_frames_R.new_ones(b, N_res, n_frames)

        # Combine atom_mask and frame mask with seq_mask for padded residues
        if seq_mask is not None:
            seq_atom_mask = seq_mask[:, :, None].expand(-1, -1, n_atoms).reshape(b, N_res * n_atoms)
            flat_atom_mask = flat_atom_mask * seq_atom_mask
            frame_mask = (seq_mask[:, :, None] * group_mask).reshape(b, N_res * n_frames)
        else:
            frame_mask = group_mask.reshape(b, N_res * n_frames)

        fape_args = dict(
            predicted_rotations=pred_R,
            predicted_translations=pred_t,
            true_rotations=true_R,
            true_translations=true_t,
            predicted_positions=pred_pos,
            true_positions=true_pos,
            frames_mask=frame_mask,
            positions_mask=flat_atom_mask,
            length_scale=self.Z,
            eps=self.eps,
        )

        if use_clamped_fape is None:
            return frame_aligned_point_error(
                **fape_args,
                l1_clamp_distance=self.d_clamp_val,
            )

        clamped = frame_aligned_point_error(
            **fape_args,
            l1_clamp_distance=self.d_clamp_val,
        )
        unclamped = frame_aligned_point_error(
            **fape_args,
            l1_clamp_distance=None,
        )
        return clamped * use_clamped_fape + unclamped * (1.0 - use_clamped_fape)

class PLDDTLoss(torch.nn.Module):
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
        # pred_plddt, true_plddt: (batch, N_res, n_plddt_bins)
        # seq_mask: (batch, N_res) — 1 for valid residues, 0 for padding

        log_pred = torch.log_softmax(pred_plddt, dim=-1)

        # Per-residue cross-entropy: (batch, N_res)
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
            dist_loss = - torch.mean(vals, dim=(1,2))

        return dist_loss

class MSALoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, msa_preds: torch.Tensor, msa_true: torch.Tensor, masked_msa_mask: torch.Tensor):
        # msa_preds: (batch, N_seq, N_res, n_msa_classes) — raw logits
        # msa_true:  (batch, N_seq, N_res, n_msa_classes) — one-hot targets
        # masked_msa_mask:  (batch, N_seq, N_res) or (N_seq, N_res)

        log_pred = torch.log_softmax(msa_preds, dim=-1)

        # Per-position cross-entropy: (batch, N_seq, N_res)
        ce = -torch.einsum('bsic, bsic -> bsi', msa_true, log_pred)

        mask = masked_msa_mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        ce = ce * mask

        # Average over masked positions only
        N_mask = torch.sum(mask, dim=(1, 2)).clamp(min=1)
        msa_loss = torch.sum(ce, dim=(1, 2)) / N_mask

        return msa_loss
    
class ExperimentallyResolvedLoss(torch.nn.Module):
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
        # Canonical AF2 semantics: predict atom37 resolution, mask BCE by atom existence.
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
