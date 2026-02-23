import torch
from typing import Optional
from utils import distance_bin
from structure_module import compute_all_atom_coordinates
from residue_constants import (
    between_res_bond_length_c_n, between_res_bond_length_stddev_c_n,
    between_res_cos_angles_c_n_ca, between_res_cos_angles_ca_c_n,
    chi_angles_mask,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group,
    restype_rigid_group_default_frame,
    restype_atom14_vdw_radius,
)

class AlphaFoldLoss(torch.nn.Module):
    default_frames: torch.Tensor
    lit_positions: torch.Tensor
    atom_frame_idx_table: torch.Tensor
    atom_mask_table: torch.Tensor

    def __init__(self, finetune = False):
        super().__init__()
        self.torsion_angle_loss = TorsionAngleLoss()
        self.plddt_loss = PLDDTLoss()
        self.distogram_loss = DistogramLoss()
        self.msa_loss = MSALoss()
        self.experimentally_resolved_loss = ExperimentallyResolvedLoss()
        self.structural_violation_loss = StructuralViolationLoss()
        self.aux_loss = AuxiliaryLoss()
        self.fape_loss = AllAtomFAPE()

        self.register_buffer("default_frames", torch.tensor(restype_rigid_group_default_frame))
        self.register_buffer("lit_positions", torch.tensor(restype_atom14_rigid_group_positions))
        self.register_buffer("atom_frame_idx_table", torch.tensor(restype_atom14_to_rigid_group))
        self.register_buffer("atom_mask_table", torch.tensor(restype_atom14_mask))

        self.finetune = finetune

    def forward(
            self,
            structure_model_prediction: dict,
            true_rotations: torch.Tensor,           # (b, N_res, 3, 3)
            true_translations: torch.Tensor,        # (b, N_res, 3)
            true_atom_positions: torch.Tensor,      # (b, N_res, 14, 3)
            true_torsion_angles: torch.Tensor,      # (b, N_res, 7, 2)
            experimentally_resolved_pred: torch.Tensor,
            experimentally_resolved_true: torch.Tensor,
            msa_pred: torch.Tensor,
            msa_true: torch.Tensor,
            msa_mask: torch.Tensor,
            plddt_pred: torch.Tensor,
            distogram_pred: torch.Tensor,
            res_types: torch.Tensor,                # (b, N_res) integer 0-20
            seq_mask: Optional[torch.Tensor] = None,  # (b, N_res) 1=valid, 0=padding
        ):
        pred_all_frames_R = structure_model_prediction["all_frames_R"]  # (batch, N_res, 8, 3, 3)
        pred_all_frames_t = structure_model_prediction["all_frames_t"]  # (batch, N_res, 8, 3)
        atom_coords = structure_model_prediction["atom14_coords"]   # (batch, N_res, 14, 3)
        atom_mask = structure_model_prediction["atom14_mask"]       # (batch, N_res, 14)

        true_all_frames_R, true_all_frames_t, _, _ = compute_all_atom_coordinates(
            true_translations,
            true_rotations,
            true_torsion_angles,
            res_types,
            self.default_frames,
            self.lit_positions,
            self.atom_frame_idx_table,
            self.atom_mask_table,
        )

        fape_loss = self.fape_loss(
                pred_all_frames_R, pred_all_frames_t, atom_coords, atom_mask,
                true_all_frames_R, true_all_frames_t, true_atom_positions,
                seq_mask=seq_mask,
            )

        # --- Derive true_torsion_angles_alt ---
        # For symmetric side chains, the alternative swaps equivalent atoms,
        # which is equivalent to negating the (sin, cos) of the relevant chi angle.
        # ASP(3), PHE(13), TYR(18): chi2 (torsion index 4)
        # GLU(6): chi3 (torsion index 5)
        true_torsion_angles_alt = true_torsion_angles.clone()
        chi2_sym = ((res_types == 3) | (res_types == 13) | (res_types == 18)).unsqueeze(-1)
        true_torsion_angles_alt[:, :, 4, :] = torch.where(
            chi2_sym, -true_torsion_angles[:, :, 4, :], true_torsion_angles[:, :, 4, :])
        chi3_sym = (res_types == 6).unsqueeze(-1)
        true_torsion_angles_alt[:, :, 5, :] = torch.where(
            chi3_sym, -true_torsion_angles[:, :, 5, :], true_torsion_angles[:, :, 5, :])

        aux_loss = self.aux_loss(structure_model_prediction, true_rotations, true_translations,
                                 true_torsion_angles, true_torsion_angles_alt, res_types,
                                 seq_mask=seq_mask)

        # --- Derive distogram_true ---
        # CB-CB distances (CA for GLY) binned into distance buckets
        is_gly = (res_types == 7)  # (batch, N_res)
        cb_idx = torch.where(is_gly, 1, 4)  # CA=1 for GLY, CB=4 otherwise
        cb_pos = torch.gather(
            true_atom_positions, 2,
            cb_idx[:, :, None, None].expand(-1, -1, 1, 3),
        ).squeeze(2)  # (batch, N_res, 3)
        n_dist_bins = distogram_pred.shape[-1]
        distogram_true = distance_bin(cb_pos, n_dist_bins)

        dist_loss = self.distogram_loss(distogram_pred, distogram_true)

        msa_loss = self.msa_loss(msa_pred, msa_true, msa_mask)

        # --- Derive plddt_true ---
        # Per-residue lDDT between predicted and true CA positions, then binned
        N_res = atom_coords.shape[1]
        with torch.no_grad():
            pred_ca = atom_coords[:, :, 1, :]       # (batch, N_res, 3)
            true_ca = true_atom_positions[:, :, 1, :]
            true_ca_dists = torch.cdist(true_ca, true_ca)  # (batch, N_res, N_res)
            pred_ca_dists = torch.cdist(pred_ca, pred_ca)
            # Include pairs within 15 Å in the true structure, exclude self
            inclusion = (true_ca_dists < 15.0).float() * (
                1.0 - torch.eye(N_res, device=pred_ca.device).unsqueeze(0))
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
            n_plddt_bins = plddt_pred.shape[-1]
            plddt_edges = torch.arange(1, n_plddt_bins, device=pred_ca.device).float() / n_plddt_bins
            plddt_bin_idx = torch.bucketize(lddt, plddt_edges)
            plddt_true = torch.nn.functional.one_hot(plddt_bin_idx, n_plddt_bins).float()

        plddt_loss = self.plddt_loss(plddt_pred, plddt_true, seq_mask=seq_mask)

        loss = 0.5*fape_loss + 0.5*aux_loss + 0.3*dist_loss + 2.0*msa_loss + 0.01*plddt_loss

        if self.finetune:
            exp_resolved_loss = self.experimentally_resolved_loss(experimentally_resolved_pred, experimentally_resolved_true)
            struct_violation_loss = self.structural_violation_loss(atom_coords, atom_mask, res_types)
            loss += 0.01*exp_resolved_loss + 1.0*struct_violation_loss

        return loss
    
class AuxiliaryLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.torsion_angle_loss = TorsionAngleLoss()
        self.fape_loss = BackboneFAPE()

    def forward(
            self,
            structure_model_prediction: dict,
            true_rotations: torch.Tensor,          # (b, N_res, 3, 3)
            true_translations: torch.Tensor,        # (b, N_res, 3)
            true_torsion_angles: torch.Tensor,      # (b, N_res, 7, 2)
            true_torsion_angles_alt: torch.Tensor,  # (b, N_res, 7, 2)
            res_types: torch.Tensor,                # (b, N_res)
            seq_mask: Optional[torch.Tensor] = None,
        ):
        traj_R = structure_model_prediction["traj_rotations"]          # (L, b, N_res, 3, 3)
        traj_t = structure_model_prediction["traj_translations"]       # (L, b, N_res, 3)
        traj_torsions = structure_model_prediction["traj_torsion_angles"]  # (L, b, N_res, 7, 2)

        num_layers = traj_R.shape[0]
        total_loss = torch.zeros(traj_R.shape[1], device=traj_R.device, dtype=traj_R.dtype)

        for l in range(num_layers):
            # Backbone FAPE: use translations as atom positions (CA coordinates)
            fape = self.fape_loss(
                traj_R[l], traj_t[l], traj_t[l],
                true_rotations, true_translations, true_translations,
                seq_mask=seq_mask,
            )
            torsion = self.torsion_angle_loss(
                traj_torsions[l], true_torsion_angles, true_torsion_angles_alt, res_types,
                seq_mask=seq_mask,
            )
            total_loss = total_loss + fape + torsion

        return total_loss / num_layers

class TorsionAngleLoss(torch.nn.Module):
    chi_mask_table: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "chi_mask_table",
            torch.tensor(chi_angles_mask + [[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        )

    def forward(
        self,
        torsion_angles: torch.Tensor,
        torsion_angles_true: torch.Tensor,
        torsion_angles_true_alt: torch.Tensor,
        res_types: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ):
        # torsion_angles shape: (batch, N_res, 7, 2)
        # seq_mask: (batch, N_res) — 1 for valid residues, 0 for padding

        norm = torch.sqrt(torch.sum(torsion_angles ** 2, dim=-1) + 1e-8)  # (batch, N_res, 7)
        pred_unit = torsion_angles / norm[..., None]  # (batch, N_res, 7, 2)

        true_dist_sq = torch.sum((torsion_angles_true - pred_unit) ** 2, dim=-1)  # (batch, N_res, 7)
        alt_true_dist_sq = torch.sum((torsion_angles_true_alt - pred_unit) ** 2, dim=-1)  # (batch, N_res, 7)
        torsion_dist_sq = torch.minimum(true_dist_sq, alt_true_dist_sq)

        chi_mask = self.chi_mask_table[res_types.long()].to(torsion_angles.dtype)  # (batch, N_res, 4)
        backbone_mask = torch.ones_like(chi_mask[..., :3])  # (batch, N_res, 3)
        torsion_mask = torch.cat([backbone_mask, chi_mask], dim=-1)  # (batch, N_res, 7)

        # Combine with seq_mask to exclude padded residues
        if seq_mask is not None:
            torsion_mask = torsion_mask * seq_mask.unsqueeze(-1)  # (batch, N_res, 7)

        normalizer = torsion_mask.sum(dim=(1, 2)).clamp(min=1.0)  # (batch,)

        torsion_loss = torch.sum(torsion_dist_sq * torsion_mask, dim=(1, 2)) / normalizer
        angle_norm_loss = torch.sum(torch.abs(norm - 1.0) * torsion_mask, dim=(1, 2)) / normalizer

        return torsion_loss + 0.02 * angle_norm_loss
    
class BackboneFAPE(torch.nn.Module):
    def __init__(self, d_clamp=10.0, eps=1e-4, Z=10.0):
        super().__init__()
        self.eps = eps
        self.d_clamp_val = d_clamp
        self.Z = Z

    def forward(self,
                predicted_rotations,      # (b, N_res, 3, 3)
                predicted_translations,   # (b, N_res, 3)
                predicted_atom_positions, # (b, N_atoms, 3)
                true_rotations,           # (b, N_res, 3, 3)
                true_translations,        # (b, N_res, 3)
                true_atom_positions,      # (b, N_atoms, 3)
                seq_mask: Optional[torch.Tensor] = None,  # (b, N_res)
    ):
        # Predicted inverse frames
        R_pred_inv = predicted_rotations.transpose(-1, -2)
        t_pred_inv = -torch.einsum('birc, bic -> bir', R_pred_inv, predicted_translations)

        # True inverse frames
        R_true_inv = true_rotations.transpose(-1, -2)
        t_true_inv = -torch.einsum('birc, bic -> bir', R_true_inv, true_translations)

        # Project ALL atoms through ALL frames (cross-product)
        # Result: (b, N_frames, N_atoms, 3)
        x_frames_pred = torch.einsum('biop, bjp -> bijo', R_pred_inv, predicted_atom_positions) \
                         + t_pred_inv[:, :, None, :]
        x_frames_true = torch.einsum('biop, bjp -> bijo', R_true_inv, true_atom_positions) \
                         + t_true_inv[:, :, None, :]

        # Distance: (b, N_frames, N_atoms)
        dist = torch.sqrt(
            torch.sum((x_frames_pred - x_frames_true) ** 2, dim=-1) + self.eps
        )

        # Clamp
        dist_clamped = torch.clamp(dist, max=self.d_clamp_val)

        if seq_mask is not None:
            # frame_mask: (b, N_frames) — valid frames from valid residues
            # atom_mask: (b, N_atoms) — valid atoms from valid residues
            frame_mask = seq_mask    # (b, N_res)
            atom_mask = seq_mask     # (b, N_res) — backbone atoms = residues
            # pair_mask: (b, N_frames, N_atoms)
            pair_mask = frame_mask[:, :, None] * atom_mask[:, None, :]
            pair_count = pair_mask.sum(dim=(-1, -2)).clamp(min=1)
            fape_loss = (1.0 / self.Z) * (dist_clamped * pair_mask).sum(dim=(-1, -2)) / pair_count
        else:
            fape_loss = (1.0 / self.Z) * torch.mean(dist_clamped, dim=(-1, -2))

        return fape_loss

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
                seq_mask: Optional[torch.Tensor] = None,  # (b, N_res)
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

        # Combine atom_mask with seq_mask for padded residues
        if seq_mask is not None:
            # Expand seq_mask to per-atom: (b, N_res) -> (b, N_res, 1) -> (b, N_res*14)
            seq_atom_mask = seq_mask.unsqueeze(-1).expand(-1, -1, n_atoms).reshape(b, N_res * n_atoms)
            flat_atom_mask = flat_atom_mask * seq_atom_mask
            # Frame mask: (b, N_res*8) — each residue has 8 frames
            frame_mask = seq_mask.unsqueeze(-1).expand(-1, -1, n_frames).reshape(b, N_res * n_frames)
        else:
            frame_mask = None

        # Predicted inverse frames
        R_pred_inv = pred_R.transpose(-1, -2)
        t_pred_inv = -torch.einsum('bfij, bfj -> bfi', R_pred_inv, pred_t)

        # True inverse frames
        R_true_inv = true_R.transpose(-1, -2)
        t_true_inv = -torch.einsum('bfij, bfj -> bfi', R_true_inv, true_t)

        # Project all atoms through all frames
        # Result: (b, N_frames, N_atoms, 3)
        x_frames_pred = torch.einsum('bfij, baj -> bfai', R_pred_inv, pred_pos) + t_pred_inv[:, :, None, :]
        x_frames_true = torch.einsum('bfij, baj -> bfai', R_true_inv, true_pos) + t_true_inv[:, :, None, :]

        # Distance: (b, N_frames, N_atoms)
        dist = torch.sqrt(
            torch.sum((x_frames_pred - x_frames_true) ** 2, dim=-1) + self.eps
        )

        # Clamp
        dist_clamped = torch.clamp(dist, max=self.d_clamp_val)

        # Masked average over atoms, then masked average over frames
        atom_count = flat_atom_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (b, 1)
        frame_means = (dist_clamped * flat_atom_mask[:, None, :]).sum(dim=-1) / atom_count  # (b, N_frames)

        if frame_mask is not None:
            frame_count = frame_mask.sum(dim=-1).clamp(min=1)  # (b,)
            fape_loss = (frame_means * frame_mask).sum(dim=-1) / (frame_count * self.Z)
        else:
            fape_loss = frame_means.mean(dim=-1) / self.Z

        return fape_loss

class PLDDTLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_plddt: torch.Tensor, true_plddt: torch.Tensor,
                seq_mask: Optional[torch.Tensor] = None):
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

        return conf_loss
    
class DistogramLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_distograms: torch.Tensor, true_distograms: torch.Tensor):
        # input shapes: (batch, N_res, N_res, num_dist_buckets)

        log_pred = torch.log_softmax(pred_distograms, dim=-1)

        vals = torch.einsum('bijc, bijc -> bij', true_distograms, log_pred)

        dist_loss = - torch.mean(vals, dim=(1,2))

        return dist_loss

class MSALoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, msa_preds: torch.Tensor, msa_true: torch.Tensor, msa_mask: torch.Tensor):
        # msa_preds: (batch, N_seq, N_res, n_msa_classes) — raw logits
        # msa_true:  (batch, N_seq, N_res, n_msa_classes) — one-hot targets
        # msa_mask:  (N_seq, N_res) — 1 for masked positions to predict, 0 otherwise

        log_pred = torch.log_softmax(msa_preds, dim=-1)

        # Per-position cross-entropy: (batch, N_seq, N_res)
        ce = -torch.einsum('bsic, bsic -> bsi', msa_true, log_pred)

        # Apply mask: (N_seq, N_res) -> (1, N_seq, N_res)
        mask = msa_mask.unsqueeze(0)
        ce = ce * mask

        # Average over masked positions only
        N_mask = torch.sum(mask).clamp(min=1)
        msa_loss = torch.sum(ce, dim=(1, 2)) / N_mask

        return msa_loss
    
class ExperimentallyResolvedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, exp_resolved_preds: torch.Tensor, exp_resolved_true: torch.Tensor):
        # exp_resolved_preds shape: (batch, N_res, 14) — raw logits
        # exp_resolved_true shape: (batch, N_res, 14) - binary, 1 if atom is exp resolved, 0 if not

        return torch.nn.functional.binary_cross_entropy_with_logits(
            exp_resolved_preds, exp_resolved_true, reduction='none'
        ).mean(dim=(1, 2))
    
class StructuralViolationLoss(torch.nn.Module):
    vdw_table: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer('vdw_table', torch.tensor(restype_atom14_vdw_radius))

    def forward(
        self,
        predicted_positions,  # (batch, N_res, 14, 3) — all-atom coordinates
        atom_mask,            # (batch, N_res, 14)    — 1 if atom exists, 0 otherwise
        residue_types,        # (batch, N_res)         — integer residue type index (0–20)
    ):
        return self.bond_length_loss(predicted_positions, atom_mask, residue_types) + self.bond_angle_loss(predicted_positions, atom_mask, residue_types) + self.clash_loss(predicted_positions, atom_mask, residue_types)

    def bond_length_loss(
        self,
        predicted_positions,  # (batch, N_res, 14, 3) — all-atom coordinates
        atom_mask,            # (batch, N_res, 14)    — 1 if atom exists, 0 otherwise
        residue_types,        # (batch, N_res)         — integer residue type index (0–20)
    ):
        # Between-residue C-N peptide bond: C of residue i to N of residue i+1
        # atom14 indices: N=0, CA=1, C=2
        C_i = predicted_positions[:, :-1, 2, :]     # (batch, N_res-1, 3)
        N_next = predicted_positions[:, 1:, 0, :]   # (batch, N_res-1, 3)

        d = torch.sqrt(torch.sum((C_i - N_next) ** 2, dim=-1) + 1e-8)  # (batch, N_res-1)

        # Ideal bond length depends on whether residue i+1 is proline (index 14)
        is_proline = (residue_types[:, 1:] == 14).float()  # (batch, N_res-1)
        d_ideal = (1.0 - is_proline) * between_res_bond_length_c_n[0] + \
                  is_proline * between_res_bond_length_c_n[1]
        d_stddev = (1.0 - is_proline) * between_res_bond_length_stddev_c_n[0] + \
                   is_proline * between_res_bond_length_stddev_c_n[1]

        # Mask: both C of residue i and N of residue i+1 must exist
        mask = atom_mask[:, :-1, 2] * atom_mask[:, 1:, 0]  # (batch, N_res-1)

        loss = ((d - d_ideal) / d_stddev) ** 2 * mask
        return torch.sum(loss, dim=-1) / torch.sum(mask, dim=-1).clamp(min=1)

    def bond_angle_loss(
        self,
        predicted_positions,  # (batch, N_res, 14, 3) — all-atom coordinates
        atom_mask,            # (batch, N_res, 14)    — 1 if atom exists, 0 otherwise
        residue_types,        # (batch, N_res)         — integer residue type index (0–20)
    ):
        # Between-residue angles via law of cosines
        # atom14 indices: N=0, CA=1, C=2
        CA_i = predicted_positions[:, :-1, 1, :]    # (batch, N_res-1, 3)
        C_i = predicted_positions[:, :-1, 2, :]     # (batch, N_res-1, 3)
        N_next = predicted_positions[:, 1:, 0, :]   # (batch, N_res-1, 3)
        CA_next = predicted_positions[:, 1:, 1, :]  # (batch, N_res-1, 3)

        eps = 1e-8

        # Pairwise squared distances
        d_CA_C_sq = torch.sum((CA_i - C_i) ** 2, dim=-1)       # (batch, N_res-1)
        d_C_N_sq = torch.sum((C_i - N_next) ** 2, dim=-1)
        d_CA_N_sq = torch.sum((CA_i - N_next) ** 2, dim=-1)
        d_N_CA_sq = torch.sum((N_next - CA_next) ** 2, dim=-1)
        d_C_CA_sq = torch.sum((C_i - CA_next) ** 2, dim=-1)

        d_CA_C = torch.sqrt(d_CA_C_sq + eps)
        d_C_N = torch.sqrt(d_C_N_sq + eps)
        d_N_CA = torch.sqrt(d_N_CA_sq + eps)

        # Angle 1: CA(i)-C(i)-N(i+1) — vertex at C(i)
        # cos = (d_CA_C^2 + d_C_N^2 - d_CA_N^2) / (2 * d_CA_C * d_C_N)
        cos_ca_c_n = (d_CA_C_sq + d_C_N_sq - d_CA_N_sq) / (2.0 * d_CA_C * d_C_N + eps)

        # Angle 2: C(i)-N(i+1)-CA(i+1) — vertex at N(i+1)
        # cos = (d_C_N^2 + d_N_CA^2 - d_C_CA^2) / (2 * d_C_N * d_N_CA)
        cos_c_n_ca = (d_C_N_sq + d_N_CA_sq - d_C_CA_sq) / (2.0 * d_C_N * d_N_CA + eps)

        # Mask: all 4 backbone atoms must exist
        mask = atom_mask[:, :-1, 1] * atom_mask[:, :-1, 2] * \
               atom_mask[:, 1:, 0] * atom_mask[:, 1:, 1]  # (batch, N_res-1)

        loss_1 = ((cos_ca_c_n - between_res_cos_angles_ca_c_n[0]) / between_res_cos_angles_ca_c_n[1]) ** 2
        loss_2 = ((cos_c_n_ca - between_res_cos_angles_c_n_ca[0]) / between_res_cos_angles_c_n_ca[1]) ** 2

        loss = (loss_1 + loss_2) * mask
        return torch.sum(loss, dim=-1) / torch.sum(mask, dim=-1).clamp(min=1)

    def clash_loss(
        self,
        predicted_positions,  # (batch, N_res, 14, 3) — all-atom coordinates
        atom_mask,            # (batch, N_res, 14)    — 1 if atom exists, 0 otherwise
        residue_types,        # (batch, N_res)         — integer residue type index (0–20)
    ):
        batch, N_res = predicted_positions.shape[:2]
        overlap_tolerance = 1.5

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

        # Exclude same-residue and adjacent-residue pairs (seq separation < 2)
        residue_idx = torch.arange(N_res, device=predicted_positions.device)
        # Each atom's residue index: (N_res*14,)
        atom_res_idx = residue_idx.repeat_interleave(14)
        # Sequence separation matrix: (M, M)
        seq_sep = torch.abs(atom_res_idx[None, :] - atom_res_idx[:, None])  # (M, M)
        sep_mask = (seq_sep >= 2).float().unsqueeze(0)  # (1, M, M)

        pair_mask = pair_mask * sep_mask

        # Overlap: vdw_i + vdw_j - tolerance - dist
        vdw_sum = vdw_flat[:, :, None] + vdw_flat[:, None, :]  # (batch, M, M)
        overlap = vdw_sum - overlap_tolerance - dist

        clash = torch.clamp(overlap, min=0) * pair_mask
        return torch.sum(clash, dim=(1, 2)) / torch.sum(pair_mask, dim=(1, 2)).clamp(min=1)
