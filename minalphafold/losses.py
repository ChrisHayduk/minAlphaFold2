import torch
from residue_constants import (
    between_res_bond_length_c_n, between_res_bond_length_stddev_c_n,
    between_res_cos_angles_c_n_ca, between_res_cos_angles_ca_c_n,
    restype_atom14_vdw_radius,
)

class AlphaFoldLoss(torch.nn.Module):
    def __init__(self, finetune = False):
        super().__init__()
        self.torsion_angle_loss = TorsionAngleLoss()
        self.plddt_loss = PLDDTLoss()
        self.distogram_loss = DistogramLoss()
        self.msa_loss = MSALoss()
        self.experimentally_resolved_loss = ExperimentallyResolvedLoss()
        self.structural_violation_loss = StructuralViolationLoss()
        self.aux_loss = AuxiliaryLoss()

        self.finetune = finetune

    def forward(
            self, 
            structure_model_prediction: dict, 
            msa_representation: torch.Tensor,
            true_rotations,           # (b, N_res, 3, 3)
            true_translations,        # (b, N_res, 3)
            true_atom_positions,      # (b, N_atoms, 3)
            true_torsion_angles
        ):
        loss = 0
        
        if self.finetune:
            loss = 0.5*self.fape_loss() + 0.5*self.aux_loss() + 0.3*self.distogram_loss() + 2.0*self.msa_loss() + 0.01*self.plddt_loss() + 0.01*self.experimentally_resolved_loss() + 1.0*self.structural_violation_loss()
        else:
            loss = 0.5*self.fape_loss() + 0.5*self.aux_loss() + 0.3*self.distogram_loss() + 2.0*self.msa_loss() + 0.01*self.plddt_loss()

        return loss
    
class AuxiliaryLoss(torch.nn.Module):
    def __init__(self):
        self.torsion_angle_loss = TorsionAngleLoss()
        self.fape_loss = FAPELoss()

    def forward(
            self, 
            structure_model_prediction: dict,
            true_rotations: torch.Tensor,
            true_translation: torch.Tensor,
            true_atom_positions: torch.Tensor,
            true_torsion_angles: torch.Tensor,
        ):
            pass

class TorsionAngleLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, torsion_angles: torch.Tensor, torsion_angles_true: torch.Tensor, torsion_angles_true_alt: torch.Tensor):
        # torsion_angles shape: (batch, N_res, 7, 2)

        norm = torch.sqrt(torch.sum(torsion_angles**2, dim=-1, keepdim=True))

        torsion_angles = torsion_angles / norm

        true_dist = torch.sqrt(torch.sum((torsion_angles_true - torsion_angles)**2, dim=-1, keepdim=True))

        alt_true_dist = torch.sqrt(torch.sum((torsion_angles_true_alt - torsion_angles)**2, dim=-1, keepdim=True))

        torsion_loss = torch.mean(torch.minimum(true_dist, alt_true_dist), dim=(2, 3))

        angle_norm_loss = torch.mean(torch.abs(norm - 1), dim=(2, 3))

        return torsion_loss + 0.02 * angle_norm_loss
    
class FAPELoss(torch.nn.Module):
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
                true_atom_positions       # (b, N_atoms, 3)
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

        # Clamp and average
        dist_clamped = torch.clamp(dist, max=self.d_clamp_val)

        fape_loss = (1.0 / self.Z) * torch.mean(dist_clamped, dim=(-1, -2))

        return fape_loss
    
class PLDDTLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_plddt: torch.Tensor, true_plddt: torch.Tensor):
        # Input shapes: (batch, N_res, n_plddt_bins)
        
        log_pred = torch.log_softmax(pred_plddt, dim=-1)

        # Per-residue cross-entropy: (batch, N_res)
        conf_loss = -torch.einsum('bic, bic -> bi', true_plddt, log_pred)

        # Mean over residues: (batch,)
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
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, ground_truth: torch.Tensor):
        # preds shape: (batch, N_res, 14)
        # groud_truth shape: (batch, N_res, 14) - binary, 1 if atom is exp resolved, 0 if not

        probs = torch.sigmoid(preds)

        log_probs = torch.log(probs + self.eps)
        log_inv_probs = torch.log(1 - probs + self.eps)

        exp_resolved_loss = torch.mean(- ground_truth * log_probs - (1 - ground_truth) * log_inv_probs, dim=(1,2))

        return exp_resolved_loss
    
class StructuralViolationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
        predicted_positions,  # (batch, N_res, 14, 3) — all-atom coordinates
        atom_mask,            # (batch, N_res, 14)    — 1 if atom exists, 0 otherwise
        residue_types,        # (batch, N_res)         — integer residue type index (0–20)
    ):
        return self.bond_length_loss(predicted_positions, atom_mask, residue_types) + self.bond_angle_loss(predicted_positions, atom_mask, residue_types) + self.clash_loss(predicted_positions, atom_mask, residue_types)

    def bond_length_loss(self,
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
        return torch.sum(loss) / torch.sum(mask).clamp(min=1)

    def bond_angle_loss(self,
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
        return torch.sum(loss) / torch.sum(mask).clamp(min=1)

    def clash_loss(self,
        predicted_positions,  # (batch, N_res, 14, 3) — all-atom coordinates
        atom_mask,            # (batch, N_res, 14)    — 1 if atom exists, 0 otherwise
        residue_types,        # (batch, N_res)         — integer residue type index (0–20)
    ):
        batch, N_res = predicted_positions.shape[:2]
        overlap_tolerance = 1.5

        # Flatten atoms: (batch, N_res*14, 3) and (batch, N_res*14)
        pos_flat = predicted_positions.reshape(batch, N_res * 14, 3)
        mask_flat = atom_mask.reshape(batch, N_res * 14)

        # VDW radii per atom: look up from precomputed table
        # residue_types: (batch, N_res) -> vdw: (batch, N_res, 14)
        vdw_table = torch.tensor(restype_atom14_vdw_radius, device=predicted_positions.device)
        residue_types_clamped = residue_types.clamp(max=20)
        vdw = vdw_table[residue_types_clamped]  # (batch, N_res, 14)
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
        return torch.sum(clash) / torch.sum(pair_mask).clamp(min=1)