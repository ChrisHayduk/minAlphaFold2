from __future__ import annotations

import torch

from residue_constants import (
    chi_angles_atoms,
    chi_angles_mask,
    restype_1to3,
    restype_name_to_atom14_names,
    restypes,
)


CHI_ATOM_INDICES = torch.full((21, 4, 4), fill_value=-1, dtype=torch.long)
CHI_EXISTS = torch.zeros((21, 4), dtype=torch.float32)

_restype_to_atom14 = {
    residue_name: {atom_name: atom_index for atom_index, atom_name in enumerate(atom_names) if atom_name}
    for residue_name, atom_names in restype_name_to_atom14_names.items()
}

for restype_index, restype_letter in enumerate(restypes):
    residue_name = restype_1to3[restype_letter]
    atom_lookup = _restype_to_atom14[residue_name]
    for chi_index, atom_names in enumerate(chi_angles_atoms[residue_name]):
        CHI_EXISTS[restype_index, chi_index] = chi_angles_mask[restype_index][chi_index]
        CHI_ATOM_INDICES[restype_index, chi_index] = torch.tensor(
            [atom_lookup[name] for name in atom_names],
            dtype=torch.long,
        )


def safe_normalize(vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vectors / torch.sqrt(torch.sum(vectors ** 2, dim=-1, keepdim=True) + eps)


def gather_atom14(atom14_tensor: torch.Tensor, atom_indices: torch.Tensor) -> torch.Tensor:
    if atom14_tensor.shape[-1] == 14:
        return torch.gather(atom14_tensor, atom14_tensor.ndim - 1, atom_indices.unsqueeze(-1)).squeeze(-1)

    if atom14_tensor.ndim >= 2 and atom14_tensor.shape[-2] == 14:
        gather_index = atom_indices.unsqueeze(-1).unsqueeze(-1).expand(*atom_indices.shape, 1, atom14_tensor.shape[-1])
        return torch.gather(atom14_tensor, atom14_tensor.ndim - 2, gather_index).squeeze(-2)

    raise ValueError(f"Could not find atom14 dimension in tensor with shape {atom14_tensor.shape}")


def dihedral_sin_cos(points_a: torch.Tensor, points_b: torch.Tensor,
                     points_c: torch.Tensor, points_d: torch.Tensor) -> torch.Tensor:
    b0 = points_b - points_a
    b1 = points_c - points_b
    b2 = points_d - points_c

    b1_unit = safe_normalize(b1)

    v = b0 - torch.sum(b0 * b1_unit, dim=-1, keepdim=True) * b1_unit
    w = b2 - torch.sum(b2 * b1_unit, dim=-1, keepdim=True) * b1_unit

    v_unit = safe_normalize(v)
    w_unit = safe_normalize(w)

    cos_angle = torch.sum(v_unit * w_unit, dim=-1).clamp(min=-1.0, max=1.0)
    sin_angle = torch.sum(torch.cross(b1_unit, v_unit, dim=-1) * w_unit, dim=-1)
    return torch.stack([sin_angle, cos_angle], dim=-1)


def backbone_frames(atom14_positions: torch.Tensor, atom14_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_pos = atom14_positions[..., 0, :]
    ca_pos = atom14_positions[..., 1, :]
    c_pos = atom14_positions[..., 2, :]

    n_mask = atom14_mask[..., 0]
    ca_mask = atom14_mask[..., 1]
    c_mask = atom14_mask[..., 2]
    valid_mask = (n_mask * ca_mask * c_mask).to(atom14_positions.dtype)

    ex = safe_normalize(c_pos - ca_pos)
    ey_seed = n_pos - ca_pos
    ez = safe_normalize(torch.cross(ex, ey_seed, dim=-1))
    ey = safe_normalize(torch.cross(ez, ex, dim=-1))

    rotations = torch.stack([ex, ey, ez], dim=-1)
    identity = torch.eye(3, device=atom14_positions.device, dtype=atom14_positions.dtype)
    rotations = torch.where(valid_mask[..., None, None] > 0, rotations, identity)
    translations = torch.where(valid_mask[..., None] > 0, ca_pos, torch.zeros_like(ca_pos))

    return rotations, translations, valid_mask


def pseudo_beta_positions(
    atom14_positions: torch.Tensor,
    atom14_mask: torch.Tensor,
    aatype: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    is_gly = (aatype == 7)
    atom_indices = torch.where(is_gly, torch.ones_like(aatype), torch.full_like(aatype, 4))
    positions = gather_atom14(atom14_positions, atom_indices)
    mask = gather_atom14(atom14_mask, atom_indices)
    return positions, mask


def torsion_angles(
    atom14_positions: torch.Tensor,
    atom14_mask: torch.Tensor,
    aatype: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_shape = atom14_positions.shape[:-2] + (7, 2)
    mask_shape = atom14_positions.shape[:-2] + (7,)

    angles = atom14_positions.new_zeros(output_shape)
    angle_mask = atom14_mask.new_zeros(mask_shape)

    n_pos = atom14_positions[..., 0, :]
    ca_pos = atom14_positions[..., 1, :]
    c_pos = atom14_positions[..., 2, :]

    n_mask = atom14_mask[..., 0]
    ca_mask = atom14_mask[..., 1]
    c_mask = atom14_mask[..., 2]

    omega_mask = ca_mask[..., :-1] * c_mask[..., :-1] * n_mask[..., 1:] * ca_mask[..., 1:]
    phi_mask = c_mask[..., :-1] * n_mask[..., 1:] * ca_mask[..., 1:] * c_mask[..., 1:]
    psi_mask = n_mask[..., :-1] * ca_mask[..., :-1] * c_mask[..., :-1] * n_mask[..., 1:]

    angles[..., 1:, 0, :] = dihedral_sin_cos(ca_pos[..., :-1, :], c_pos[..., :-1, :], n_pos[..., 1:, :], ca_pos[..., 1:, :])
    angles[..., 1:, 1, :] = dihedral_sin_cos(c_pos[..., :-1, :], n_pos[..., 1:, :], ca_pos[..., 1:, :], c_pos[..., 1:, :])
    angles[..., :-1, 2, :] = dihedral_sin_cos(n_pos[..., :-1, :], ca_pos[..., :-1, :], c_pos[..., :-1, :], n_pos[..., 1:, :])

    angle_mask[..., 1:, 0] = omega_mask
    angle_mask[..., 1:, 1] = phi_mask
    angle_mask[..., :-1, 2] = psi_mask

    chi_atom_indices = CHI_ATOM_INDICES.to(device=aatype.device)
    chi_exists = CHI_EXISTS.to(device=aatype.device, dtype=atom14_positions.dtype)

    residue_chi_indices = chi_atom_indices[aatype.long()]
    residue_chi_exists = chi_exists[aatype.long()]

    for chi_index in range(4):
        atom_indices = residue_chi_indices[..., chi_index, :]
        safe_atom_indices = atom_indices.clamp(min=0)
        gathered_positions = []
        gathered_masks = []
        for atom_offset in range(4):
            gathered_positions.append(gather_atom14(atom14_positions, safe_atom_indices[..., atom_offset]))
            gathered_masks.append(
                gather_atom14(atom14_mask, safe_atom_indices[..., atom_offset])
                * (atom_indices[..., atom_offset] >= 0).to(atom14_positions.dtype)
            )

        valid_mask = residue_chi_exists[..., chi_index]
        for gathered_mask in gathered_masks:
            valid_mask = valid_mask * gathered_mask

        chi_angles = dihedral_sin_cos(
            gathered_positions[0],
            gathered_positions[1],
            gathered_positions[2],
            gathered_positions[3],
        )

        angles[..., 3 + chi_index, :] = chi_angles
        angle_mask[..., 3 + chi_index] = valid_mask

    angles = angles * angle_mask[..., None]
    return angles, angle_mask


def alternative_torsion_angles(torsion_angles_tensor: torch.Tensor, aatype: torch.Tensor) -> torch.Tensor:
    alternative = torsion_angles_tensor.clone()

    chi2_symmetric = ((aatype == 3) | (aatype == 13) | (aatype == 18)).unsqueeze(-1)
    chi3_symmetric = (aatype == 6).unsqueeze(-1)

    alternative[..., 4, :] = torch.where(
        chi2_symmetric,
        -torsion_angles_tensor[..., 4, :],
        torsion_angles_tensor[..., 4, :],
    )
    alternative[..., 5, :] = torch.where(
        chi3_symmetric,
        -torsion_angles_tensor[..., 5, :],
        torsion_angles_tensor[..., 5, :],
    )
    return alternative
