from __future__ import annotations

import torch

try:
    from .residue_constants import (
        chi_angles_atoms,
        chi_angles_mask,
        chi_pi_periodic,
        restype_atom14_is_ambiguous,
        restype_atom14_renaming_matrices,
        restype_rigidgroup_ambiguity_rot,
        restype_rigidgroup_base_atom14_idx,
        restype_rigidgroup_is_ambiguous,
        restype_rigid_group_mask,
        restype_1to3,
        restype_name_to_atom14_names,
        restypes,
    )
except ImportError:  # pragma: no cover - compatibility for direct module imports in tests/scripts.
    from residue_constants import (
        chi_angles_atoms,
        chi_angles_mask,
        chi_pi_periodic,
        restype_atom14_is_ambiguous,
        restype_atom14_renaming_matrices,
        restype_rigidgroup_ambiguity_rot,
        restype_rigidgroup_base_atom14_idx,
        restype_rigidgroup_is_ambiguous,
        restype_rigid_group_mask,
        restype_1to3,
        restype_name_to_atom14_names,
        restypes,
    )


CHI_ATOM_INDICES = torch.full((21, 4, 4), fill_value=-1, dtype=torch.long)
CHI_EXISTS = torch.zeros((21, 4), dtype=torch.float32)
CHI_PI_PERIODIC = torch.tensor(chi_pi_periodic + [[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
RIGID_GROUP_BASE_ATOM14_INDICES = torch.tensor(restype_rigidgroup_base_atom14_idx, dtype=torch.long)
RIGID_GROUP_MASK = torch.tensor(restype_rigid_group_mask, dtype=torch.float32)
RIGID_GROUP_AMBIGUOUS = torch.tensor(restype_rigidgroup_is_ambiguous, dtype=torch.float32)
RIGID_GROUP_ALT_ROTATIONS = torch.tensor(restype_rigidgroup_ambiguity_rot, dtype=torch.float32)
BACKBONE_FRAME_ADAPTATION = torch.diag(torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float32))

_restype_to_atom14 = {
    residue_name: {atom_name: atom_index for atom_index, atom_name in enumerate(atom_names) if atom_name}
    for residue_name, atom_names in restype_name_to_atom14_names.items()
}

for restype_index, restype_letter in enumerate(restypes):
    residue_name = restype_1to3[restype_letter]
    atom_lookup = _restype_to_atom14[residue_name]
    RIGID_GROUP_BASE_ATOM14_INDICES[restype_index, 0] = torch.tensor(
        [atom_lookup["C"], atom_lookup["CA"], atom_lookup["N"]],
        dtype=torch.long,
    )
    RIGID_GROUP_BASE_ATOM14_INDICES[restype_index, 3] = torch.tensor(
        [atom_lookup["CA"], atom_lookup["C"], atom_lookup["O"]],
        dtype=torch.long,
    )
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
        prefix_shape = atom14_tensor.shape[:-1]
        if atom_indices.shape[: len(prefix_shape)] != prefix_shape:
            raise ValueError(
                f"atom_indices shape {tuple(atom_indices.shape)} does not match atom14 prefix {tuple(prefix_shape)}"
            )
        selection_shape = atom_indices.shape[len(prefix_shape):]
        selection_size = int(torch.tensor(selection_shape).prod().item()) if selection_shape else 1
        tensor_flat = atom14_tensor.reshape(-1, 14)
        index_flat = atom_indices.reshape(-1, selection_size)
        gathered = torch.gather(tensor_flat, 1, index_flat)
        result = gathered.reshape(*prefix_shape, *selection_shape)
        return result

    if atom14_tensor.ndim >= 2 and atom14_tensor.shape[-2] == 14:
        prefix_shape = atom14_tensor.shape[:-2]
        if atom_indices.shape[: len(prefix_shape)] != prefix_shape:
            raise ValueError(
                f"atom_indices shape {tuple(atom_indices.shape)} does not match atom14 prefix {tuple(prefix_shape)}"
            )
        selection_shape = atom_indices.shape[len(prefix_shape):]
        selection_size = int(torch.tensor(selection_shape).prod().item()) if selection_shape else 1
        channel_dim = atom14_tensor.shape[-1]
        tensor_flat = atom14_tensor.reshape(-1, 14, channel_dim)
        index_flat = atom_indices.reshape(-1, selection_size)
        gather_index = index_flat.unsqueeze(-1).expand(-1, -1, channel_dim)
        gathered = torch.gather(tensor_flat, 1, gather_index)
        result = gathered.reshape(*prefix_shape, *selection_shape, channel_dim)
        return result

    raise ValueError(f"Could not find atom14 dimension in tensor with shape {atom14_tensor.shape}")


def torsion_sin_cos_from_four_points(
    point_on_xy_plane: torch.Tensor,
    point_on_neg_x_axis: torch.Tensor,
    origin: torch.Tensor,
    point_to_measure: torch.Tensor,
    *,
    flip_sin: bool = False,
    flip_cos: bool = False,
) -> torch.Tensor:
    rotations, translations = rigid_frame_from_three_points(
        point_on_neg_x_axis=point_on_neg_x_axis,
        origin=origin,
        point_on_xy_plane=point_on_xy_plane,
    )
    relative_position = torch.einsum(
        "...ij,...j->...i",
        rotations.transpose(-1, -2),
        point_to_measure - translations,
    )
    sin_cos = safe_normalize(torch.stack([relative_position[..., 2], relative_position[..., 1]], dim=-1))
    if flip_sin:
        sin_cos[..., 0] = -sin_cos[..., 0]
    if flip_cos:
        sin_cos[..., 1] = -sin_cos[..., 1]
    return sin_cos


def rigid_frame_from_three_points(
    point_on_neg_x_axis: torch.Tensor,
    origin: torch.Tensor,
    point_on_xy_plane: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_axis = safe_normalize(origin - point_on_neg_x_axis)
    xy_axis = point_on_xy_plane - origin
    z_axis = safe_normalize(torch.cross(x_axis, xy_axis, dim=-1))
    y_axis = safe_normalize(torch.cross(z_axis, x_axis, dim=-1))
    rotations = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return rotations, origin


def atom14_to_rigid_group_frames(
    atom14_positions: torch.Tensor,
    atom14_mask: torch.Tensor,
    aatype: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_indices = RIGID_GROUP_BASE_ATOM14_INDICES.to(device=aatype.device)
    group_mask = RIGID_GROUP_MASK.to(device=aatype.device, dtype=atom14_positions.dtype)[aatype.long()]
    ambiguity_rotations = RIGID_GROUP_ALT_ROTATIONS.to(device=aatype.device, dtype=atom14_positions.dtype)[aatype.long()]

    residue_base_indices = base_indices[aatype.long()]
    safe_indices = residue_base_indices.clamp(min=0)

    base_positions = []
    base_masks = []
    for atom_offset in range(3):
        base_positions.append(gather_atom14(atom14_positions, safe_indices[..., atom_offset]))
        base_masks.append(
            gather_atom14(atom14_mask, safe_indices[..., atom_offset])
            * (residue_base_indices[..., atom_offset] >= 0).to(atom14_positions.dtype)
        )

    base_atom_mask = base_masks[0] * base_masks[1] * base_masks[2]
    group_exists = group_mask * base_atom_mask

    rotations, translations = rigid_frame_from_three_points(
        base_positions[0],
        base_positions[1],
        base_positions[2],
    )
    backbone_adaptation = BACKBONE_FRAME_ADAPTATION.to(device=atom14_positions.device, dtype=atom14_positions.dtype)
    rotations[..., 0, :, :] = rotations[..., 0, :, :] @ backbone_adaptation

    identity = torch.eye(3, device=atom14_positions.device, dtype=atom14_positions.dtype)
    zero_translation = torch.zeros_like(translations)
    rotations = torch.where(group_exists[..., None, None] > 0, rotations, identity)
    translations = torch.where(group_exists[..., None] > 0, translations, zero_translation)

    alternative_rotations = torch.einsum("...gij,...gjk->...gik", rotations, ambiguity_rotations)
    alternative_translations = translations.clone()
    alternative_rotations = torch.where(group_exists[..., None, None] > 0, alternative_rotations, identity)
    alternative_translations = torch.where(group_exists[..., None] > 0, alternative_translations, zero_translation)

    return rotations, translations, group_exists, alternative_rotations, alternative_translations


def backbone_frames(
    atom14_positions: torch.Tensor,
    atom14_mask: torch.Tensor,
    aatype: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if aatype is None:
        aatype = torch.zeros(atom14_positions.shape[:-2], dtype=torch.long, device=atom14_positions.device)
    rotations, translations, group_exists, _, _ = atom14_to_rigid_group_frames(
        atom14_positions,
        atom14_mask,
        aatype,
    )
    return rotations[..., 0, :, :], translations[..., 0, :], group_exists[..., 0]


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
    o_pos = atom14_positions[..., 3, :]

    n_mask = atom14_mask[..., 0]
    ca_mask = atom14_mask[..., 1]
    c_mask = atom14_mask[..., 2]
    o_mask = atom14_mask[..., 3]

    omega_mask = ca_mask[..., :-1] * c_mask[..., :-1] * n_mask[..., 1:] * ca_mask[..., 1:]
    phi_mask = c_mask[..., :-1] * n_mask[..., 1:] * ca_mask[..., 1:] * c_mask[..., 1:]
    psi_mask = n_mask * ca_mask * c_mask * o_mask

    angles[..., 1:, 0, :] = torsion_sin_cos_from_four_points(
        ca_pos[..., :-1, :],
        c_pos[..., :-1, :],
        n_pos[..., 1:, :],
        ca_pos[..., 1:, :],
    )
    angles[..., 1:, 1, :] = torsion_sin_cos_from_four_points(
        c_pos[..., :-1, :],
        n_pos[..., 1:, :],
        ca_pos[..., 1:, :],
        c_pos[..., 1:, :],
    )
    # Psi is defined from the carbonyl oxygen. In this atom14 implementation the
    # default psi frame is already shifted by pi around the rotation axis, so we
    # mirror both components to match compute_all_atom_coordinates.
    angles[..., 2, :] = torsion_sin_cos_from_four_points(
        n_pos,
        ca_pos,
        c_pos,
        o_pos,
        flip_sin=True,
        flip_cos=True,
    )

    angle_mask[..., 1:, 0] = omega_mask
    angle_mask[..., 1:, 1] = phi_mask
    angle_mask[..., 2] = psi_mask

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

        chi_angles = torsion_sin_cos_from_four_points(
            gathered_positions[0],
            gathered_positions[1],
            gathered_positions[2],
            gathered_positions[3],
        )

        angles[..., 3 + chi_index, :] = chi_angles
        angle_mask[..., 3 + chi_index] = valid_mask

    angles = angles * angle_mask[..., None]
    return angles, angle_mask


def rigid_group_exists(
    atom14_mask: torch.Tensor,
    torsion_mask: torch.Tensor,
) -> torch.Tensor:
    backbone_mask = atom14_mask[..., 0] * atom14_mask[..., 1] * atom14_mask[..., 2]
    return torch.cat(
        [
            backbone_mask.unsqueeze(-1),
            torch.zeros_like(backbone_mask).unsqueeze(-1),
            torch.zeros_like(backbone_mask).unsqueeze(-1),
            torsion_mask[..., 2:],
        ],
        dim=-1,
    )


def alternative_torsion_angles(torsion_angles_tensor: torch.Tensor, aatype: torch.Tensor) -> torch.Tensor:
    alternative = torsion_angles_tensor.clone()
    periodic_mask = CHI_PI_PERIODIC.to(device=aatype.device, dtype=torsion_angles_tensor.dtype)[aatype.long()]
    alternative[..., 3:, :] = torch.where(
        periodic_mask[..., None] > 0,
        -torsion_angles_tensor[..., 3:, :],
        torsion_angles_tensor[..., 3:, :],
    )
    return alternative


def alternative_atom14_ground_truth(
    aatype: torch.Tensor,
    atom14_positions: torch.Tensor,
    atom14_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    renaming_matrices = torch.tensor(
        restype_atom14_renaming_matrices,
        device=aatype.device,
        dtype=atom14_positions.dtype,
    )[aatype.long()]
    ambiguous_mask = torch.tensor(
        restype_atom14_is_ambiguous,
        device=aatype.device,
        dtype=atom14_positions.dtype,
    )[aatype.long()]

    alternative_positions = torch.einsum("...ij,...jc->...ic", renaming_matrices, atom14_positions)
    alternative_mask = torch.einsum("...ij,...j->...i", renaming_matrices, atom14_mask)
    return alternative_positions, alternative_mask, ambiguous_mask
