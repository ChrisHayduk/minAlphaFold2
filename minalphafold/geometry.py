"""Rigid frame construction, torsion angles, and atom14 helpers.

This module builds the **ground-truth side** of the Structure Module's
supervision. From a chain's atom14 coordinates it computes:

* the 8 per-residue rigid-group frames ``T_i^{true,f}`` (Algorithm 21 /
  Table 2) — backbone + ω/φ/ψ + χ1..χ4;
* the 7 per-residue torsion angles ``α_i^f`` as sin/cos pairs (supplement
  1.8.4 / Algorithm 24 in reverse) — ω, φ, ψ, χ1..χ4;
* pseudo-β positions (Cβ except for glycine, where Cα) used for the
  distogram (1.9.8), recycling (Algorithm 32), and template distogram
  (1.7.1);
* the 180°-rotation symmetry handling from Algorithm 26 / supplement 1.8.5
  — "alt truth" torsion angles and atom positions for ASP/GLU/PHE/TYR.

Everything here is called from ``data.build_supervision``; nothing in this
file is on the prediction path. Predictions go through
``structure_module.compute_all_atom_coordinates`` (Algorithm 24), which
uses parametric literature frames instead of Algorithm 21.

Frame convention subtleties
---------------------------

Supplement 1.8.1 specifies the backbone frame via Algorithm 21 with
``(x1, x2, x3) = (N, Cα, C)``, which gives ``e1 = (C − Cα)/||C − Cα||`` —
i.e. the +x axis points from Cα toward C.

``rigid_frame_from_three_points`` below takes ``(point_on_neg_x_axis,
origin, point_on_xy_plane)`` and builds ``x_axis = (origin −
point_on_neg_x_axis)/||·||``. The backbone atoms are fed in as
``(C, Cα, N)``, so our x-axis points from C toward Cα — the **opposite** of
the paper's convention. ``BACKBONE_FRAME_ADAPTATION = diag(−1, 1, −1)`` is
the rotation that reconciles the two (right-multiplication into the
backbone rotation at the end of ``atom14_to_rigid_group_frames``). It flips
x and z while preserving handedness (``det = +1``), which is equivalent to
a 180° rotation about the y-axis.

For non-backbone groups (ω, φ, ψ, χ1..χ4) the literature constants in
``residue_constants.restype_rigid_group_default_frame`` were derived in
the same convention the wrapper below produces, so no adaptation is
applied and the ground-truth frames match the frames the Structure Module
predicts.
"""

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


# Chi-angle lookup tables, indexed by (restype_id, chi_index):
#   - CHI_ATOM_INDICES[r, k, :] gives the 4 atom14 slots that define χ_{k+1}
#     for residue type r (N-CA-CB-CG for χ1, etc.); -1 where undefined.
#   - CHI_EXISTS[r, k] is 1.0 iff residue type r has χ_{k+1} defined.
# Filled in the loop below by cross-referencing residue_constants tables.
CHI_ATOM_INDICES = torch.full((21, 4, 4), fill_value=-1, dtype=torch.long)
CHI_EXISTS = torch.zeros((21, 4), dtype=torch.float32)

# Per-residue flag: which χ angles are π-periodic (torsion symmetry flipping
# the side chain gives an indistinguishable structure — supplement 1.8.5 /
# Table 3 — namely ASP χ2, GLU χ3, PHE χ2, TYR χ2).  Used by
# ``alternative_torsion_angles`` to negate sin/cos for those χs.
# The trailing row of zeros is the "UNK" restype (index 20) which has no χs.
CHI_PI_PERIODIC = torch.tensor(chi_pi_periodic + [[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

# Tables indexed by aatype → per-rigid-group metadata (shape (21, 8, ...)):
#   - RIGID_GROUP_BASE_ATOM14_INDICES: 3 atom14 slots per group feeding
#     ``rigid_frame_from_three_points`` (-1 where the group is undefined).
#   - RIGID_GROUP_MASK: 1.0 iff the group exists for this residue type.
#   - RIGID_GROUP_AMBIGUOUS / RIGID_GROUP_ALT_ROTATIONS: which groups have
#     180°-rotation symmetry (Table 3) and the local rotation that swaps
#     them, used to build the "alt truth" frames for Algorithm 26.
RIGID_GROUP_BASE_ATOM14_INDICES = torch.tensor(restype_rigidgroup_base_atom14_idx, dtype=torch.long)
RIGID_GROUP_MASK = torch.tensor(restype_rigid_group_mask, dtype=torch.float32)
RIGID_GROUP_AMBIGUOUS = torch.tensor(restype_rigidgroup_is_ambiguous, dtype=torch.float32)
RIGID_GROUP_ALT_ROTATIONS = torch.tensor(restype_rigidgroup_ambiguity_rot, dtype=torch.float32)

# Rotation applied to the ground-truth backbone frame to match the paper's
# convention (+x from Cα→C). See the module docstring for the derivation.
# det(diag(-1, 1, -1)) = +1 so this is a proper rotation (180° about y).
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
    """L2-normalise along the last axis, with an ε floor on the norm."""
    return vectors / torch.sqrt(torch.sum(vectors ** 2, dim=-1, keepdim=True) + eps)


def gather_atom14(atom14_tensor: torch.Tensor, atom_indices: torch.Tensor) -> torch.Tensor:
    """Index into an atom14 tensor along its atom14 axis.

    Handles both representations we use:

    * ``(..., N_res, 14)`` — e.g. atom14 masks. Last axis is atom14.
    * ``(..., N_res, 14, C)`` — e.g. atom14 positions. Second-to-last is atom14.

    ``atom_indices`` has a matching prefix shape and any trailing selection
    shape, so callers can gather either one index per residue
    (``atom_indices`` shape ``(..., N_res)``) or many
    (e.g. ``(..., N_res, 8)`` to gather one atom per rigid group).
    Implemented via ``torch.gather`` after flattening the prefix dims.
    """
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
    """Measure a torsion angle (as (sin, cos)) from its four defining atoms.

    A proper dihedral ``p1-p2-p3-p4`` is the angle between the planes
    ``(p1, p2, p3)`` and ``(p2, p3, p4)``, measured around the ``p2-p3``
    axis. We compute it by:

    1. Building a local frame from ``(p1, p2, p3)`` via
       ``rigid_frame_from_three_points`` — ``p3`` becomes the origin, the
       x-axis points along the rotation axis, and ``p1`` sits on the
       negative-x side (lying in the xy-plane by construction).
    2. Expressing ``p4`` in that frame — its x-component is along the axis
       and irrelevant; its (y, z) components give the rotation angle.
    3. Returning ``(sin, cos) = (z, y) / ||(y, z)||`` (sin from z because
       the rotation is a right-hand rule about +x: a point initially on +y
       rotates toward +z at +90°).

    ``flip_sin`` / ``flip_cos`` let the caller negate either component —
    used for the ψ angle convention (see ``torsion_angles``).
    """
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
    """Gram–Schmidt frame construction (Algorithm 21).

    Returns ``(R, t)`` where ``t = origin`` and ``R`` is a 3×3 rotation
    matrix whose columns are the local-frame axes expressed in global
    coordinates.

    Axis conventions for the three inputs:

    * ``origin`` becomes the frame centre (translation component).
    * ``point_on_neg_x_axis`` sits at **negative** x in the built frame:
      the positive x-axis points *from this atom toward the origin*
      (``x_axis = (origin − point_on_neg_x_axis) / ||·||``). **This is the
      opposite direction from Algorithm 21's ``e1 = (x3 − x2)/||x3 − x2||``**
      — when we feed backbone atoms as ``(C, Cα, N)`` below, the backbone
      rotation gets reconciled with the paper via
      ``BACKBONE_FRAME_ADAPTATION``.
    * ``point_on_xy_plane`` lies in the xy-plane by construction. Its
      residual (after subtracting the x-axis component) defines ±y.

    The y/z axes are completed by two cross products so R is guaranteed
    orthonormal and right-handed (``det R = +1``).
    """
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
    """Build all 8 ground-truth rigid-group frames per residue (supplement 1.8.1 + 1.8.5).

    Returns ``(R, t, group_exists, R_alt, t_alt)``:

    * ``R``: ``(..., N_res, 8, 3, 3)`` rotation matrices.
    * ``t``: ``(..., N_res, 8, 3)`` translation vectors.
    * ``group_exists``: ``(..., N_res, 8)`` — 1 iff the group is defined for
      the residue type and all three base atoms are present in the ground
      truth.
    * ``R_alt``, ``t_alt``: the "alt truth" frames from supplement 1.8.5 /
      Algorithm 26 for 180°-symmetric groups (ASP/GLU/PHE/TYR, Table 3).
      Non-ambiguous groups have ``R_alt == R``, ``t_alt == t``.

    Non-existing groups (missing atoms or restypes without the group, e.g.
    ALA has no χ frames) are replaced with the identity frame so
    downstream tensor ops don't see NaNs; ``group_exists`` tells the loss
    to mask them out.
    """
    base_indices = RIGID_GROUP_BASE_ATOM14_INDICES.to(device=aatype.device)
    group_mask = RIGID_GROUP_MASK.to(device=aatype.device, dtype=atom14_positions.dtype)[aatype.long()]
    ambiguity_rotations = RIGID_GROUP_ALT_ROTATIONS.to(device=aatype.device, dtype=atom14_positions.dtype)[aatype.long()]

    residue_base_indices = base_indices[aatype.long()]
    # Negative indices mean "group undefined for this restype"; clamp to 0
    # before gather so we get valid indices, then zero the mask below.
    safe_indices = residue_base_indices.clamp(min=0)

    base_positions = []
    base_masks = []
    for atom_offset in range(3):
        base_positions.append(gather_atom14(atom14_positions, safe_indices[..., atom_offset]))
        base_masks.append(
            gather_atom14(atom14_mask, safe_indices[..., atom_offset])
            * (residue_base_indices[..., atom_offset] >= 0).to(atom14_positions.dtype)
        )

    # A group "exists" only when all three base atoms are present AND the
    # restype defines the group (``RIGID_GROUP_MASK``).
    base_atom_mask = base_masks[0] * base_masks[1] * base_masks[2]
    group_exists = group_mask * base_atom_mask

    rotations, translations = rigid_frame_from_three_points(
        base_positions[0],
        base_positions[1],
        base_positions[2],
    )
    # Reconcile the backbone rotation with the paper's +x = (Cα → C)
    # convention — see the module docstring for the derivation.
    backbone_adaptation = BACKBONE_FRAME_ADAPTATION.to(device=atom14_positions.device, dtype=atom14_positions.dtype)
    rotations[..., 0, :, :] = rotations[..., 0, :, :] @ backbone_adaptation

    identity = torch.eye(3, device=atom14_positions.device, dtype=atom14_positions.dtype)
    zero_translation = torch.zeros_like(translations)
    rotations = torch.where(group_exists[..., None, None] > 0, rotations, identity)
    translations = torch.where(group_exists[..., None] > 0, translations, zero_translation)

    # Supplement 1.8.5 / Table 3: for 180°-symmetric groups (ASP/GLU/PHE/
    # TYR), the alt truth is the same atoms with the group rotated 180°
    # about its own local x-axis. In frame space that's a right-composition
    # by ``ambiguity_rotations``, which is the identity for non-ambiguous
    # groups (so their alt frames equal their true frames).
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
    """Extract just the backbone (rigid group 0) frames from atom14 positions.

    Thin wrapper over ``atom14_to_rigid_group_frames``; produces the
    ``true_rotations`` / ``true_translations`` used by the backbone FAPE
    loss via ``data.build_supervision``. ``aatype`` defaults to all-ALA
    because group 0 is defined identically for every restype.
    """
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
    """Pseudo-β atoms: Cβ for all residues except GLY, where Cα is used.

    Returns ``(positions, mask)`` of shapes ``(..., N_res, 3)`` and
    ``(..., N_res)``. Shared convention for the distogram target (1.9.8),
    template distogram (1.7.1 / Table 1), and recycling distance feature
    (Algorithm 32 line 1). atom14 slot 1 is Cα, slot 4 is Cβ, and
    ``aatype == 7`` is glycine (see ``a3m.RESTYPES``).
    """
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
    """Compute the 7 per-residue torsion angles (as sin/cos pairs).

    Returns ``(angles, mask)`` with shapes ``(..., N_res, 7, 2)`` and
    ``(..., N_res, 7)``. The 7 angles are ``[ω, φ, ψ, χ1, χ2, χ3, χ4]``
    (indices 0..6) matching the ordering the Structure Module predicts
    in Algorithm 20 line 11 and consumes in Algorithm 24.

    * **ω** (i ≥ 1): Cα_{i-1}–C_{i-1}–N_i–Cα_i.
    * **φ** (i ≥ 1): C_{i-1}–N_i–Cα_i–C_i.
    * **ψ** (all i): N_i–Cα_i–C_i–O_i. Defined by the carbonyl O rather
      than the next residue's N; the (α̂ = α̃/||α̃||) sign correction below
      compensates for the 180° offset this introduces vs Algorithm 24's
      ψ frame.
    * **χ1..χ4**: side-chain torsions from ``chi_angles_atoms``
      (``CHI_ATOM_INDICES``); undefined χs stay at (0, 0) with mask 0.

    Residue 0 has ω and φ undefined (no predecessor) — those positions stay
    zero with mask 0.
    """
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
    # Psi is measured from (N, Cα, C, O), whereas ``compute_all_atom_coordinates``
    # in the Structure Module builds the ψ rigid group from (Cα, C, O) — its
    # literature frame is already rotated 180° about the ψ axis relative to the
    # frame this function builds. Negating both sin and cos is a 180° rotation
    # about the axis, reconciling the two conventions so the torsion loss sees
    # predicted and ground-truth angles in the same frame.
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
    """Derive an 8-slot rigid-group existence mask from atom/torsion masks.

    The 8 rigid groups per residue are ``[backbone, ω, φ, ψ, χ1, χ2, χ3,
    χ4]``. Supplement 1.8.4 notes that ω and φ have no side-chain atoms
    (the corresponding backbone atoms are supervised by the backbone group
    0), so this helper reports them as non-existent. ψ + χ_k reuse the
    torsion mask computed by ``torsion_angles`` — a rigid group is "present"
    iff the torsion angle that drives its rotation is defined.

    Returns a tensor of shape ``(..., N_res, 8)`` with:
    ``[backbone_mask, 0, 0, psi_mask, chi1_mask, chi2_mask, chi3_mask,
    chi4_mask]``.
    """
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
    """Compute the π-periodic "alt truth" torsion angles (supplement 1.8.5).

    For ASP χ2, GLU χ3, PHE χ2, and TYR χ2 (Table 3), the side chain is
    180°-rotation symmetric about that torsion axis, so the torsion
    measured from atom naming can be off by π from the physically
    equivalent angle. Per Algorithm 27 line 3, the torsion loss takes the
    minimum over the two labels; this helper returns the alternative.

    Negating (sin, cos) rotates the unit-circle point by 180°, which is the
    π-shift we need. For non-periodic χs (and for ω, φ, ψ which are never
    π-periodic), the alt truth equals the true value.
    """
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
    """Compute the "alt truth" atom14 positions for Algorithm 26.

    For 180°-symmetric side-chain groups (Table 3: ASP/GLU/PHE/TYR), the
    atom naming is ambiguous — e.g. ASP Oδ1 and Oδ2 are chemically
    indistinguishable. Algorithm 26 picks the naming that best matches the
    prediction by comparing ``d^true`` and ``d^{alt truth}`` distances; this
    function produces ``d^{alt truth}`` positions by applying a pre-computed
    per-restype permutation matrix to the atom14 tensor.

    Returns ``(positions_alt, mask_alt, is_ambiguous_per_atom)`` shapes
    ``(..., N_res, 14, 3)``, ``(..., N_res, 14)``, ``(..., N_res, 14)``.
    ``is_ambiguous_per_atom`` flags which atom14 slots are affected by the
    swap (e.g. ASP Oδ1/Oδ2, ALA has no flagged atoms).
    """
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
