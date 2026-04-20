import torch

from minalphafold.a3m import sequence_to_ids
from minalphafold.geometry import (
    alternative_torsion_angles,
    atom14_to_rigid_group_frames,
    backbone_frames,
    pseudo_beta_positions,
    rigid_group_exists,
    torsion_angles,
)
from minalphafold.residue_constants import (
    chi_pi_periodic,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group,
    restype_rigid_group_default_frame,
    restype_rigid_group_mask,
)
from minalphafold.structure_module import compute_all_atom_coordinates


def make_atom14_example(sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.zeros(len(sequence), 14, 3, dtype=torch.float32)
    mask = torch.zeros(len(sequence), 14, dtype=torch.float32)

    for index, residue in enumerate(sequence):
        offset = float(index) * 3.8
        positions[index, 0] = torch.tensor([offset, 1.0, 0.0])
        positions[index, 1] = torch.tensor([offset + 1.3, 0.0, 0.0])
        positions[index, 2] = torch.tensor([offset + 2.6, 0.2, 0.0])
        positions[index, 3] = torch.tensor([offset + 3.0, 0.7, 0.0])
        mask[index, :4] = 1.0
        if residue != "G":
            positions[index, 4] = torch.tensor([offset + 1.2, -0.8, 1.1])
            mask[index, 4] = 1.0

    return positions, mask


def masked_rmsd(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_error = torch.sum((predicted - target) ** 2, dim=-1)
    return torch.sqrt((squared_error * mask).sum() / mask.sum().clamp(min=1.0))


def test_pseudo_beta_uses_cb_for_non_gly_and_ca_for_gly():
    positions, mask = make_atom14_example("AG")
    aatype = torch.from_numpy(sequence_to_ids("AG")).long()

    pseudo_beta, pseudo_beta_mask = pseudo_beta_positions(positions, mask, aatype)

    assert torch.allclose(pseudo_beta[0], positions[0, 4])
    assert torch.allclose(pseudo_beta[1], positions[1, 1])
    assert torch.equal(pseudo_beta_mask, torch.tensor([1.0, 1.0]))


def test_backbone_frames_mask_out_residues_missing_backbone_atoms():
    positions, mask = make_atom14_example("AA")
    mask[1, 2] = 0.0

    rotations, translations, frame_mask = backbone_frames(positions, mask)

    assert frame_mask.tolist() == [1.0, 0.0]
    assert torch.allclose(translations[1], torch.zeros(3))
    assert torch.allclose(rotations[1], torch.eye(3), atol=1e-6)


def test_torsion_angles_return_backbone_and_sidechain_masks():
    positions, mask = make_atom14_example("AGA")
    aatype = torch.from_numpy(sequence_to_ids("AGA")).long()

    torsions, torsion_mask = torsion_angles(positions, mask, aatype)

    assert torsions.shape == (3, 7, 2)
    assert torsion_mask.shape == (3, 7)
    assert torsion_mask[1, 0] == 1.0
    assert torsion_mask[1, 1] == 1.0
    assert torsion_mask[1, 2] == 1.0
    assert torsion_mask[:, 3:].sum() == 0.0


def test_psi_uses_carbonyl_oxygen_instead_of_next_residue():
    positions, mask = make_atom14_example("A")
    aatype = torch.from_numpy(sequence_to_ids("A")).long()

    _, torsion_mask = torsion_angles(positions, mask, aatype)

    assert torsion_mask[0, 2] == 1.0


def test_rigid_group_exists_uses_backbone_and_torsion_masks():
    _, mask = make_atom14_example("AA")
    torsion_mask = torch.zeros((2, 7), dtype=torch.float32)
    torsion_mask[1, :3] = 1.0
    torsion_mask[1, 3] = 1.0

    exists = rigid_group_exists(mask, torsion_mask)

    assert exists.shape == (2, 8)
    assert exists[0, 0] == 1.0
    assert exists[0, 1:].sum() == 0.0
    assert torch.equal(exists[1], torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]))


def test_atom14_to_rigid_group_frames_excludes_empty_preomega_and_phi_groups():
    positions, mask = make_atom14_example("AA")
    aatype = torch.from_numpy(sequence_to_ids("AA")).long()

    _, _, group_exists, _, _ = atom14_to_rigid_group_frames(positions, mask, aatype)

    assert torch.equal(group_exists[:, 1], torch.zeros(2))
    assert torch.equal(group_exists[:, 2], torch.zeros(2))
    assert torch.equal(group_exists[:, 0], torch.ones(2))
    assert torch.equal(group_exists[:, 3], torch.ones(2))


def test_atom14_to_rigid_group_frames_keep_masked_alternative_frames_canonical():
    positions, mask = make_atom14_example("AA")
    aatype = torch.from_numpy(sequence_to_ids("AA")).long()

    _, _, group_exists, alt_rotations, alt_translations = atom14_to_rigid_group_frames(positions, mask, aatype)

    masked_groups = group_exists == 0
    # `.item()` is `Number`; `expand`/`zeros` need `int` — cast once so pyright
    # sees a concrete int size.
    num_masked_groups = int(masked_groups.sum().item())
    expected_rotations = torch.eye(3, dtype=alt_rotations.dtype).expand(num_masked_groups, -1, -1)
    expected_translations = torch.zeros(num_masked_groups, 3, dtype=alt_translations.dtype)

    assert torch.allclose(alt_rotations[masked_groups], expected_rotations, atol=1e-6)
    assert torch.allclose(alt_translations[masked_groups], expected_translations, atol=1e-6)


def test_alternative_torsion_angles_use_canonical_pi_periodic_mask():
    aatype = torch.tensor([3, 6, 13, 18], dtype=torch.long)  # ASP, GLU, PHE, TYR
    torsions = torch.zeros((4, 7, 2), dtype=torch.float32)
    torsions[:, 3:, 0] = 0.5
    torsions[:, 3:, 1] = 0.8660254

    alternative = alternative_torsion_angles(torsions, aatype)

    assert torch.allclose(alternative[0, 4], -torsions[0, 4])  # ASP chi2
    assert torch.allclose(alternative[1, 5], -torsions[1, 5])  # GLU chi3
    assert torch.allclose(alternative[2, 4], -torsions[2, 4])  # PHE chi2
    assert torch.allclose(alternative[3, 4], -torsions[3, 4])  # TYR chi2
    assert torch.allclose(alternative[1, 4], torsions[1, 4])   # GLU chi2 unchanged


def test_chi_pi_periodic_matches_canonical_af2_values():
    assert chi_pi_periodic[2] == [0.0, 0.0, 0.0, 0.0]   # ASN
    assert chi_pi_periodic[3] == [0.0, 1.0, 0.0, 0.0]   # ASP
    assert chi_pi_periodic[5] == [0.0, 0.0, 0.0, 0.0]   # GLN
    assert chi_pi_periodic[6] == [0.0, 0.0, 1.0, 0.0]   # GLU
    assert chi_pi_periodic[13] == [0.0, 1.0, 0.0, 0.0]  # PHE
    assert chi_pi_periodic[18] == [0.0, 1.0, 0.0, 0.0]  # TYR


def test_proline_cd_literature_position_matches_canonical_value():
    proline_cd = torch.tensor(restype_atom14_rigid_group_positions[14, 6], dtype=torch.float32)
    assert torch.allclose(proline_cd, torch.tensor([0.477, 1.424, 0.0]), atol=1e-6)


def test_unknown_rigid_group_defaults_match_canonical_masking():
    unknown_mask = torch.tensor(restype_rigid_group_mask[20], dtype=torch.float32)
    assert torch.equal(unknown_mask, torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(
        torch.tensor(restype_rigid_group_default_frame[20, 0], dtype=torch.float32),
        torch.zeros((4, 4), dtype=torch.float32),
    )
    assert torch.allclose(
        torch.tensor(restype_rigid_group_default_frame[20, 3], dtype=torch.float32),
        torch.zeros((4, 4), dtype=torch.float32),
    )


def test_torsion_round_trip_matches_atom_construction_path():
    sequence = "RKYD"
    aatype = torch.from_numpy(sequence_to_ids(sequence)).long().unsqueeze(0)

    translations = torch.tensor(
        [[[0.0, 0.0, 0.0], [3.8, 0.5, 0.2], [7.6, -0.4, 0.1], [11.4, 0.2, -0.3]]],
        dtype=torch.float32,
    )
    rotations = torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3).repeat(1, len(sequence), 1, 1)
    torsions = torch.tensor(
        [
            [
                [[0.0, 1.0], [0.0, 1.0], [0.6, 0.8], [0.5, 0.8660254], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [0.8, 0.6], [0.70710677, 0.70710677], [0.5, 0.8660254], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [-0.5, 0.8660254], [0.8660254, 0.5], [0.70710677, 0.70710677], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [0.25881904, 0.9659258], [0.5, 0.8660254], [0.25881904, 0.9659258], [0.70710677, 0.70710677], [0.0, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )

    default_frames = torch.tensor(restype_rigid_group_default_frame, dtype=torch.float32)
    lit_positions = torch.tensor(restype_atom14_rigid_group_positions, dtype=torch.float32)
    atom_frame_idx_table = torch.tensor(restype_atom14_to_rigid_group, dtype=torch.long)
    atom_mask_table = torch.tensor(restype_atom14_mask, dtype=torch.float32)

    _, _, atom_positions, atom_mask = compute_all_atom_coordinates(
        translations,
        rotations,
        torsions,
        aatype,
        default_frames,
        lit_positions,
        atom_frame_idx_table,
        atom_mask_table,
    )

    backbone_rotations, backbone_translations, _, _, _ = atom14_to_rigid_group_frames(
        atom_positions[0],
        atom_mask[0],
        aatype[0],
    )
    recovered_torsions, _ = torsion_angles(atom_positions[0], atom_mask[0], aatype[0])

    _, _, reconstructed_positions, _ = compute_all_atom_coordinates(
        backbone_translations[:, 0].unsqueeze(0),
        backbone_rotations[:, 0].unsqueeze(0),
        recovered_torsions.unsqueeze(0),
        aatype,
        default_frames,
        lit_positions,
        atom_frame_idx_table,
        atom_mask_table,
    )

    all_atom_rmsd = masked_rmsd(reconstructed_positions[0], atom_positions[0], atom_mask[0])
    oxygen_rmsd = masked_rmsd(reconstructed_positions[0, :, 3:4], atom_positions[0, :, 3:4], atom_mask[0, :, 3:4])

    assert all_atom_rmsd.item() < 1e-3
    assert oxygen_rmsd.item() < 1e-3
