import os
import sys

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "minalphafold"))


from a3m import sequence_to_ids
from geometry import backbone_frames, pseudo_beta_positions, torsion_angles


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
