from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minalphafold"))


from a3m import sequence_to_ids
from pdbio import atom14_to_pdb_string, write_model_output_pdb
from residue_constants import restype_1to3, restype_name_to_atom14_names, restypes


def make_full_atom14_example(sequence: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    aatype = torch.from_numpy(sequence_to_ids(sequence)).long()
    positions = torch.zeros(len(sequence), 14, 3, dtype=torch.float32)
    mask = torch.zeros(len(sequence), 14, dtype=torch.float32)

    for residue_index, aatype_index in enumerate(aatype.tolist()):
        residue_name = restype_1to3[restypes[aatype_index]]
        for atom_index, atom_name in enumerate(restype_name_to_atom14_names[residue_name]):
            if not atom_name:
                continue
            positions[residue_index, atom_index] = torch.tensor(
                [10.0 * residue_index + atom_index, atom_index + 0.25, -atom_index - 0.5],
                dtype=torch.float32,
            )
            mask[residue_index, atom_index] = 1.0

    return aatype, positions, mask


def test_atom14_to_pdb_string_writes_backbone_and_sidechains():
    aatype, positions, mask = make_full_atom14_example("GD")
    residue_index = torch.tensor([4, 7], dtype=torch.long)
    b_factors = torch.tensor([11.5, 27.25], dtype=torch.float32)
    occupancies = torch.tensor([1.0, 0.5], dtype=torch.float32)

    pdb_text = atom14_to_pdb_string(
        aatype,
        positions,
        mask,
        residue_index=residue_index,
        chain_id="B",
        b_factors=b_factors,
        occupancies=occupancies,
    )

    atom_lines = [line for line in pdb_text.splitlines() if line.startswith("ATOM")]
    assert len(atom_lines) == int(mask.sum().item())
    assert any(" OD1 ASP B   8" in line for line in atom_lines)
    assert any(" OD2 ASP B   8" in line for line in atom_lines)
    assert any("  CA GLY B   5" in line for line in atom_lines)
    assert atom_lines[0][21] == "B"
    assert float(atom_lines[0][54:60]) == 1.0
    assert float(atom_lines[-1][54:60]) == 0.5
    assert float(atom_lines[0][60:66]) == 11.5
    assert float(atom_lines[-1][60:66]) == 27.25
    assert pdb_text.splitlines()[-2].startswith("TER")
    assert pdb_text.splitlines()[-1] == "END"


def test_write_model_output_pdb_uses_plddt_b_factors(tmp_path):
    aatype, positions, mask = make_full_atom14_example("AC")
    output_path = tmp_path / "prediction.pdb"

    model_output = {
        "atom14_coords": positions.unsqueeze(0),
        "atom14_mask": mask.unsqueeze(0),
        "plddt_logits": torch.tensor([[[0.0, 0.0, 0.0, 8.0], [0.0, 0.0, 8.0, 0.0]]], dtype=torch.float32),
    }
    batch = {
        "aatype": aatype.unsqueeze(0),
        "residue_index": torch.tensor([[0, 1]], dtype=torch.long),
    }

    write_model_output_pdb(output_path, model_output, batch, chain_id="A")

    lines = [line for line in output_path.read_text().splitlines() if line.startswith("ATOM")]
    first_b_factor = float(lines[0][60:66])
    second_b_factor = float(lines[-1][60:66])

    assert abs(first_b_factor - 87.5) < 1.0
    assert abs(second_b_factor - 62.5) < 1.0
