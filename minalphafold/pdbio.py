from pathlib import Path
from typing import Any

import torch

try:
    from .residue_constants import restype_1to3, restype_name_to_atom14_names, restypes
except ImportError:  # pragma: no cover - compatibility for direct module imports in tests/scripts.
    from residue_constants import restype_1to3, restype_name_to_atom14_names, restypes


def _to_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype)


def _residue_name_from_aatype(aatype_index: int) -> str:
    if 0 <= aatype_index < len(restypes):
        return restype_1to3[restypes[aatype_index]]
    return "UNK"


def _element_from_atom_name(atom_name: str) -> str:
    for character in atom_name:
        if character.isalpha():
            return character.upper()
    return ""


def atom14_to_pdb_string(
    aatype: Any,
    atom14_positions: Any,
    atom14_mask: Any,
    *,
    residue_index: Any | None = None,
    chain_id: str = "A",
    b_factors: Any | None = None,
    occupancies: Any | None = None,
    serial_start: int = 1,
) -> str:
    """Serialize one atom14 structure to a simple PDB string."""

    if len(chain_id) != 1:
        raise ValueError(f"chain_id must be a single character, got {chain_id!r}")

    aatype_tensor = _to_tensor(aatype, dtype=torch.long).reshape(-1)
    positions_tensor = _to_tensor(atom14_positions, dtype=torch.float32)
    mask_tensor = _to_tensor(atom14_mask, dtype=torch.float32)

    if positions_tensor.ndim != 3 or positions_tensor.shape[-2:] != (14, 3):
        raise ValueError(
            "atom14_positions must have shape (num_residues, 14, 3), "
            f"got {tuple(positions_tensor.shape)}"
        )
    if mask_tensor.shape != positions_tensor.shape[:-1]:
        raise ValueError(
            "atom14_mask must have shape (num_residues, 14), "
            f"got {tuple(mask_tensor.shape)}"
        )
    if aatype_tensor.shape[0] != positions_tensor.shape[0]:
        raise ValueError(
            "aatype length must match the number of residues, "
            f"got {aatype_tensor.shape[0]} and {positions_tensor.shape[0]}"
        )

    num_residues = aatype_tensor.shape[0]
    if residue_index is None:
        residue_index_tensor = torch.arange(num_residues, dtype=torch.long)
    else:
        residue_index_tensor = _to_tensor(residue_index, dtype=torch.long).reshape(-1)
    if residue_index_tensor.shape[0] != num_residues:
        raise ValueError(
            "residue_index length must match the number of residues, "
            f"got {residue_index_tensor.shape[0]} and {num_residues}"
        )

    if b_factors is None:
        b_factor_tensor = torch.zeros(num_residues, dtype=torch.float32)
    else:
        b_factor_tensor = _to_tensor(b_factors, dtype=torch.float32).reshape(-1)
    if b_factor_tensor.shape[0] != num_residues:
        raise ValueError(
            "b_factors length must match the number of residues, "
            f"got {b_factor_tensor.shape[0]} and {num_residues}"
        )

    if occupancies is None:
        occupancy_tensor = torch.ones(num_residues, dtype=torch.float32)
    else:
        occupancy_tensor = _to_tensor(occupancies, dtype=torch.float32).reshape(-1)
    if occupancy_tensor.shape[0] != num_residues:
        raise ValueError(
            "occupancies length must match the number of residues, "
            f"got {occupancy_tensor.shape[0]} and {num_residues}"
        )

    lines: list[str] = []
    atom_serial = serial_start
    last_residue_name = "UNK"
    last_residue_number = 0

    for residue_offset in range(num_residues):
        residue_mask = mask_tensor[residue_offset]
        if float(residue_mask.max().item()) < 0.5:
            continue

        residue_name = _residue_name_from_aatype(int(aatype_tensor[residue_offset].item()))
        residue_number = int(residue_index_tensor[residue_offset].item()) + 1
        atom_names = restype_name_to_atom14_names.get(residue_name, [""] * 14)
        occupancy = float(occupancy_tensor[residue_offset].item())
        b_factor = float(b_factor_tensor[residue_offset].item())

        for atom_index, atom_name in enumerate(atom_names):
            if not atom_name or float(residue_mask[atom_index].item()) < 0.5:
                continue

            x, y, z = positions_tensor[residue_offset, atom_index].tolist()
            element = _element_from_atom_name(atom_name)
            lines.append(
                f"ATOM  {atom_serial:5d} {atom_name:>4s} {residue_name:>3s} {chain_id:1s}"
                f"{residue_number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"{occupancy:6.2f}{b_factor:6.2f}          {element:>2s}"
            )
            atom_serial += 1

        last_residue_name = residue_name
        last_residue_number = residue_number

    if lines:
        lines.append(
            f"TER   {atom_serial:5d}      {last_residue_name:>3s} {chain_id:1s}{last_residue_number:4d}"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_atom14_pdb(
    path: str | Path,
    aatype: Any,
    atom14_positions: Any,
    atom14_mask: Any,
    *,
    residue_index: Any | None = None,
    chain_id: str = "A",
    b_factors: Any | None = None,
    occupancies: Any | None = None,
    serial_start: int = 1,
) -> Path:
    output_path = Path(path)
    pdb_text = atom14_to_pdb_string(
        aatype,
        atom14_positions,
        atom14_mask,
        residue_index=residue_index,
        chain_id=chain_id,
        b_factors=b_factors,
        occupancies=occupancies,
        serial_start=serial_start,
    )
    output_path.write_text(pdb_text)
    return output_path


def write_model_output_pdb(
    path: str | Path,
    model_output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    example_index: int = 0,
    chain_id: str = "A",
    b_factors: Any | None = None,
    occupancies: Any | None = None,
) -> Path:
    if b_factors is None and "plddt_logits" in model_output:
        plddt_logits = model_output["plddt_logits"][example_index]
        num_bins = plddt_logits.shape[-1]
        bin_centers = (torch.arange(num_bins, dtype=plddt_logits.dtype, device=plddt_logits.device) + 0.5)
        bin_centers = bin_centers * (100.0 / num_bins)
        b_factors = torch.softmax(plddt_logits, dim=-1) @ bin_centers

    return write_atom14_pdb(
        path,
        batch["aatype"][example_index],
        model_output["atom14_coords"][example_index],
        model_output["atom14_mask"][example_index],
        residue_index=batch["residue_index"][example_index],
        chain_id=chain_id,
        b_factors=b_factors,
        occupancies=occupancies,
    )
