"""mmCIF parser → per-chain atom14 structures for ground-truth supervision.

mmCIF is the modern replacement for legacy PDB files that the RCSB / PDBe
distribute for every deposited structure. It is a flat text format built
out of two record types:

* **Scalars**: ``_category.tag value`` lines, e.g.
  ``_refine.ls_d_res_high 2.10``.
* **Loops**: ``loop_`` + one or more column names + a free-form value table,
  used for tabular data like atom coordinates (``_atom_site.*``) and
  residue sequences.

The supplement's data pipeline is described in Section 1.2.1 ("Parsing"):
*"for mmCIF this is the sequence, atom coordinates, release date, name,
and resolution. We also resolve alternative locations for atoms/residues,
taking the one with the largest occupancy"*. This module implements that
step, extracting a single chain from one mmCIF file into a
``ChainAtoms`` record carrying ``atom14_positions``, ``atom14_mask``,
``aatype``, ``residue_index`` (contiguous 0..N-1 per supplement 1.2.9,
*not* author numbering), and ``resolution`` (for the pLDDT / exp-resolved
loss resolution filters in supplement 1.9.6 / 1.9.10).

This is a deliberately minimal parser — it assumes single-model
structures, does not validate against the mmCIF schema, and is not fast.
For a pedagogical run on a handful of chains that's fine; anything larger
should switch to BioPython or gemmi.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .a3m import sequence_to_ids
from .residue_constants import restype_name_to_atom14_names


THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

ATOM14_INDEX = {
    residue_name: {atom_name: atom_idx for atom_idx, atom_name in enumerate(atom_names) if atom_name}
    for residue_name, atom_names in restype_name_to_atom14_names.items()
}


def _clean_sequence(raw_sequence: str) -> str:
    cleaned = "".join(char for char in raw_sequence if char.isalpha())
    if not cleaned:
        raise ValueError("Could not parse a polymer sequence from mmCIF text.")
    return cleaned.upper()


def _tokenize_mmcif(text: str) -> List[str]:
    """Tokenise raw mmCIF text into a flat list of string tokens.

    Handles three conventions:

    * Blank lines and ``#`` comment lines are dropped.
    * Multi-line semicolon-delimited values (``; ...\\n...\\n;``) are read
      back as one token.
    * Everything else is split by ``shlex`` so quoted strings survive intact.
    """
    tokens: List[str] = []
    lines = text.splitlines()
    line_index = 0

    while line_index < len(lines):
        line = lines[line_index]
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            line_index += 1
            continue

        if line.startswith(";"):
            block_lines: List[str] = []
            line_index += 1
            while line_index < len(lines) and not lines[line_index].startswith(";"):
                block_lines.append(lines[line_index].rstrip("\n"))
                line_index += 1
            tokens.append("\n".join(block_lines))
            line_index += 1
            continue

        tokens.extend(shlex.split(line, posix=True))
        line_index += 1

    return tokens


def _parse_mmcif(text: str) -> Tuple[Dict[str, str], List[Tuple[List[str], List[List[str]]]]]:
    """Split tokenised mmCIF into ``(scalars, loops)``.

    ``scalars`` is a mapping from ``_category.tag`` to its value. ``loops`` is
    a list of ``(column_names, rows)`` pairs — one entry per ``loop_`` block.
    """
    tokens = _tokenize_mmcif(text)
    scalars: Dict[str, str] = {}
    loops: List[Tuple[List[str], List[List[str]]]] = []

    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token == "loop_":
            index += 1
            columns: List[str] = []
            while index < len(tokens) and tokens[index].startswith("_"):
                columns.append(tokens[index])
                index += 1

            if not columns:
                raise ValueError("Encountered loop_ without any columns.")

            flat_values: List[str] = []
            while index < len(tokens):
                if len(flat_values) % len(columns) == 0:
                    candidate = tokens[index]
                    if candidate == "loop_" or candidate.startswith("_") or candidate.startswith("data_"):
                        break
                flat_values.append(tokens[index])
                index += 1

            if len(flat_values) % len(columns) != 0:
                raise ValueError("Loop values do not align with loop columns.")

            rows = [
                flat_values[row_start: row_start + len(columns)]
                for row_start in range(0, len(flat_values), len(columns))
            ]
            loops.append((columns, rows))
            continue

        if token.startswith("_"):
            if index + 1 >= len(tokens):
                raise ValueError(f"Missing value for mmCIF tag {token}")
            scalars[token] = tokens[index + 1]
            index += 2
            continue

        index += 1

    return scalars, loops


def _entity_sequences(
    scalars: Dict[str, str],
    loops: Iterable[Tuple[List[str], List[List[str]]]],
) -> Dict[str, str]:
    entity_sequences: Dict[str, str] = {}

    entity_id = scalars.get("_entity_poly.entity_id")
    seq = scalars.get("_entity_poly.pdbx_seq_one_letter_code_can")
    if entity_id is not None and seq is not None:
        entity_sequences[str(entity_id)] = _clean_sequence(seq)

    for columns, rows in loops:
        if "_entity_poly.entity_id" not in columns or "_entity_poly.pdbx_seq_one_letter_code_can" not in columns:
            continue
        entity_col = columns.index("_entity_poly.entity_id")
        seq_col = columns.index("_entity_poly.pdbx_seq_one_letter_code_can")
        for row in rows:
            entity_sequences[str(row[entity_col])] = _clean_sequence(row[seq_col])

    return entity_sequences


def _one_letter_from_resname(resname: str) -> str:
    return THREE_TO_ONE.get(resname.upper(), "X")


def _select_atom_rows(
    columns: List[str],
    rows: List[List[str]],
    chain_id: str,
) -> Tuple[List[List[str]], str]:
    """Filter ``_atom_site`` rows to one chain, preferring author chain IDs.

    mmCIF carries two chain-ID columns: ``auth_asym_id`` (the author-assigned
    letter, used in PDB downloads and in the literature) and
    ``label_asym_id`` (the internal mmCIF ID, which can differ for
    modified chains). We match against the author IDs first since that's
    what users normally supply (e.g. ``"A"`` in ``1abc_A``), and fall back
    to label IDs so we can still parse chains where the two diverge.
    """
    auth_chain_col = columns.index("_atom_site.auth_asym_id")
    label_chain_col = columns.index("_atom_site.label_asym_id")

    auth_rows = [row for row in rows if row[auth_chain_col] == chain_id]
    if auth_rows:
        return auth_rows, "auth"

    label_rows = [row for row in rows if row[label_chain_col] == chain_id]
    if label_rows:
        return label_rows, "label"

    raise KeyError(f"Chain '{chain_id}' was not found in the mmCIF atom_site loop.")


def _best_atom_rows(rows: List[List[str]], columns: List[str]) -> Dict[Tuple[int, str], Tuple[np.ndarray, str]]:
    """Collapse alternate-location (altloc) rows into one atom per (residue, atom name).

    Many PDB structures record multiple positions for partially-disordered
    side chains ("A" vs "B" altlocs). Supplement 1.2.1 specifies *"taking
    the one with the largest occupancy"* — we implement that with the
    priority tuple ``(preferred_altloc, occupancy)``, where
    ``preferred_altloc`` flags altloc codes we trust most (``"."``, ``"?"``,
    or ``"A"``), and ``occupancy`` breaks ties within the same altloc class.
    This way a missing-altloc row dominates a secondary conformer even if
    both have occupancy 0.5.
    """
    label_seq_col = columns.index("_atom_site.label_seq_id")
    label_alt_col = columns.index("_atom_site.label_alt_id")
    label_comp_col = columns.index("_atom_site.label_comp_id")
    auth_comp_col = columns.index("_atom_site.auth_comp_id")
    label_atom_col = columns.index("_atom_site.label_atom_id")
    auth_atom_col = columns.index("_atom_site.auth_atom_id")
    x_col = columns.index("_atom_site.Cartn_x")
    y_col = columns.index("_atom_site.Cartn_y")
    z_col = columns.index("_atom_site.Cartn_z")
    occupancy_col = columns.index("_atom_site.occupancy")

    best_rows: Dict[Tuple[int, str], Tuple[Tuple[int, float], np.ndarray, str]] = {}

    for row in rows:
        label_seq = row[label_seq_col]
        if label_seq in {"?", "."}:
            continue

        seq_index = int(label_seq) - 1
        atom_name = row[auth_atom_col] if row[auth_atom_col] not in {"?", "."} else row[label_atom_col]
        residue_name = row[auth_comp_col] if row[auth_comp_col] not in {"?", "."} else row[label_comp_col]
        altloc = row[label_alt_col]
        occupancy = float(row[occupancy_col]) if row[occupancy_col] not in {"?", "."} else 0.0

        preferred_altloc = 1 if altloc in {".", "?", "A"} else 0
        priority = (preferred_altloc, occupancy)

        coordinates = np.asarray([float(row[x_col]), float(row[y_col]), float(row[z_col])], dtype=np.float32)
        key = (seq_index, atom_name)

        if key not in best_rows or priority > best_rows[key][0]:
            best_rows[key] = (priority, coordinates, residue_name)

    return {key: (value[1], value[2]) for key, value in best_rows.items()}


def _fallback_sequence(rows: List[List[str]], columns: List[str]) -> str:
    label_seq_col = columns.index("_atom_site.label_seq_id")
    label_comp_col = columns.index("_atom_site.label_comp_id")

    residue_names: Dict[int, str] = {}
    max_seq = 0
    for row in rows:
        label_seq = row[label_seq_col]
        if label_seq in {"?", "."}:
            continue
        seq_index = int(label_seq) - 1
        residue_names[seq_index] = row[label_comp_col]
        max_seq = max(max_seq, seq_index + 1)

    if max_seq == 0:
        raise ValueError("Could not infer a residue sequence from atom_site rows.")

    letters = ["X"] * max_seq
    for seq_index, residue_name in residue_names.items():
        letters[seq_index] = _one_letter_from_resname(residue_name)
    return "".join(letters)


@dataclass
class ChainAtoms:
    """One parsed chain, ready to become a labelled training example.

    Fields match what the downstream data pipeline (``data.py``) expects:

    * ``aatype``: ``(N_res,)`` int IDs using the alphabet from ``a3m.py``.
    * ``residue_index``: ``(N_res,)`` contiguous 0..N-1, per supplement
      1.2.9 — *not* the author numbering from the mmCIF. This is the ID
      fed into ``RelPos`` / Algorithm 4.
    * ``atom14_positions``: ``(N_res, 14, 3)`` Å coordinates in the
      per-residue atom14 slot ordering (from ``residue_constants``).
    * ``atom14_mask``: ``(N_res, 14)`` 1 where a coordinate was present in
      the mmCIF, 0 otherwise — atoms missing in the ground truth are masked
      out of the FAPE / torsion losses.
    * ``resolution``: Å (0.0 when none was found). Consumed by the pLDDT
      and experimentally-resolved loss resolution filters (supplement
      1.9.6 / 1.9.10).
    """

    pdb_id: str
    chain_id: str
    sequence: str
    aatype: np.ndarray
    residue_index: np.ndarray
    atom14_positions: np.ndarray
    atom14_mask: np.ndarray
    resolution: float


def _first_tag_value(
    tag: str,
    scalars: Dict[str, str],
    loops: Iterable[Tuple[List[str], List[List[str]]]],
) -> str | None:
    if tag in scalars:
        return scalars[tag]

    for columns, rows in loops:
        if tag not in columns or not rows:
            continue
        tag_index = columns.index(tag)
        return rows[0][tag_index]

    return None


def _parse_resolution(
    scalars: Dict[str, str],
    loops: Iterable[Tuple[List[str], List[List[str]]]],
) -> float:
    """Extract structure resolution (Å) from mmCIF metadata.

    Different experimental methods record resolution in different tags:
    X-ray crystallography uses ``_refine.ls_d_res_high`` /
    ``_reflns.d_resolution_high``; cryo-EM uses
    ``_em_3d_reconstruction.resolution``. NMR has no resolution — we fall
    through to 0.0, which the loss resolution filters interpret as
    "outside [0.1, 3.0]" and zero out the corresponding terms
    (supplement 1.9.6 / 1.9.10 only train on resolutions in that range).
    """
    for tag in (
        "_refine.ls_d_res_high",
        "_em_3d_reconstruction.resolution",
        "_reflns.d_resolution_high",
    ):
        raw_value = _first_tag_value(tag, scalars, loops)
        if raw_value is None or raw_value in {".", "?"}:
            continue
        try:
            return float(raw_value)
        except ValueError:
            continue
    return 0.0


def extract_chain_atoms(
    mmcif_path: str | Path,
    pdb_id: str,
    chain_id: str,
) -> ChainAtoms:
    """Parse ``mmcif_path`` and return the atom14 structure for ``chain_id``.

    Follows supplement 1.2.1 "Parsing": resolves altlocs by occupancy, uses
    the first model (rejecting NMR multi-model ensembles), filters to
    ``ATOM`` records (skipping ``HETATM`` like ligands and water), and
    derives the one-letter sequence from ``_entity_poly`` where possible,
    falling back to residue names in ``_atom_site`` if the entity table
    is missing. Residues with no ATOM coordinates get zeroed positions
    and an all-zero atom14 mask — the downstream loss masks will drop
    them automatically.
    """
    mmcif_path = Path(mmcif_path)
    scalars, loops = _parse_mmcif(mmcif_path.read_text())
    entity_sequences = _entity_sequences(scalars, loops)
    resolution = _parse_resolution(scalars, loops)

    atom_columns: List[str] | None = None
    atom_rows: List[List[str]] | None = None
    for columns, rows in loops:
        if columns and columns[0].startswith("_atom_site."):
            atom_columns = columns
            atom_rows = rows
            break

    if atom_columns is None or atom_rows is None:
        raise ValueError(f"No _atom_site loop found in {mmcif_path}")

    filtered_rows, _selected_by = _select_atom_rows(atom_columns, atom_rows, chain_id)

    group_col = atom_columns.index("_atom_site.group_PDB")
    model_col = atom_columns.index("_atom_site.pdbx_PDB_model_num")
    entity_col = atom_columns.index("_atom_site.label_entity_id")

    atom_only_rows = [row for row in filtered_rows if row[group_col] == "ATOM"]
    if not atom_only_rows:
        raise ValueError(f"No ATOM rows found for chain '{chain_id}' in {mmcif_path}")

    first_model = atom_only_rows[0][model_col]
    atom_only_rows = [row for row in atom_only_rows if row[model_col] == first_model]

    entity_id = atom_only_rows[0][entity_col]
    sequence = entity_sequences.get(entity_id)
    if sequence is None:
        sequence = _fallback_sequence(atom_only_rows, atom_columns)

    atom14_positions = np.zeros((len(sequence), 14, 3), dtype=np.float32)
    atom14_mask = np.zeros((len(sequence), 14), dtype=np.float32)
    for (seq_index, atom_name), (coordinates, residue_name) in _best_atom_rows(atom_only_rows, atom_columns).items():
        if seq_index < 0 or seq_index >= len(sequence):
            continue
        residue_name = residue_name.upper()
        if residue_name not in ATOM14_INDEX:
            continue
        atom_index = ATOM14_INDEX[residue_name].get(atom_name)
        if atom_index is None:
            continue
        atom14_positions[seq_index, atom_index] = coordinates
        atom14_mask[seq_index, atom_index] = 1.0

    return ChainAtoms(
        pdb_id=pdb_id.lower(),
        chain_id=chain_id,
        sequence=sequence,
        aatype=sequence_to_ids(sequence),
        # Canonical AF2/OpenFold sequence features use contiguous 0..N-1 residue
        # indices from sequence order, not author numbering from the mmCIF.
        residue_index=np.arange(len(sequence), dtype=np.int32),
        atom14_positions=atom14_positions,
        atom14_mask=atom14_mask,
        resolution=resolution,
    )
