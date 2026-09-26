"""Small mmCIF reader for one deposited protein polymer chain.

Sequence indices are full-polymer positions, not author residue numbers. Only
one label chain in the first deposited model is admitted. Missing atoms stay
masked; alternate locations are selected per residue, never per atom. This is
not a general CIF schema implementation: ambiguous mappings fail explicitly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

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
    residue_name: {
        atom_name: atom_idx
        for atom_idx, atom_name in enumerate(atom_names)
        if atom_name
    }
    for residue_name, atom_names in restype_name_to_atom14_names.items()
}


class _Token(str):
    def __new__(cls, text, quoted=False):
        token = super().__new__(cls, text)
        token.quoted = quoted
        return token


def _tokenize_mmcif(text: str) -> list[str]:
    """Read CIF quotes/comments/text fields without applying shell escapes."""
    tokens = []
    lines = text.splitlines()
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        line_index += 1
        if line.startswith(";"):
            block = [line[1:]]
            while line_index < len(lines) and not lines[line_index].startswith(";"):
                block.append(lines[line_index])
                line_index += 1
            if line_index == len(lines) or lines[line_index][1:].strip():
                raise ValueError("Unterminated or malformed CIF text field")
            line_index += 1
            tokens.append(_Token("\n".join(block), quoted=True))
            continue
        pos = 0
        while pos < len(line):
            if line[pos].isspace():
                pos += 1
                continue
            if line[pos] == "#":
                break
            if line[pos] in "\"'":
                quote = line[pos]
                start = pos = pos + 1
                while pos < len(line):
                    if line[pos] == quote and (
                        pos + 1 == len(line) or line[pos + 1].isspace()
                    ):
                        break
                    pos += 1
                if pos == len(line):
                    raise ValueError("Unterminated CIF quoted value")
                tokens.append(_Token(line[start:pos], quoted=True))
                pos += 1
            else:
                start = pos
                while pos < len(line) and not line[pos].isspace():
                    pos += 1
                tokens.append(_Token(line[start:pos]))
    return tokens


def _control(token: str) -> bool:
    return not token.quoted and (
        token.startswith("_")
        or token.lower() in {"loop_", "stop_", "global_"}
        or token.lower().startswith(("data_", "save_"))
    )


def _parse_mmcif(text: str):
    tokens = _tokenize_mmcif(text)
    scalars, loops, seen_tags = {}, [], set()
    index = 0
    blocks = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token.quoted and token.lower().startswith("data_"):
            blocks += 1
            if blocks > 1:
                raise ValueError("Multiple CIF data blocks are not a single structure")
        elif not token.quoted and token.lower() == "loop_":
            columns = []
            while (
                index < len(tokens)
                and not tokens[index].quoted
                and tokens[index].startswith("_")
            ):
                columns.append(str(tokens[index]).lower())
                index += 1
            if (
                not columns
                or len(set(columns)) != len(columns)
                or seen_tags.intersection(columns)
            ):
                raise ValueError("Missing or duplicate CIF loop columns")
            values = []
            while index < len(tokens) and not _control(tokens[index]):
                values.append(str(tokens[index]))
                index += 1
            if len(values) % len(columns):
                raise ValueError("Loop values do not align with loop columns")
            seen_tags.update(columns)
            loops.append(
                (
                    columns,
                    [
                        values[i : i + len(columns)]
                        for i in range(0, len(values), len(columns))
                    ],
                )
            )
            if (
                index < len(tokens)
                and not tokens[index].quoted
                and tokens[index].lower() == "stop_"
            ):
                index += 1
        elif not token.quoted and token.startswith("_"):
            tag = str(token).lower()
            if tag in seen_tags or index == len(tokens) or _control(tokens[index]):
                raise ValueError(f"Missing or duplicate CIF tag {tag}")
            seen_tags.add(tag)
            scalars[tag] = str(tokens[index])
            index += 1
        else:
            raise ValueError(f"Unsupported or unexpected CIF token: {token}")
    return scalars, loops


def _records(prefix, scalars, loops):
    single = {
        key[len(prefix) :]: value
        for key, value in scalars.items()
        if key.startswith(prefix)
    }
    if single:
        yield single
    for columns, rows in loops:
        if any(column.startswith(prefix) for column in columns):
            for row in rows:
                yield {
                    column[len(prefix) :]: value
                    for column, value in zip(columns, row)
                    if column.startswith(prefix)
                }


def _present(value):
    return value not in {None, "", ".", "?"}


def _positive_index(value):
    if not _present(value) or not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError(f"Invalid deposited polymer position: {value}")
    return int(value) - 1


def _clean_sequence(raw_sequence: str, parents=None) -> str:
    parents = parents or {}
    compact = re.sub(r"\s+", "", raw_sequence).upper()
    compact = re.sub(
        r"\(([^()]+)\)",
        lambda m: THREE_TO_ONE.get(parents.get(m[1], m[1]), "X"),
        compact,
    )
    if not compact or not re.fullmatch("[A-Z]+", compact):
        raise ValueError("Invalid polymer sequence")
    return compact


@dataclass
class ChainAtoms:
    """Polymer sequence and atom14 labels; absent atoms remain masked.

    polymer_sequence_complete is false when only an observed atom-row span
    could be inferred. That fallback is unsuitable for training/template caches.

    residue_index stays contiguous for the educational model's RelPos API.
    between_segment_residues marks an unverified/broken deposited adjacency;
    author numbering (including insertion codes) is never used as array indices.
    """

    pdb_id: str
    chain_id: str
    sequence: str
    aatype: np.ndarray
    residue_index: np.ndarray
    atom14_positions: np.ndarray
    atom14_mask: np.ndarray
    resolution: float
    between_segment_residues: np.ndarray | None = None
    release_date: str | None = None
    label_chain_id: str | None = None
    model_id: str | None = None
    polymer_sequence_complete: bool = False


def _metadata(scalars, loops):
    resolution = 0.0
    for tag in (
        "_refine.ls_d_res_high",
        "_em_3d_reconstruction.resolution",
        "_reflns.d_resolution_high",
    ):
        category, key = tag.rsplit(".", 1)
        values = [
            row[key]
            for row in _records(category + ".", scalars, loops)
            if _present(row.get(key))
        ]
        if values:
            parsed = [float(value) for value in values]
            if not all(np.isfinite(value) and value > 0 for value in parsed):
                raise ValueError("Invalid structure resolution")
            resolution = min(parsed)
            break
    # Original release, never deposition date or a later revision date.
    originals = [
        row["date_original"]
        for row in _records("_database_pdb_rev.", scalars, loops)
        if _present(row.get("date_original"))
    ]
    revisions = [
        row["revision_date"]
        for row in _records("_pdbx_audit_revision_history.", scalars, loops)
        if _present(row.get("revision_date"))
    ]
    dates = originals or revisions
    if any(not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) for value in dates):
        raise ValueError("Invalid original release date")
    parsed_dates = [date.fromisoformat(value) for value in dates]
    return resolution, min(parsed_dates).isoformat() if parsed_dates else None


def _residue_atoms(rows, residue_name):
    """Choose one named conformer plus shared atoms, then occupancy.

    Like the current Competition reader, prefer complete backbone support before
    occupancy so alternate labels cannot improve apparent geometry by hiding it.
    Ties are deterministic; zero-occupancy coordinates are not observations.
    """
    sites = {}
    for row in rows:
        atom = row.get("label_atom_id")
        if atom not in ATOM14_INDEX.get(residue_name, {}):
            continue
        occupancy = float(row.get("occupancy", "1"))
        xyz = np.asarray(
            [float(row[key]) for key in ("cartn_x", "cartn_y", "cartn_z")],
            dtype=np.float32,
        )
        if not np.isfinite(occupancy) or occupancy < 0 or not np.isfinite(xyz).all():
            raise ValueError("Nonfinite coordinates or invalid atom occupancy")
        if occupancy == 0:
            continue
        alt = row.get("label_alt_id", ".")
        alt = alt if _present(alt) else ""
        key = (atom, alt)
        candidate = (occupancy, tuple(float(x) for x in xyz))
        if key not in sites or candidate > sites[key]:
            sites[key] = candidate
    alternatives = sorted({alt for _, alt in sites if alt}) or [""]
    candidates = []
    for alt in alternatives:
        atoms = {}
        for atom in ATOM14_INDEX.get(residue_name, {}):
            value = sites.get((atom, alt), sites.get((atom, "")))
            if value is not None:
                atoms[atom] = value
        support = tuple(int(atom in atoms) for atom in ("CA", "N", "C", "O"))
        named = [value[0] for (atom, loc), value in sites.items() if loc == alt]
        occupancy = float(np.mean(named)) if named else 0.0
        candidates.append(((sum(support), support, occupancy), atoms))
    return max(candidates, key=lambda item: item[0])[1]


def extract_chain_atoms(
    mmcif_path: str | Path, pdb_id: str, chain_id: str
) -> ChainAtoms:
    scalars, loops = _parse_mmcif(Path(mmcif_path).read_text())
    resolution, release_date = _metadata(scalars, loops)
    parents = {}
    for row in _records("_chem_comp.", scalars, loops):
        parent = row.get("mon_nstd_parent_comp_id")
        if _present(parent) and parent in THREE_TO_ONE:
            parents[row["id"]] = parent
    entities = {}
    for row in _records("_entity_poly.", scalars, loops):
        entity = row["entity_id"]
        if entity in entities:
            raise ValueError("Duplicate polymer entity")
        kind = row.get("type")
        seq = row.get(
            "pdbx_seq_one_letter_code_can", row.get("pdbx_seq_one_letter_code")
        )
        entities[entity] = (
            _clean_sequence(seq, parents) if _present(seq) else None,
            kind is None or kind in {"polypeptide(L)", "polypeptide(D)"},
        )
    # An explicit sequence table is also complete polymer evidence. Observed
    # atom rows alone cannot establish unobserved termini or total length.
    sequence_tables = {}
    for row in _records("_entity_poly_seq.", scalars, loops):
        table = sequence_tables.setdefault(row["entity_id"], {})
        index = _positive_index(row["num"])
        if index in table:
            raise ValueError(
                "Duplicate or microheterogeneous polymer sequence position"
            )
        name = parents.get(row["mon_id"], row["mon_id"])
        table[index] = THREE_TO_ONE.get(name, "X")
    for entity, table in sequence_tables.items():
        if set(table) != set(range(len(table))):
            raise ValueError("Incomplete deposited polymer sequence table")
        sequence = "".join(table[index] for index in range(len(table)))
        declared, protein = entities.get(entity, (None, True))
        if declared is not None and declared != sequence:
            raise ValueError("Conflicting deposited polymer sequences")
        entities[entity] = (sequence, protein)
    rows = list(_records("_atom_site.", scalars, loops))
    if not rows:
        raise ValueError("No atom_site records")
    for row in rows:
        _positive_index(row.get("pdbx_pdb_model_num", "1"))
    first_model = rows[0].get("pdbx_pdb_model_num", "1")
    rows = [row for row in rows if row.get("pdbx_pdb_model_num", "1") == first_model]
    # Entity membership admits modified polymer HETATM, excludes ligands/water
    # sharing the same author chain. Old minimal fixtures may lack entity tables.
    rows = [
        row
        for row in rows
        if (
            row.get("label_entity_id") in entities
            and entities[row["label_entity_id"]][1]
        )
        or (not entities and row.get("group_pdb") == "ATOM")
    ]
    selected = [row for row in rows if row.get("auth_asym_id") == chain_id]
    if not selected:
        selected = [row for row in rows if row.get("label_asym_id") == chain_id]
    if not selected:
        raise KeyError(
            f"Protein chain {chain_id!r} absent from first deposited model {first_model}"
        )
    identities = {
        (row.get("label_asym_id"), row.get("label_entity_id")) for row in selected
    }
    if len(identities) != 1 or not all(
        _present(value) for value in next(iter(identities))
    ):
        raise ValueError(
            "Author chain maps to multiple or unknown deposited polymer chains"
        )
    label_chain, entity_id = next(iter(identities))
    if (
        len(
            {
                row.get("auth_asym_id")
                for row in selected
                if _present(row.get("auth_asym_id"))
            }
        )
        > 1
    ):
        raise ValueError("Label chain maps to multiple author chains")
    scheme = {}
    for row in _records("_pdbx_poly_seq_scheme.", scalars, loops):
        if row.get("asym_id") != label_chain:
            continue
        author = row.get("auth_seq_num")
        if not _present(author):
            continue
        insertion = row.get("pdb_ins_code")
        key = (author, insertion if _present(insertion) else "")
        index = _positive_index(row["seq_id"])
        if key in scheme and scheme[key] != index:
            raise ValueError("Ambiguous author insertion mapping")
        scheme[key] = index
    residues, author_to_index, index_to_author = {}, {}, {}
    for row in selected:
        author = row.get("auth_seq_id")
        insertion = row.get("pdbx_pdb_ins_code")
        author_key = (author, insertion if _present(insertion) else "")
        label = row.get("label_seq_id")
        if _present(label):
            index = _positive_index(label)
            if author_key in scheme and scheme[author_key] != index:
                raise ValueError(
                    "Atom polymer position disagrees with insertion mapping"
                )
        elif author_key in scheme:
            index = scheme[author_key]
        else:
            raise ValueError(
                "Polymer atom lacks an unambiguous deposited sequence position"
            )
        if _present(author):
            if author_key in author_to_index and author_to_index[author_key] != index:
                raise ValueError("Author residue maps to multiple polymer positions")
            if index in index_to_author and index_to_author[index] != author_key:
                raise ValueError("Multiple author residues map to one polymer position")
            author_to_index[author_key] = index
            index_to_author[index] = author_key
        residues.setdefault(index, []).append(row)
    sequence = entities.get(entity_id, (None, True))[0]
    polymer_sequence_complete = sequence is not None
    residue_names = {}
    for index, atom_rows in residues.items():
        names = {row["label_comp_id"] for row in atom_rows}
        if len(names) != 1:
            raise ValueError(
                "Microheterogeneous residue needs an explicit sequence policy"
            )
        name = names.pop()
        residue_names[index] = parents.get(name, name)
    if sequence is None:
        letters = ["X"] * (max(residues) + 1)
        for index, name in residue_names.items():
            letters[index] = THREE_TO_ONE.get(name, "X")
        sequence = "".join(letters)
    positions = np.zeros((len(sequence), 14, 3), dtype=np.float32)
    mask = np.zeros((len(sequence), 14), dtype=np.float32)
    for index, atom_rows in residues.items():
        if index >= len(sequence):
            raise ValueError("Atom position exceeds deposited polymer sequence")
        name = residue_names[index]
        if THREE_TO_ONE.get(name, "X") != sequence[index]:
            raise ValueError("Atom residue disagrees with deposited polymer sequence")
        for atom, (_occupancy, xyz) in _residue_atoms(atom_rows, name).items():
            slot = ATOM14_INDEX[name][atom]
            positions[index, slot] = xyz
            mask[index, slot] = 1
    segments = np.zeros(len(sequence), dtype=np.int32)
    for index in range(1, len(sequence)):
        if not (mask[index - 1, 2] and mask[index, 0]):
            segments[index] = 1
        else:
            distance = np.linalg.norm(positions[index - 1, 2] - positions[index, 0])
            segments[index] = int(not 1.0 <= distance <= 1.8)
    return ChainAtoms(
        pdb_id.lower(),
        chain_id,
        sequence,
        sequence_to_ids(sequence),
        np.arange(len(sequence), dtype=np.int32),
        positions,
        mask,
        resolution,
        segments,
        release_date,
        label_chain,
        first_model,
        polymer_sequence_complete,
    )
