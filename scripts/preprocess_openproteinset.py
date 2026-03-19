from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "minalphafold") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "minalphafold"))

from a3m import read_a3m, sequence_to_ids, ungap_query_columns
from mmcif import extract_chain_atoms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess OpenProteinSet chains into per-chain caches.")
    parser.add_argument("--raw-root", type=str, default="data/openproteinset")
    parser.add_argument("--processed-features-dir", type=str, default="data/processed_features")
    parser.add_argument("--processed-labels-dir", type=str, default="data/processed_labels")
    parser.add_argument("--max-msa-seqs", type=int, default=2048)
    parser.add_argument("--msa-depth", type=int, default=192)
    parser.add_argument("--extra-msa-depth", type=int, default=1024)
    parser.add_argument("--max-templates", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--msa-name", type=str, default="uniref90_hits.a3m")
    parser.add_argument("--template-hhr-name", type=str, default="pdb70_hits.hhr")
    parser.add_argument("--skip-templates", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class HHRHit:
    pdb_id: str
    chain_id: str
    aligned_pairs: tuple[tuple[int, int], ...]


def global_alignment_pairs(query: str, target: str) -> List[Tuple[int, int]]:
    if not query or not target:
        return []

    match_score = 2
    mismatch_score = -1
    gap_score = -2

    scores = [[0] * (len(target) + 1) for _ in range(len(query) + 1)]
    trace = [[""] * (len(target) + 1) for _ in range(len(query) + 1)]

    for i in range(1, len(query) + 1):
        scores[i][0] = i * gap_score
        trace[i][0] = "U"
    for j in range(1, len(target) + 1):
        scores[0][j] = j * gap_score
        trace[0][j] = "L"

    for i in range(1, len(query) + 1):
        for j in range(1, len(target) + 1):
            diag = scores[i - 1][j - 1] + (match_score if query[i - 1] == target[j - 1] else mismatch_score)
            up = scores[i - 1][j] + gap_score
            left = scores[i][j - 1] + gap_score
            best = max(diag, up, left)
            scores[i][j] = best
            if best == diag:
                trace[i][j] = "D"
            elif best == up:
                trace[i][j] = "U"
            else:
                trace[i][j] = "L"

    pairs: List[Tuple[int, int]] = []
    i = len(query)
    j = len(target)
    while i > 0 or j > 0:
        step = trace[i][j]
        if step == "D":
            i -= 1
            j -= 1
            if query[i] == target[j]:
                pairs.append((i, j))
        elif step == "U":
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def project_to_query(
    query_sequence: str,
    structure_sequence: str,
    atom14_positions: np.ndarray,
    atom14_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pairs = global_alignment_pairs(query_sequence, structure_sequence)
    projected_positions = np.zeros((len(query_sequence), 14, 3), dtype=np.float32)
    projected_mask = np.zeros((len(query_sequence), 14), dtype=np.float32)

    for query_index, structure_index in pairs:
        projected_positions[query_index] = atom14_positions[structure_index]
        projected_mask[query_index] = atom14_mask[structure_index]

    return projected_positions, projected_mask


def _parse_hhr_token(token: str) -> tuple[str, str] | None:
    token = token.strip()
    match = re.match(r"^([0-9A-Za-z]{4})[_\-]([0-9A-Za-z]+)$", token)
    if match:
        return match.group(1).lower(), match.group(2)

    match = re.match(r"^([0-9A-Za-z]{4})([0-9A-Za-z]+)$", token)
    if match:
        return match.group(1).lower(), match.group(2)
    return None


def alignment_pairs_with_offsets(
    query_start: int,
    query_aligned: str,
    template_start: int,
    template_aligned: str,
) -> List[Tuple[int, int]]:
    if len(query_aligned) != len(template_aligned):
        raise ValueError("Aligned query and template strings must have the same length.")

    pairs: List[Tuple[int, int]] = []
    query_index = query_start - 1
    template_index = template_start - 1
    for query_char, template_char in zip(query_aligned, template_aligned):
        query_has = query_char != "-"
        template_has = template_char != "-"
        if query_has and template_has:
            pairs.append((query_index, template_index))
        if query_has:
            query_index += 1
        if template_has:
            template_index += 1
    return pairs


def parse_hhr_hits(hhr_path: Path) -> List[HHRHit]:
    hits: List[HHRHit] = []
    current_template: tuple[str, str] | None = None
    chunk_pairs: List[tuple[int, str, int, str]] = []
    pending_query: tuple[int, str] | None = None

    def flush() -> None:
        nonlocal current_template, chunk_pairs, pending_query
        if current_template is None:
            return
        aligned_pairs: List[Tuple[int, int]] = []
        for query_start, query_aligned, template_start, template_aligned in chunk_pairs:
            aligned_pairs.extend(
                alignment_pairs_with_offsets(
                    query_start=query_start,
                    query_aligned=query_aligned,
                    template_start=template_start,
                    template_aligned=template_aligned,
                )
            )
        if aligned_pairs:
            hits.append(
                HHRHit(
                    pdb_id=current_template[0],
                    chain_id=current_template[1],
                    aligned_pairs=tuple(aligned_pairs),
                )
            )
        current_template = None
        chunk_pairs = []
        pending_query = None

    for raw_line in hhr_path.read_text(errors="ignore").splitlines():
        line = raw_line.rstrip()
        if line.startswith("No "):
            flush()
            continue
        if line.startswith(">"):
            flush()
            token = line[1:].strip().split()[0]
            current_template = _parse_hhr_token(token)
            continue
        if current_template is None:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        if parts[0] == "Q" and parts[1] == "query":
            try:
                pending_query = (int(parts[2]), parts[3].strip())
            except ValueError:
                pending_query = None
        elif parts[0] == "T" and parts[1] not in {"Consensus", "ss_dssp", "ss_pred", "ss_conf"}:
            if pending_query is None:
                continue
            try:
                template_start = int(parts[2])
            except ValueError:
                continue
            query_start, query_aligned = pending_query
            template_aligned = parts[3].strip()
            if len(query_aligned) == len(template_aligned):
                chunk_pairs.append((query_start, query_aligned, template_start, template_aligned))

    flush()
    return hits


def empty_templates(length: int) -> dict[str, np.ndarray]:
    return {
        "template_aatype": np.zeros((0, length), dtype=np.int32),
        "template_atom14_positions": np.zeros((0, length, 14, 3), dtype=np.float32),
        "template_atom14_mask": np.zeros((0, length, 14), dtype=np.float32),
    }


def template_features(
    chain_dir: Path,
    mmcif_root: Path,
    target_pdb_id: str,
    target_chain_id: str,
    query_sequence: str,
    query_length: int,
    template_hhr_name: str,
    max_templates: int,
) -> dict[str, np.ndarray]:
    hhr_path = chain_dir / "hhr" / template_hhr_name
    if not hhr_path.exists() or max_templates <= 0:
        return empty_templates(query_length)

    template_aatype_list: List[np.ndarray] = []
    template_positions_list: List[np.ndarray] = []
    template_mask_list: List[np.ndarray] = []

    for hit in parse_hhr_hits(hhr_path):
        if hit.pdb_id == target_pdb_id and hit.chain_id == target_chain_id:
            continue

        template_path = mmcif_root / f"{hit.pdb_id}.cif"
        if not template_path.exists():
            continue

        try:
            template_chain = extract_chain_atoms(template_path, hit.pdb_id, hit.chain_id)
        except Exception:
            continue

        template_aatype = np.full((query_length,), fill_value=20, dtype=np.int32)
        template_positions = np.zeros((query_length, 14, 3), dtype=np.float32)
        template_mask = np.zeros((query_length, 14), dtype=np.float32)

        for query_index, template_index in hit.aligned_pairs:
            if query_index >= query_length or template_index >= len(template_chain.sequence):
                continue
            template_aatype[query_index] = template_chain.aatype[template_index]
            template_positions[query_index] = template_chain.atom14_positions[template_index]
            template_mask[query_index] = template_chain.atom14_mask[template_index]

        if float(template_mask[:, 1].sum()) < 1:
            continue

        template_aatype_list.append(template_aatype)
        template_positions_list.append(template_positions)
        template_mask_list.append(template_mask)

        if len(template_aatype_list) >= max_templates:
            break

    if not template_aatype_list:
        return empty_templates(query_length)

    return {
        "template_aatype": np.stack(template_aatype_list).astype(np.int32),
        "template_atom14_positions": np.stack(template_positions_list).astype(np.float32),
        "template_atom14_mask": np.stack(template_mask_list).astype(np.float32),
    }


def iter_chain_dirs(roda_root: Path) -> Iterable[Path]:
    for path in sorted(roda_root.iterdir()):
        if path.is_dir():
            yield path


def preprocess_chain(
    chain_dir: Path,
    *,
    mmcif_root: Path,
    max_msa_seqs: int,
    max_templates: int,
    msa_name: str,
    template_hhr_name: str,
    skip_templates: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    chain_id = chain_dir.name
    pdb_id, auth_chain_id = chain_id.split("_", 1)
    msa_path = chain_dir / "a3m" / msa_name
    mmcif_path = mmcif_root / f"{pdb_id.lower()}.cif"

    if not msa_path.exists():
        raise FileNotFoundError(f"Missing MSA file: {msa_path}")
    if not mmcif_path.exists():
        raise FileNotFoundError(f"Missing mmCIF file: {mmcif_path}")

    a3m = read_a3m(msa_path)
    msa, deletions = a3m.to_tokens(max_seqs=max_msa_seqs)
    aligned_sequences, _ = a3m.to_aligned_msa()
    query_aligned = aligned_sequences[0]
    msa, deletions, query_sequence = ungap_query_columns(msa, deletions, query_aligned)

    structure_chain = extract_chain_atoms(mmcif_path, pdb_id.lower(), auth_chain_id)
    if structure_chain.sequence.upper() == query_sequence.upper():
        atom14_positions = structure_chain.atom14_positions.astype(np.float32)
        atom14_mask = structure_chain.atom14_mask.astype(np.float32)
    else:
        atom14_positions, atom14_mask = project_to_query(
            query_sequence,
            structure_chain.sequence,
            structure_chain.atom14_positions,
            structure_chain.atom14_mask,
        )

    features = {
        "aatype": sequence_to_ids(query_sequence).astype(np.int32),
        "msa": msa.astype(np.int32),
        "deletions": deletions.astype(np.int32),
    }
    labels = {
        "atom14_positions": atom14_positions.astype(np.float32),
        "atom14_mask": atom14_mask.astype(np.float32),
    }

    if skip_templates:
        features.update(empty_templates(len(query_sequence)))
    else:
        features.update(
            template_features(
                chain_dir,
                mmcif_root,
                target_pdb_id=pdb_id.lower(),
                target_chain_id=auth_chain_id,
                query_sequence=query_sequence,
                query_length=len(query_sequence),
                template_hhr_name=template_hhr_name,
                max_templates=max_templates,
            )
        )

    return features, labels


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    roda_root = raw_root / "roda_pdb"
    mmcif_root = raw_root / "pdb_data" / "mmcif_files"
    processed_features_dir = Path(args.processed_features_dir)
    processed_labels_dir = Path(args.processed_labels_dir)

    processed_features_dir.mkdir(parents=True, exist_ok=True)
    processed_labels_dir.mkdir(parents=True, exist_ok=True)

    chain_dirs = list(iter_chain_dirs(roda_root))
    if args.limit > 0:
        chain_dirs = chain_dirs[: args.limit]

    ok = 0
    failed = 0
    for chain_dir in chain_dirs:
        chain_id = chain_dir.name
        feature_path = processed_features_dir / f"{chain_id}.npz"
        label_path = processed_labels_dir / f"{chain_id}.npz"
        if not args.overwrite and feature_path.exists() and label_path.exists():
            ok += 1
            continue

        try:
            features, labels = preprocess_chain(
                chain_dir,
                mmcif_root=mmcif_root,
                max_msa_seqs=args.max_msa_seqs,
                max_templates=args.max_templates,
                msa_name=args.msa_name,
                template_hhr_name=args.template_hhr_name,
                skip_templates=args.skip_templates,
            )
            np.savez_compressed(feature_path, **features)
            np.savez_compressed(label_path, **labels)
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"Failed to preprocess {chain_id}: {exc}")

    print(f"Preprocess complete: ok={ok}, failed={failed}")


if __name__ == "__main__":
    main()
