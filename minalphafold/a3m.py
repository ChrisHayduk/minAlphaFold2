from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


RESTYPES = "ARNDCQEGHILKMFPSTWYV"
RESTYPE_TO_ID = {aa: idx for idx, aa in enumerate(RESTYPES)}

UNK_ID = 20
GAP_ID = 21
MASK_ID = 22

SEQ_ALPHABET_SIZE = 21
MSA_ALPHABET_SIZE = 23


def aa_to_id(aa: str) -> int:
    aa = aa.upper()
    if aa == "-":
        return GAP_ID
    return RESTYPE_TO_ID.get(aa, UNK_ID)


def sequence_to_ids(sequence: str) -> np.ndarray:
    return np.fromiter((aa_to_id(aa) for aa in sequence), dtype=np.int32, count=len(sequence))


def ungap_query_columns(
    msa: np.ndarray,
    deletions: np.ndarray,
    query_aligned: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    query_mask = np.asarray([char != "-" for char in query_aligned], dtype=bool)
    if query_mask.shape[0] != msa.shape[1]:
        raise ValueError(
            f"Aligned query length {query_mask.shape[0]} does not match MSA width {msa.shape[1]}."
        )

    target_sequence = "".join(char for char in query_aligned if char != "-")
    return msa[:, query_mask], deletions[:, query_mask], target_sequence


@dataclass
class A3M:
    headers: List[str]
    seqs_raw: List[str]

    def to_aligned_msa(self) -> Tuple[List[str], np.ndarray]:
        aligned_sequences: List[str] = []
        deletion_rows: List[List[int]] = []

        for raw_sequence in self.seqs_raw:
            aligned_chars: List[str] = []
            deletion_counts: List[int] = []
            insertions_since_last_column = 0

            for char in raw_sequence:
                if char.islower():
                    insertions_since_last_column += 1
                    continue

                aligned_chars.append(char.upper())
                deletion_counts.append(insertions_since_last_column)
                insertions_since_last_column = 0

            aligned_sequence = "".join(aligned_chars)
            aligned_sequences.append(aligned_sequence)
            deletion_rows.append(deletion_counts)

        aligned_lengths = {len(sequence) for sequence in aligned_sequences}
        if len(aligned_lengths) != 1:
            raise ValueError(f"Inconsistent aligned lengths in A3M: {sorted(aligned_lengths)}")

        deletions = np.asarray(deletion_rows, dtype=np.int32)
        return aligned_sequences, deletions

    def to_tokens(self, max_seqs: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        aligned_sequences, deletions = self.to_aligned_msa()
        if max_seqs is not None:
            aligned_sequences = aligned_sequences[:max_seqs]
            deletions = deletions[:max_seqs]

        n_sequences = len(aligned_sequences)
        n_columns = len(aligned_sequences[0])

        msa = np.zeros((n_sequences, n_columns), dtype=np.int32)
        for row_index, sequence in enumerate(aligned_sequences):
            msa[row_index] = np.fromiter((aa_to_id(char) for char in sequence), dtype=np.int32, count=n_columns)

        return msa, deletions


def read_a3m(path: str | Path) -> A3M:
    path = Path(path)

    headers: List[str] = []
    sequences: List[str] = []

    current_header: str | None = None
    current_parts: List[str] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                headers.append(current_header)
                sequences.append("".join(current_parts))
            current_header = line[1:]
            current_parts = []
            continue
        current_parts.append(line)

    if current_header is not None:
        headers.append(current_header)
        sequences.append("".join(current_parts))

    if not sequences:
        raise ValueError(f"No sequences found in {path}")

    return A3M(headers=headers, seqs_raw=sequences)
