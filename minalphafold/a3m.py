"""A3M multiple-sequence-alignment parsing and amino-acid tokenisation.

A3M is HHsuite's FASTA-like MSA format. Each record's sequence encodes one
row of an alignment with three character classes:

* **Uppercase letters** are aligned match states (columns of the MSA).
* **Lowercase letters** are insertions relative to the query — they carry
  information but do not occupy an alignment column. The count of lowercase
  characters preceding each match state becomes the ``deletion`` feature
  (supplement 1.2.9, Table 1: ``cluster_has_deletion`` /
  ``cluster_deletion_value``).
* **Dashes** are deletions relative to the query: they occupy an aligned
  column but carry no residue.

This file converts raw A3M text into the two arrays the rest of the pipeline
needs: a ``(N_seq, N_res)`` integer MSA and a matching ``(N_seq, N_res)``
deletion-count array.

Alphabet conventions (supplement 1.9.9):

* ``target_feat`` uses 21 classes — 20 amino acids + unknown — matching
  ``SEQ_ALPHABET_SIZE`` and Table 1 ``aatype``.
* MSA features use 23 classes — 20 amino acids + unknown + gap + mask token
  — matching ``MSA_ALPHABET_SIZE`` and Table 1 ``cluster_msa`` /
  ``extra_msa``.

The one-letter ordering in ``RESTYPES`` matches DeepMind's canonical AF2
alphabet (A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V), which
is what every downstream module assumes. Glycine is index 7 — this is why
pseudo-β helpers and recycling both use ``aatype == 7`` as the GLY check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


# Canonical AF2 one-letter alphabet (20 standard amino acids).
RESTYPES = "ARNDCQEGHILKMFPSTWYV"
RESTYPE_TO_ID = {aa: idx for idx, aa in enumerate(RESTYPES)}

# Fixed IDs for non-standard tokens (supplement 1.9.9: "20 common amino acid
# types, an unknown type, a gap token, and a mask token").
UNK_ID = 20   # any letter outside RESTYPES
GAP_ID = 21   # alignment gap '-'
MASK_ID = 22  # BERT-style mask token used by the masked MSA loss

# Alphabet sizes used by feature builders (Table 1).
SEQ_ALPHABET_SIZE = 21  # target_feat: 20 AAs + unknown
MSA_ALPHABET_SIZE = 23  # cluster_msa / extra_msa: + gap + mask


def aa_to_id(aa: str) -> int:
    """Map a single character (one-letter AA code, '-', or unknown) to an ID."""
    aa = aa.upper()
    if aa == "-":
        return GAP_ID
    return RESTYPE_TO_ID.get(aa, UNK_ID)


def sequence_to_ids(sequence: str) -> np.ndarray:
    """Tokenise an ungapped sequence to an int32 array of AA IDs."""
    return np.fromiter((aa_to_id(aa) for aa in sequence), dtype=np.int32, count=len(sequence))


def ungap_query_columns(
    msa: np.ndarray,
    deletions: np.ndarray,
    query_aligned: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Drop columns where the query has a gap, returning the ungapped target.

    In an A3M alignment the query (first row) can contain dashes when other
    rows inserted residues that couldn't be absorbed as lowercase
    insertions. These gap columns are not part of the original target
    sequence, so we strip them before producing the MSA / deletion arrays
    consumed by the model. Returns (msa without gap columns, deletions
    without gap columns, concatenated target sequence).
    """
    query_mask = np.asarray([char != "-" for char in query_aligned], dtype=bool)
    if query_mask.shape[0] != msa.shape[1]:
        raise ValueError(
            f"Aligned query length {query_mask.shape[0]} does not match MSA width {msa.shape[1]}."
        )

    target_sequence = "".join(char for char in query_aligned if char != "-")
    return msa[:, query_mask], deletions[:, query_mask], target_sequence


@dataclass
class A3M:
    """One parsed A3M file: FASTA-style headers and raw (mixed-case) sequences."""

    headers: List[str]
    seqs_raw: List[str]

    def to_aligned_msa(self) -> Tuple[List[str], np.ndarray]:
        """Split each A3M row into (aligned characters, per-column deletion counts).

        Lowercase characters are insertions relative to the query column grid
        (the ``deletion`` feature from Table 1). Each uppercase character or
        dash sits in one alignment column; lowercase characters before it
        accumulate into that column's deletion count. Returns one
        ``aligned_sequence`` string per row (all uppercase + dash, all the
        same length) and a ``(N_seq, N_res)`` integer array of deletion
        counts.
        """
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
        """Integer-tokenise the aligned MSA and return ``(msa, deletions)``.

        Optional ``max_seqs`` truncates to the first N rows (the query is row
        0 by convention).
        """
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
    """Parse an A3M file from disk (FASTA-style: ``>`` headers, sequence lines)."""
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
