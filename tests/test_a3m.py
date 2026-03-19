import os
import sys

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "minalphafold"))


from a3m import GAP_ID, MASK_ID, aa_to_id, read_a3m, sequence_to_ids, ungap_query_columns


def test_read_a3m_parses_insertions_and_deletions(tmp_path):
    a3m_path = tmp_path / "toy.a3m"
    a3m_path.write_text(
        ">query\n"
        "ACd-EF\n"
        ">hit1\n"
        "ACh-EF\n"
    )

    a3m = read_a3m(a3m_path)
    msa, deletions = a3m.to_tokens()

    assert msa.shape == (2, 5)
    assert deletions.shape == (2, 5)
    assert msa[0].tolist() == sequence_to_ids("AC-EF").tolist()
    assert deletions[0].tolist() == [0, 0, 1, 0, 0]
    assert deletions[1].tolist() == [0, 0, 1, 0, 0]


def test_ungap_query_columns_projects_to_query_positions():
    msa = np.asarray(
        [
            sequence_to_ids("A-CG"),
            sequence_to_ids("ATCG"),
        ],
        dtype=np.int32,
    )
    deletions = np.asarray(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.int32,
    )

    ungapped_msa, ungapped_deletions, query_sequence = ungap_query_columns(msa, deletions, "A-CG")

    assert query_sequence == "ACG"
    assert ungapped_msa.shape == (2, 3)
    assert ungapped_deletions.shape == (2, 3)
    assert ungapped_msa[0].tolist() == sequence_to_ids("ACG").tolist()
    assert ungapped_msa[1].tolist() == sequence_to_ids("ACG").tolist()
    assert ungapped_deletions[1].tolist() == [0, 0, 0]


def test_token_ids_cover_gap_and_unknown_cases():
    assert aa_to_id("-") == GAP_ID
    assert aa_to_id("x") == 20
    assert MASK_ID == 22
