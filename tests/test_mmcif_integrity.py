"""Deposited residue identities and temporal template isolation using actual CIFs."""

import json

import numpy as np
import pytest
from preprocess_openproteinset import preprocess_chain, template_features

from minalphafold.mmcif import extract_chain_atoms

COLUMNS = [
    "group_PDB",
    "pdbx_PDB_model_num",
    "auth_asym_id",
    "label_asym_id",
    "label_entity_id",
    "label_seq_id",
    "label_alt_id",
    "auth_seq_id",
    "pdbx_PDB_ins_code",
    "label_comp_id",
    "label_atom_id",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
]


def atom(
    index=1,
    atom="CA",
    *,
    model=1,
    author="A",
    chain="A",
    entity=1,
    alt=".",
    author_index=None,
    insertion="?",
    residue="ALA",
    x=1,
    occupancy=1,
    group="ATOM",
):
    return f"{group} {model} {author} {chain} {entity} {index} {alt} {author_index or index} {insertion} {residue} {atom} {x} 0 0 {occupancy}"


def cif(path, rows, sequence="A", metadata="", release="2000-01-01"):
    path.write_text(
        "data_example\n_entity_poly.entity_id 1\n_entity_poly.type 'polypeptide(L)'\n"
        + f"_entity_poly.pdbx_seq_one_letter_code_can {sequence}\n"
        + (f"_database_PDB_rev.date_original {release}\n" if release else "")
        + metadata
        + "\nloop_\n"
        + "\n".join("_atom_site." + col for col in COLUMNS)
        + "\n"
        + "\n".join(rows)
        + "\n#\n"
    )
    return path


def extract(path):
    return extract_chain_atoms(path, "1abc", "A")


def test_insertion_codes_missing_polymer_position_and_first_model(tmp_path):
    rows = [
        atom(1, author_index=10, x=1),
        atom(2, author_index=10, insertion="A", residue="GLY", x=2),
        atom(4, author_index=11, x=4),
        atom(1, model=2, author_index=10, x=99),
    ]
    chain = extract(cif(tmp_path / "one.cif", rows, sequence="AGAA"))
    assert chain.sequence == "AGAA"
    assert chain.residue_index.tolist() == [0, 1, 2, 3]
    assert chain.atom14_positions[:, 1, 0].tolist() == [1, 2, 0, 4]
    assert chain.atom14_mask[:, 1].tolist() == [1, 1, 0, 1]
    assert chain.model_id == "1"
    # It must not switch to another model just because the requested chain is absent.
    cif(tmp_path / "absent.cif", [atom(author="B", chain="B"), atom(model=2)])
    with pytest.raises(KeyError, match="first deposited model"):
        extract(tmp_path / "absent.cif")


def test_missing_label_index_uses_explicit_insertion_scheme(tmp_path):
    scheme = """loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.auth_seq_num
_pdbx_poly_seq_scheme.pdb_ins_code
A 1 10 .
A 2 10 A
"""
    rows = [
        atom("?", author_index=10, x=1),
        atom("?", author_index=10, insertion="A", residue="GLY", x=7),
    ]
    chain = extract(cif(tmp_path / "scheme.cif", rows, "AG", scheme))
    assert chain.atom14_positions[:, 1, 0].tolist() == [1, 7]
    with pytest.raises(ValueError, match="unambiguous"):
        extract(cif(tmp_path / "missing.cif", rows, "AG"))


@pytest.mark.parametrize(
    "rows,match",
    [
        ([atom(), atom(chain="B")], "multiple.*polymer chains"),
        ([atom(), atom(2, author_index=1)], "multiple polymer positions"),
        ([atom(), atom(author_index=2)], "Multiple author residues"),
        ([atom(0)], "Invalid deposited"),
        ([atom(2)], "exceeds deposited"),
        ([atom(residue="GLY")], "disagrees with deposited"),
    ],
)
def test_ambiguous_mapping_never_merges_or_relabels_atoms(tmp_path, rows, match):
    with pytest.raises(ValueError, match=match):
        extract(cif(tmp_path / "bad.cif", rows))


def test_altlocs_are_whole_residues_occupancy_not_atomwise_or_label_a(tmp_path):
    rows = [
        atom(atom="N", alt="A", x=1, occupancy=0.9),
        atom(alt="A", x=2, occupancy=0.2),
        atom(atom="N", alt="B", x=8, occupancy=0.6),
        atom(alt="B", x=9, occupancy=0.8),
        atom(atom="C", x=10),
        atom(atom="O", x=11),
        atom(atom="CB", x=12, occupancy=0),
    ]
    first = extract(cif(tmp_path / "a.cif", rows))
    second = extract(cif(tmp_path / "b.cif", list(reversed(rows))))
    assert first.atom14_positions[0, :4, 0].tolist() == [8, 9, 10, 11]
    assert first.atom14_mask[0, 4] == 0
    np.testing.assert_array_equal(first.atom14_positions, second.atom14_positions)


@pytest.mark.parametrize(
    "kwargs", [{"occupancy": "nan"}, {"occupancy": -1}, {"x": "inf"}]
)
def test_nonfinite_or_negative_observations_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError, match="Nonfinite"):
        extract(cif(tmp_path / "bad.cif", [atom(**kwargs)]))


def test_modified_polymer_and_ligand_same_author_do_not_contaminate(tmp_path):
    metadata = "_chem_comp.id MSE\n_chem_comp.mon_nstd_parent_comp_id MET\n"
    rows = [
        atom(residue="MSE", group="HETATM", x=7),
        atom(residue="HOH", group="HETATM", entity=2, chain="W", x=99),
    ]
    chain = extract(cif(tmp_path / "modified.cif", rows, "M", metadata))
    assert chain.sequence == "M"
    assert chain.atom14_positions[0, 1, 0] == 7
    assert chain.label_chain_id == "A"


def test_cif_text_fields_quotes_comments_and_block_isolation(tmp_path):
    path = cif(
        tmp_path / "text.cif",
        [atom()],
        sequence="\n;A\n;",
        metadata="_note.value 'loop_' # inline comment\n",
    )
    assert extract(path).sequence == "A"
    path.write_text(path.read_text() + "data_another\n_note.value X\n")
    with pytest.raises(ValueError, match="Multiple CIF data"):
        extract(path)
    path.write_text("data_a\n_entity_poly.pdbx_seq_one_letter_code_can\n;A\n")
    with pytest.raises(ValueError, match="Unterminated"):
        extract(path)


def test_release_is_original_not_revision_or_deposition(tmp_path):
    metadata = """loop_
_pdbx_audit_revision_history.ordinal
_pdbx_audit_revision_history.revision_date
1 2001-02-03
2 2024-01-01
"""
    assert (
        extract(
            cif(tmp_path / "dates.cif", [atom()], metadata=metadata, release=None)
        ).release_date
        == "2001-02-03"
    )
    assert (
        extract(
            cif(
                tmp_path / "none.cif",
                [atom()],
                metadata="_pdbx_database_status.recvd_initial_deposition_date 1990-01-01\n",
                release=None,
            )
        ).release_date
        is None
    )


def prepare_target(tmp_path, sequence="AGA", query="AGA"):
    chain_dir = tmp_path / "1abc_A"
    (chain_dir / "a3m").mkdir(parents=True)
    (chain_dir / "hhr").mkdir()
    (chain_dir / "a3m/query.a3m").write_text(">query\n" + query + "\n")
    rows = []
    for index, residue in enumerate(sequence, 1):
        name = {"A": "ALA", "G": "GLY", "S": "SER"}[residue]
        for name_atom, offset in (("N", 0), ("CA", 1.3), ("C", 2.5)):
            rows.append(
                atom(index, atom=name_atom, residue=name, x=(index - 1) * 3.8 + offset)
            )
    cif(tmp_path / "1abc.cif", rows, sequence)
    return chain_dir


def test_parser_to_preprocess_preserves_projected_and_deposited_breaks(tmp_path):
    chain_dir = prepare_target(tmp_path, sequence="AGSA", query="AGA")
    features, labels = preprocess_chain(
        chain_dir,
        mmcif_root=tmp_path,
        max_msa_seqs=2,
        max_templates=0,
        msa_name="query.a3m",
        template_hhr_name="hits.hhr",
        skip_templates=True,
    )
    assert features["residue_index"].tolist() == [0, 1, 2]
    assert features["between_segment_residues"].tolist() == [0, 0, 1]
    assert labels["atom14_positions"][:, 1, 0].tolist() == pytest.approx(
        [1.3, 5.1, 12.7]
    )
    # Consecutive sequence labels alone cannot hide a disconnected peptide.
    path = tmp_path / "1abc.cif"
    path.write_text(path.read_text().replace(" 3.8 0 0 1", " 50 0 0 1"))
    chain = extract(path)
    assert chain.between_segment_residues.tolist()[:2] == [0, 1]


def test_templates_cutoff_actual_cifs_missing_date_same_pdb_and_stale_hhr(tmp_path):
    chain_dir = prepare_target(tmp_path)
    for pdb, release in [
        ("2old", "2020-01-01"),
        ("3new", "2020-01-02"),
        ("4non", None),
        ("5bad", "2010-01-01"),
    ]:
        cif(tmp_path / f"{pdb}.cif", [atom()], "A", release=release)
    hhr = ""
    for pdb, letter in [
        ("1abc", "A"),
        ("3new", "A"),
        ("4non", "A"),
        ("5bad", "G"),
        ("2old", "A"),
    ]:
        hhr += f">{pdb}_A\nQ query 1 A 1\nT {pdb}_A 1 {letter} 1\n"
    (chain_dir / "hhr/hits.hhr").write_text(hhr)
    kwargs = {
        "chain_dir": chain_dir,
        "mmcif_root": tmp_path,
        "target_pdb_id": "1abc",
        "target_chain_id": "A",
        "query_sequence": "AGA",
        "query_length": 3,
        "template_hhr_name": "hits.hhr",
        "max_templates": 4,
    }
    with pytest.raises(ValueError, match="explicit max_template_date"):
        template_features(**kwargs)
    result = template_features(**kwargs, max_template_date="2020-01-01")
    assert result["template_atom14_mask"].shape == (1, 3, 14)
    receipt = json.loads(result["template_provenance"].item())
    assert receipt["max_template_date"] == "2020-01-01"
    assert [item["pdb_id"] for item in receipt["templates"]] == ["2old"]
    assert receipt["templates"][0]["release_date"] == "2020-01-01"
    # The actual target preprocessing route enforces and retains the same policy.
    features, _ = preprocess_chain(
        chain_dir,
        mmcif_root=tmp_path,
        max_msa_seqs=2,
        max_templates=4,
        msa_name="query.a3m",
        template_hhr_name="hits.hhr",
        skip_templates=False,
        max_template_date="2020-01-01",
    )
    assert (
        features["template_provenance"].item() == result["template_provenance"].item()
    )


def test_ambiguous_repeated_sequence_cannot_select_arbitrary_labels(tmp_path):
    chain_dir = prepare_target(tmp_path, sequence="AAAA", query="AAA")
    with pytest.raises(ValueError, match="Ambiguous query-to-polymer"):
        preprocess_chain(
            chain_dir,
            mmcif_root=tmp_path,
            max_msa_seqs=2,
            max_templates=0,
            msa_name="query.a3m",
            template_hhr_name="hits.hhr",
            skip_templates=True,
        )


def test_written_native_cache_breaks_reach_actual_loss_supervision(tmp_path):
    from preprocess_openproteinset import write_cache_pair

    from minalphafold.data import (
        ProcessedOpenProteinSetDataset,
        build_processed_example_from_cropped,
    )

    chain_dir = prepare_target(tmp_path, sequence="AGSA", query="AGA")
    features, labels = preprocess_chain(
        chain_dir,
        mmcif_root=tmp_path,
        max_msa_seqs=2,
        max_templates=0,
        msa_name="query.a3m",
        template_hhr_name="hits.hhr",
        skip_templates=True,
    )
    feature_dir, label_dir = tmp_path / "features", tmp_path / "labels"
    feature_dir.mkdir()
    label_dir.mkdir()
    write_cache_pair(
        feature_dir / "1abc_A.npz", label_dir / "1abc_A.npz", features, labels
    )
    raw = ProcessedOpenProteinSetDataset(feature_dir, label_dir, split="all")[0]
    result = build_processed_example_from_cropped(
        raw,
        msa_depth=1,
        extra_msa_depth=0,
        max_templates=0,
        training=False,
        random_seed=0,
    )
    assert result["residue_index"].tolist() == [0, 1, 2]
    assert result["loss_residue_index"].tolist() == [0, 1, 3]
    assert result["true_torsion_mask"][2, :2].tolist() == [0, 0]
    assert result["true_torsion_mask"][1, :2].tolist() == [1, 1]
    with np.load(feature_dir / "1abc_A.npz", allow_pickle=False) as arrays:
        provenance = json.loads(arrays["preprocessing_provenance"].item())
    assert (
        provenance["parser_policy"] == "first_model_full_polymer_residue_conformer_v2"
    )
    assert len(provenance["sources"]) == 2


def test_unknown_model_and_incomplete_polymer_do_not_enter_preprocessing(tmp_path):
    with pytest.raises(ValueError, match="Invalid deposited"):
        extract(cif(tmp_path / "unknown.cif", [atom(model="?")]))
    chain_dir = prepare_target(tmp_path)
    path = tmp_path / "1abc.cif"
    path.write_text(
        "\n".join(
            line
            for line in path.read_text().splitlines()
            if not line.startswith("_entity_poly.")
        )
    )
    observed = extract(path)
    assert observed.sequence == "AGA"
    assert observed.polymer_sequence_complete is False
    with pytest.raises(ValueError, match="complete deposited polymer sequence"):
        preprocess_chain(
            chain_dir,
            mmcif_root=tmp_path,
            max_msa_seqs=2,
            max_templates=0,
            msa_name="query.a3m",
            template_hhr_name="hits.hhr",
            skip_templates=True,
        )


def test_explicit_sequence_table_and_valid_loop_stop(tmp_path):
    path = cif(tmp_path / "sequence.cif", [atom(2, residue="GLY")], "?")
    path.write_text(
        path.read_text()
        + "stop_\nloop_\n_entity_poly_seq.entity_id\n_entity_poly_seq.num\n_entity_poly_seq.mon_id\n1 1 ALA\n1 2 GLY\n1 3 ALA\nstop_\n"
    )
    result = extract(path)
    assert result.polymer_sequence_complete is True
    assert result.sequence == "AGA"
    assert result.atom14_mask[:, 1].tolist() == [0, 1, 0]
