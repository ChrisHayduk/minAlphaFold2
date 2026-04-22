import json
import os
from pathlib import Path

import numpy as np

from download_openproteinset import (
    build_alignment_sync_command,
    read_chain_ids,
    expand_duplicate_alignments,
    normalize_alignment_layout,
    subset_alignment_urls,
    subset_structure_url,
)
from filter_openproteinset import (
    build_manifest,
    load_cluster_tsv,
    max_single_aa_fraction,
)
from minalphafold.data import load_accepted_chains_from_manifest
from preprocess_openproteinset import alignment_pairs_with_offsets, parse_hhr_hits


def test_build_alignment_sync_command_filters_to_minimal_assets():
    command = build_alignment_sync_command(
        Path("data/openproteinset"),
        "uniref90_hits.a3m",
        "pdb70_hits.hhr",
        skip_templates=False,
        full_alignments=False,
    )

    assert command[:4] == ["aws", "s3", "sync", "s3://openfold/pdb/"]
    assert "--include" in command
    assert "*/a3m/uniref90_hits.a3m" in command
    assert "*/hhr/pdb70_hits.hhr" in command


def test_normalize_alignment_layout_moves_flat_files_into_subdirectories(tmp_path):
    chain_dir = tmp_path / "roda_pdb" / "1abc_A"
    chain_dir.mkdir(parents=True)
    (chain_dir / "uniref90_hits.a3m").write_text(">query\nAAA\n")
    (chain_dir / "pdb70_hits.hhr").write_text("")

    normalize_alignment_layout(tmp_path, "uniref90_hits.a3m", "pdb70_hits.hhr", skip_templates=False)

    assert not (chain_dir / "uniref90_hits.a3m").exists()
    assert not (chain_dir / "pdb70_hits.hhr").exists()
    assert (chain_dir / "a3m" / "uniref90_hits.a3m").exists()
    assert (chain_dir / "hhr" / "pdb70_hits.hhr").exists()


def test_expand_duplicate_alignments_creates_missing_symlink(tmp_path):
    roda_root = tmp_path / "roda_pdb"
    representative = roda_root / "1abc_A"
    representative.mkdir(parents=True)
    duplicate_file = tmp_path / "duplicate_pdb_chains.txt"
    duplicate_file.write_text("1abc_A 2xyz_B\n")

    expand_duplicate_alignments(roda_root, duplicate_file)

    duplicate_dir = roda_root / "2xyz_B"
    assert duplicate_dir.is_symlink()
    assert duplicate_dir.resolve() == representative.resolve()


def test_read_chain_ids_respects_comments_and_limit(tmp_path):
    chain_file = tmp_path / "chains.txt"
    chain_file.write_text("# comment\n1abc_A\n\n2xyz_B\n3def_C\n")

    chain_ids = read_chain_ids(str(chain_file), limit=2)

    assert chain_ids == ["1abc_A", "2xyz_B"]


def test_subset_urls_use_openfold_for_alignments_and_rcsb_for_structures():
    alignment_urls = subset_alignment_urls(
        "1abc_A",
        "uniref90_hits.a3m",
        "pdb70_hits.hhr",
        skip_templates=False,
    )
    structure_url = subset_structure_url("1abc_A")

    assert alignment_urls[0][0] == "https://openfold.s3.amazonaws.com/pdb/1abc_A/a3m/uniref90_hits.a3m"
    assert alignment_urls[1][0] == "https://openfold.s3.amazonaws.com/pdb/1abc_A/hhr/pdb70_hits.hhr"
    assert structure_url[0] == "https://files.rcsb.org/download/1ABC.cif"


def test_alignment_pairs_with_offsets_keep_absolute_positions():
    pairs = alignment_pairs_with_offsets(2, "A-GT", 5, "ACGT")
    assert pairs == [(1, 4), (2, 6), (3, 7)]


def test_parse_hhr_hits_reconstructs_template_pairs_with_offsets(tmp_path):
    hhr_path = tmp_path / "toy.hhr"
    hhr_path.write_text(
        "No 1\n"
        ">2XYZ_A Example template\n"
        "Q query             2 A-GT 5 (8)\n"
        "Q Consensus         2 A-GT 5 (8)\n"
        "T Consensus         5 ACGT 8 (10)\n"
        "T 2XYZ_A            5 ACGT 8 (10)\n"
    )

    hits = parse_hhr_hits(hhr_path)

    assert len(hits) == 1
    assert hits[0].pdb_id == "2xyz"
    assert hits[0].chain_id == "A"
    assert hits[0].aligned_pairs == ((1, 4), (2, 6), (3, 7))


# ---------------------------------------------------------------------
# Filter manifest — supplement §1.2.5 deterministic filters.
# ---------------------------------------------------------------------


def _write_minimal_chain_npzs(
    feature_dir: Path,
    label_dir: Path,
    chain_id: str,
    *,
    aatype: list[int],
    resolution: float,
) -> None:
    """Write a minimal pair of feature + label NPZs for a filter test.

    The filter only reads ``aatype`` from features and ``resolution``
    from labels, so we skip every other field.
    """
    np.savez_compressed(
        feature_dir / f"{chain_id}.npz",
        aatype=np.asarray(aatype, dtype=np.int32),
    )
    np.savez_compressed(
        label_dir / f"{chain_id}.npz",
        resolution=np.asarray(resolution, dtype=np.float32),
    )


def test_max_single_aa_fraction_computes_dominant_share():
    aatype = np.array([0, 0, 0, 0, 1, 2, 3, 3, 3, 3], dtype=np.int32)
    # Both 0 and 3 appear 4/10 times → max fraction 0.4.
    assert max_single_aa_fraction(aatype) == 0.4
    # All-one-AA → 1.0.
    assert max_single_aa_fraction(np.zeros(20, dtype=np.int32)) == 1.0


def test_build_manifest_rejects_on_resolution_and_single_aa(tmp_path):
    feature_dir = tmp_path / "features"
    label_dir = tmp_path / "labels"
    feature_dir.mkdir()
    label_dir.mkdir()

    # Clean chain — should be accepted.
    _write_minimal_chain_npzs(
        feature_dir, label_dir, "1aaa_A",
        aatype=list(range(20)) * 2,  # 40 residues, every AA once
        resolution=2.0,
    )
    # Low-resolution chain — rejected by the 9 Å cutoff.
    _write_minimal_chain_npzs(
        feature_dir, label_dir, "1bbb_A",
        aatype=list(range(20)) * 2,
        resolution=12.5,
    )
    # Single-AA-dominant chain — rejected by the 80 % cutoff.
    _write_minimal_chain_npzs(
        feature_dir, label_dir, "1ccc_A",
        aatype=[7] * 95 + [1, 2, 3, 4, 5],  # 95 % of residues are AA 7
        resolution=2.0,
    )

    manifest = build_manifest(
        processed_features_dir=feature_dir,
        processed_labels_dir=label_dir,
        max_resolution=9.0,
        max_single_aa_fraction_threshold=0.8,
        min_length=10,
        mmseqs_cluster_tsv=None,
    )

    entries = {entry["chain_id"]: entry for entry in manifest["chains"]}
    assert entries["1aaa_A"]["accepted"] is True
    assert entries["1bbb_A"]["accepted"] is False
    assert "resolution" in entries["1bbb_A"]["reject_reasons"]
    assert entries["1ccc_A"]["accepted"] is False
    assert "single_aa" in entries["1ccc_A"]["reject_reasons"]
    assert manifest["summary"]["accepted"] == 1
    assert manifest["summary"]["rejected_resolution"] == 1
    assert manifest["summary"]["rejected_single_aa"] == 1


def test_build_manifest_applies_min_length_filter(tmp_path):
    feature_dir = tmp_path / "features"
    label_dir = tmp_path / "labels"
    feature_dir.mkdir()
    label_dir.mkdir()
    # Chain of length 3 with min_length=10 → rejected.
    _write_minimal_chain_npzs(
        feature_dir, label_dir, "1tiny_A",
        aatype=[0, 1, 2],
        resolution=2.0,
    )
    manifest = build_manifest(
        processed_features_dir=feature_dir,
        processed_labels_dir=label_dir,
        max_resolution=9.0,
        max_single_aa_fraction_threshold=0.8,
        min_length=10,
        mmseqs_cluster_tsv=None,
    )
    entry = manifest["chains"][0]
    assert entry["accepted"] is False
    assert "min_length" in entry["reject_reasons"]


def test_load_cluster_tsv_counts_cluster_sizes(tmp_path):
    tsv = tmp_path / "clusters.tsv"
    tsv.write_text(
        "rep1\t1abc_A\n"
        "rep1\t1abc_B\n"
        "rep1\t1xyz_C\n"
        "rep2\t2aaa_A\n"
    )
    info = load_cluster_tsv(tsv)
    assert info["1abc_A"] == ("rep1", 3)
    assert info["1xyz_C"] == ("rep1", 3)
    assert info["2aaa_A"] == ("rep2", 1)


def test_build_manifest_with_cluster_tsv_embeds_cluster_sizes(tmp_path):
    feature_dir = tmp_path / "features"
    label_dir = tmp_path / "labels"
    feature_dir.mkdir()
    label_dir.mkdir()
    _write_minimal_chain_npzs(
        feature_dir, label_dir, "1aaa_A",
        aatype=list(range(20)) * 2,
        resolution=2.0,
    )
    tsv = tmp_path / "clusters.tsv"
    tsv.write_text("rep1\t1aaa_A\nrep1\t1other_A\n")  # rep1 has size 2

    manifest = build_manifest(
        processed_features_dir=feature_dir,
        processed_labels_dir=label_dir,
        max_resolution=9.0,
        max_single_aa_fraction_threshold=0.8,
        min_length=10,
        mmseqs_cluster_tsv=tsv,
    )
    entry = manifest["chains"][0]
    assert entry["cluster_id"] == "rep1"
    assert entry["cluster_size"] == 2


def test_load_accepted_chains_from_manifest_drops_rejected(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "chains": [
            {"chain_id": "pass_1", "accepted": True, "reject_reasons": []},
            {"chain_id": "pass_2", "accepted": True, "reject_reasons": []},
            {"chain_id": "fail_1", "accepted": False, "reject_reasons": ["resolution"]},
        ],
    }))
    accepted = load_accepted_chains_from_manifest(manifest_path)
    assert accepted == ["pass_1", "pass_2"]
