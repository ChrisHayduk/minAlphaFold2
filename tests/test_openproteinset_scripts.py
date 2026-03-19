import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


from download_openproteinset import (
    build_alignment_sync_command,
    expand_duplicate_alignments,
    normalize_alignment_layout,
)
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
