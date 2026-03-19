from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the minimal OpenProteinSet assets used by this repo.")
    parser.add_argument("--data-root", type=str, default="data/openproteinset")
    parser.add_argument("--msa-name", type=str, default="uniref90_hits.a3m")
    parser.add_argument("--template-hhr-name", type=str, default="pdb70_hits.hhr")
    parser.add_argument("--skip-templates", action="store_true")
    parser.add_argument("--full-alignments", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], dry_run: bool) -> None:
    print("+", " ".join(command))
    if not dry_run:
        subprocess.check_call(command)


def build_alignment_sync_command(
    data_root: Path,
    msa_name: str,
    template_hhr_name: str,
    *,
    skip_templates: bool,
    full_alignments: bool,
) -> list[str]:
    destination = str(data_root / "roda_pdb")
    command = ["aws", "s3", "sync", "s3://openfold/pdb/", destination, "--no-sign-request"]
    if full_alignments:
        return command

    command.extend(["--exclude", "*", "--include", f"*/a3m/{msa_name}"])
    if not skip_templates:
        command.extend(["--include", f"*/hhr/{template_hhr_name}"])
    return command


def normalize_alignment_layout(data_root: Path, msa_name: str, template_hhr_name: str, *, skip_templates: bool) -> None:
    roda_root = data_root / "roda_pdb"
    for chain_dir in sorted(path for path in roda_root.iterdir() if path.is_dir()):
        msa_path = chain_dir / msa_name
        if msa_path.exists():
            target_dir = chain_dir / "a3m"
            target_dir.mkdir(exist_ok=True)
            msa_path.rename(target_dir / msa_name)

        if skip_templates:
            continue

        hhr_path = chain_dir / template_hhr_name
        if hhr_path.exists():
            target_dir = chain_dir / "hhr"
            target_dir.mkdir(exist_ok=True)
            hhr_path.rename(target_dir / template_hhr_name)


def expand_duplicate_alignments(roda_root: Path, duplicate_chains_file: Path) -> None:
    if not duplicate_chains_file.exists():
        return

    for line in duplicate_chains_file.read_text().splitlines():
        chain_ids = [token.strip() for token in line.split() if token.strip()]
        if len(chain_ids) < 2:
            continue

        representative = None
        for chain_id in chain_ids:
            candidate = roda_root / chain_id
            if candidate.exists():
                representative = candidate
                break

        if representative is None:
            continue

        for chain_id in chain_ids:
            target = roda_root / chain_id
            if target.exists():
                continue
            os.symlink(representative, target, target_is_directory=True)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "roda_pdb").mkdir(exist_ok=True)
    (data_root / "pdb_data").mkdir(exist_ok=True)

    run_command(
        build_alignment_sync_command(
            data_root,
            args.msa_name,
            args.template_hhr_name,
            skip_templates=args.skip_templates,
            full_alignments=args.full_alignments,
        ),
        dry_run=args.dry_run,
    )
    run_command(
        [
            "aws",
            "s3",
            "cp",
            "s3://openfold/pdb_mmcif.zip",
            str(data_root / "pdb_data" / "pdb_mmcif.zip"),
            "--no-sign-request",
        ],
        dry_run=args.dry_run,
    )
    run_command(
        [
            "aws",
            "s3",
            "cp",
            "s3://openfold/duplicate_pdb_chains.txt",
            str(data_root / "pdb_data" / "duplicate_pdb_chains.txt"),
            "--no-sign-request",
        ],
        dry_run=args.dry_run,
    )
    run_command(
        [
            "unzip",
            "-o",
            str(data_root / "pdb_data" / "pdb_mmcif.zip"),
            "-d",
            str(data_root / "pdb_data"),
        ],
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        normalize_alignment_layout(
            data_root,
            msa_name=args.msa_name,
            template_hhr_name=args.template_hhr_name,
            skip_templates=args.skip_templates,
        )
        expand_duplicate_alignments(
            data_root / "roda_pdb",
            data_root / "pdb_data" / "duplicate_pdb_chains.txt",
        )


if __name__ == "__main__":
    main()
