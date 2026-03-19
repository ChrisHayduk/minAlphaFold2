from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
from urllib.error import HTTPError
from urllib.request import urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the minimal OpenProteinSet assets used by this repo.")
    parser.add_argument("--data-root", type=str, default="data/openproteinset")
    parser.add_argument("--msa-name", type=str, default="uniref90_hits.a3m")
    parser.add_argument("--template-hhr-name", type=str, default="pdb70_hits.hhr")
    parser.add_argument("--skip-templates", action="store_true")
    parser.add_argument("--full-alignments", action="store_true")
    parser.add_argument("--chain-id-file", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], dry_run: bool) -> None:
    print("+", " ".join(command))
    if not dry_run:
        subprocess.check_call(command)


def read_chain_ids(chain_id_file: str, limit: int) -> list[str]:
    chain_ids = [
        line.strip()
        for line in Path(chain_id_file).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if limit > 0:
        chain_ids = chain_ids[:limit]
    return chain_ids


def download_url(url: str, destination: Path, dry_run: bool) -> bool:
    print("+", url, "->", destination)
    if dry_run:
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url) as response:
            destination.write_bytes(response.read())
    except HTTPError as exc:
        print(f"WARNING: failed to download {url} ({exc.code})")
        return False
    return True


def subset_alignment_urls(chain_id: str, msa_name: str, template_hhr_name: str, *, skip_templates: bool) -> list[tuple[str, Path]]:
    targets = [
        (
            f"https://openfold.s3.amazonaws.com/pdb/{chain_id}/a3m/{msa_name}",
            Path("roda_pdb") / chain_id / "a3m" / msa_name,
        ),
    ]
    if not skip_templates:
        targets.append(
            (
                f"https://openfold.s3.amazonaws.com/pdb/{chain_id}/hhr/{template_hhr_name}",
                Path("roda_pdb") / chain_id / "hhr" / template_hhr_name,
            )
        )
    return targets


def subset_structure_url(chain_id: str) -> tuple[str, Path]:
    pdb_id = chain_id.split("_", 1)[0].upper()
    return (
        f"https://files.rcsb.org/download/{pdb_id}.cif",
        Path("pdb_data") / "mmcif_files" / f"{pdb_id.lower()}.cif",
    )


def download_subset(
    data_root: Path,
    chain_ids: list[str],
    msa_name: str,
    template_hhr_name: str,
    *,
    skip_templates: bool,
    dry_run: bool,
) -> None:
    for chain_id in chain_ids:
        for url, relative_destination in subset_alignment_urls(
            chain_id,
            msa_name,
            template_hhr_name,
            skip_templates=skip_templates,
        ):
            download_url(url, data_root / relative_destination, dry_run=dry_run)

        structure_url, structure_destination = subset_structure_url(chain_id)
        download_url(structure_url, data_root / structure_destination, dry_run=dry_run)


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

    if args.chain_id_file:
        chain_ids = read_chain_ids(args.chain_id_file, limit=args.limit)
        download_subset(
            data_root,
            chain_ids,
            msa_name=args.msa_name,
            template_hhr_name=args.template_hhr_name,
            skip_templates=args.skip_templates,
            dry_run=args.dry_run,
        )
        return

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
