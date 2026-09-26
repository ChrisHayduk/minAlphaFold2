from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError
from urllib.request import urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the minimal OpenProteinSet assets used by this repo.")
    parser.add_argument("--data-root", type=str, default="data/openproteinset")
    parser.add_argument("--msa-name", type=str, default="uniref90_hits.a3m")
    parser.add_argument("--msa-names", type=str, default="")
    parser.add_argument("--template-hhr-name", type=str, default="pdb70_hits.hhr")
    parser.add_argument("--skip-templates", action="store_true")
    parser.add_argument("--full-alignments", action="store_true")
    parser.add_argument("--chain-id-file", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_msa_names(
    msa_name: str,
    msa_names: str | Sequence[str] | None = None,
) -> list[str]:
    if msa_names is None:
        return [msa_name]
    if isinstance(msa_names, str):
        parsed = [token.strip() for token in msa_names.split(",") if token.strip()]
    else:
        parsed = [token.strip() for token in msa_names if token.strip()]
    return parsed or [msa_name]


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
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    for chain_id in chain_ids:
        _validate_chain_id(chain_id)
    if len(set(chain_ids)) != len(chain_ids):
        raise ValueError("Duplicate requested chain IDs")
    if limit > 0:
        chain_ids = chain_ids[:limit]
    return chain_ids


def _validate_chain_id(chain_id: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9]{4}_[A-Za-z0-9]+", chain_id) is None:
        raise ValueError(f"Invalid chain ID: {chain_id}")


def _validate_filename(name: str) -> None:
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("Asset names must be plain filenames")


def download_url(url: str, destination: Path, dry_run: bool) -> bool:
    print("+", url, "->", destination)
    if dry_run:
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if not data:
            raise ValueError(f"Empty downloaded asset: {url}")
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    except HTTPError as exc:
        print(f"WARNING: failed to download {url} ({exc.code})")
        return False
    return True


def subset_alignment_urls(
    chain_id: str,
    msa_name: str | Sequence[str],
    template_hhr_name: str,
    *,
    skip_templates: bool,
) -> list[tuple[str, Path]]:
    _validate_chain_id(chain_id)
    msa_names = [msa_name] if isinstance(msa_name, str) else list(msa_name)
    for name in [*msa_names, template_hhr_name]:
        _validate_filename(name)
    targets = [
        (
            f"https://openfold.s3.amazonaws.com/pdb/{chain_id}/a3m/{name}",
            Path("roda_pdb") / chain_id / "a3m" / name,
        )
        for name in msa_names
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
    _validate_chain_id(chain_id)
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
    msa_names: list[str] | tuple[str, ...] | None = None,
    skip_templates: bool,
    dry_run: bool,
) -> None:
    resolved_msa_names = normalize_msa_names(msa_name, msa_names)
    failures = []
    for chain_id in chain_ids:
        for url, relative_destination in subset_alignment_urls(
            chain_id,
            resolved_msa_names,
            template_hhr_name,
            skip_templates=skip_templates,
        ):
            if not download_url(url, data_root / relative_destination, dry_run=dry_run):
                failures.append(str(relative_destination))

        structure_url, structure_destination = subset_structure_url(chain_id)
        if not download_url(structure_url, data_root / structure_destination, dry_run=dry_run):
            failures.append(str(structure_destination))
    if failures:
        raise RuntimeError(f"Subset download incomplete: {len(failures)} required assets failed")


def build_alignment_sync_command(
    data_root: Path,
    msa_name: str,
    template_hhr_name: str,
    *,
    msa_names: list[str] | tuple[str, ...] | None = None,
    skip_templates: bool,
    full_alignments: bool,
) -> list[str]:
    destination = str(data_root / "roda_pdb")
    command = ["aws", "s3", "sync", "s3://openfold/pdb/", destination, "--no-sign-request"]
    if full_alignments:
        return command

    command.extend(["--exclude", "*"])
    for name in normalize_msa_names(msa_name, msa_names):
        command.extend(["--include", f"*/a3m/{name}"])
    if not skip_templates:
        command.extend(["--include", f"*/hhr/{template_hhr_name}"])
    return command


def normalize_alignment_layout(
    data_root: Path,
    msa_name: str,
    template_hhr_name: str,
    *,
    msa_names: list[str] | tuple[str, ...] | None = None,
    skip_templates: bool,
) -> None:
    roda_root = data_root / "roda_pdb"
    for chain_dir in sorted(path for path in roda_root.iterdir() if path.is_dir()):
        for name in normalize_msa_names(msa_name, msa_names):
            msa_path = chain_dir / name
            if msa_path.exists():
                target_dir = chain_dir / "a3m"
                target_dir.mkdir(exist_ok=True)
                msa_path.rename(target_dir / name)

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
        for chain_id in chain_ids:
            _validate_chain_id(chain_id)

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
            os.symlink(representative.resolve(), target, target_is_directory=True)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    msa_names = normalize_msa_names(args.msa_name, args.msa_names)
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
            msa_names=msa_names,
            skip_templates=args.skip_templates,
            dry_run=args.dry_run,
        )
        return

    run_command(
        build_alignment_sync_command(
            data_root,
            args.msa_name,
            args.template_hhr_name,
            msa_names=msa_names,
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
            msa_names=msa_names,
            skip_templates=args.skip_templates,
        )
        expand_duplicate_alignments(
            data_root / "roda_pdb",
            data_root / "pdb_data" / "duplicate_pdb_chains.txt",
        )


if __name__ == "__main__":
    main()
