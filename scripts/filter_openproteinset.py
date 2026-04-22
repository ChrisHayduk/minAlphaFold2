"""Apply AlphaFold2 supplement §1.2.5 training-data filters to OpenProteinSet.

The preprocessed caches produced by ``preprocess_openproteinset.py`` are the
input (labels NPZs carry ``resolution``, feature NPZs carry ``aatype``).
This script applies the deterministic §1.2.5 filters on top and emits a
JSON manifest listing the chain IDs that pass.

Deterministic filters implemented (§1.2.5):

1. **Resolution** — reject chains whose mmCIF resolution is ≥ 9 Å
   (removes ~0.2 % of structures per the supplement). Read directly
   from ``labels[<chain>]["resolution"]``.
2. **Single-AA dominance** — reject chains where any single amino
   acid accounts for > 80 % of the primary sequence (removes ~0.8 %
   per the supplement). Computed from ``aatype``.
3. **Minimum length** — reject chains shorter than ``--min-length``
   residues (default 10). Not in the paper strictly but guards against
   degenerate NPZs.

Probabilistic filters from §1.2.5 that are **not** applied here — they
belong in the dataloader sampler, not a pre-filter:

- Length rebalancing: ``p = clamp(N_res, 256, 512) / 512``.
- MMseqs2 40 %-identity cluster inverse sampling: ``p ∝ 1 / cluster_size``.

If ``--mmseqs-cluster-tsv`` is given, the manifest additionally includes
per-chain ``cluster_id`` and ``cluster_size`` so the dataloader can
implement the cluster-inverse-size sampler. Generate the TSV yourself
with ``mmseqs easy-cluster <fasta> <cluster_out> tmp --min-seq-id 0.4``
on the accepted chains; this script does not invoke MMseqs2 directly
to keep the dependency optional.

Usage::

    python scripts/filter_openproteinset.py \\
      --processed-features-dir data/processed_features \\
      --processed-labels-dir   data/processed_labels \\
      --manifest-out            data/filter_manifest.json

The produced manifest is compatible with
:class:`minalphafold.data.ProcessedOpenProteinSetDataset` via the
``chains_manifest`` option in :class:`~minalphafold.trainer.DataConfig`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minalphafold.data import discover_chain_ids


DEFAULT_MAX_RESOLUTION_ANGSTROMS = 9.0  # §1.2.5 bullet 1
DEFAULT_MAX_SINGLE_AA_FRACTION = 0.8    # §1.2.5 bullet 3
DEFAULT_MIN_LENGTH = 10                 # guard against degenerate chains


def max_single_aa_fraction(aatype: np.ndarray) -> float:
    """Fraction of the primary sequence taken by its most-frequent AA."""
    if aatype.size == 0:
        return 1.0  # treat empty as "100% one AA" so it's rejected
    counts = Counter(int(value) for value in aatype.tolist())
    most_common_count = max(counts.values())
    return most_common_count / float(aatype.size)


def load_cluster_tsv(path: Path) -> dict[str, tuple[str, int]]:
    """Parse an MMseqs2 easy-cluster TSV into ``{chain_id: (cluster_id, size)}``.

    The ``easy-cluster`` output has two columns (cluster_representative,
    member). Cluster size = number of members sharing a representative.
    The chain IDs in the TSV must match the preprocessed NPZ stems (both
    ``<pdb_id>_<chain>``).
    """
    members: dict[str, str] = {}
    cluster_sizes: Counter[str] = Counter()
    with path.open() as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            representative, member = parts[0], parts[1]
            members[member] = representative
            cluster_sizes[representative] += 1
    return {
        member: (representative, cluster_sizes[representative])
        for member, representative in members.items()
    }


def _evaluate_chain(
    chain_id: str,
    processed_features_dir: Path,
    processed_labels_dir: Path,
    *,
    max_resolution: float,
    max_single_aa_fraction_threshold: float,
    min_length: int,
    cluster_info: dict[str, tuple[str, int]],
) -> dict[str, Any]:
    """Run §1.2.5 filters on a single chain; return a manifest entry."""
    features_path = processed_features_dir / f"{chain_id}.npz"
    labels_path = processed_labels_dir / f"{chain_id}.npz"

    features = np.load(features_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)

    aatype = np.asarray(features["aatype"])
    length = int(aatype.shape[0])
    resolution = (
        float(labels["resolution"].item())
        if "resolution" in labels.files
        else float("nan")
    )
    single_aa_fraction = max_single_aa_fraction(aatype)

    reject_reasons: list[str] = []
    if length < min_length:
        reject_reasons.append("min_length")
    if not np.isnan(resolution) and resolution >= max_resolution:
        reject_reasons.append("resolution")
    if single_aa_fraction > max_single_aa_fraction_threshold:
        reject_reasons.append("single_aa")

    entry: dict[str, Any] = {
        "chain_id": chain_id,
        "length": length,
        "resolution": resolution,
        "max_single_aa_fraction": round(single_aa_fraction, 4),
        "accepted": not reject_reasons,
        "reject_reasons": reject_reasons,
    }
    if chain_id in cluster_info:
        cluster_id, cluster_size = cluster_info[chain_id]
        entry["cluster_id"] = cluster_id
        entry["cluster_size"] = cluster_size
    return entry


def build_manifest(
    *,
    processed_features_dir: Path,
    processed_labels_dir: Path,
    max_resolution: float,
    max_single_aa_fraction_threshold: float,
    min_length: int,
    mmseqs_cluster_tsv: Path | None,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Scan all NPZs under the processed directories and return a manifest dict."""
    chain_ids = discover_chain_ids(processed_features_dir, processed_labels_dir)
    if sample_limit is not None:
        chain_ids = chain_ids[:sample_limit]

    cluster_info: dict[str, tuple[str, int]] = {}
    if mmseqs_cluster_tsv is not None:
        cluster_info = load_cluster_tsv(mmseqs_cluster_tsv)

    entries = [
        _evaluate_chain(
            chain_id,
            processed_features_dir,
            processed_labels_dir,
            max_resolution=max_resolution,
            max_single_aa_fraction_threshold=max_single_aa_fraction_threshold,
            min_length=min_length,
            cluster_info=cluster_info,
        )
        for chain_id in chain_ids
    ]

    summary: dict[str, int] = {
        "total": len(entries),
        "accepted": sum(1 for entry in entries if entry["accepted"]),
    }
    for reason in ("min_length", "resolution", "single_aa"):
        summary[f"rejected_{reason}"] = sum(
            1 for entry in entries if reason in entry["reject_reasons"]
        )

    return {
        "config": {
            "max_resolution_angstroms": max_resolution,
            "max_single_aa_fraction": max_single_aa_fraction_threshold,
            "min_length": min_length,
            "mmseqs_cluster_tsv": str(mmseqs_cluster_tsv) if mmseqs_cluster_tsv else None,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_features_dir": str(processed_features_dir),
        "source_labels_dir": str(processed_labels_dir),
        "summary": summary,
        "chains": entries,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply supplement §1.2.5 training-data filters to "
                    "preprocessed OpenProteinSet chains and emit a JSON manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--processed-features-dir", type=Path, default=Path("data/processed_features"))
    parser.add_argument("--processed-labels-dir", type=Path, default=Path("data/processed_labels"))
    parser.add_argument("--manifest-out", type=Path, required=True,
                        help="Destination path for the filter manifest (JSON).")
    parser.add_argument(
        "--max-resolution-angstroms", type=float,
        default=DEFAULT_MAX_RESOLUTION_ANGSTROMS,
        help="Reject chains with resolution ≥ this (§1.2.5: 9 Å).",
    )
    parser.add_argument(
        "--max-single-aa-fraction", type=float,
        default=DEFAULT_MAX_SINGLE_AA_FRACTION,
        help="Reject chains where a single amino acid covers > this fraction "
             "of the primary sequence (§1.2.5: 0.8).",
    )
    parser.add_argument("--min-length", type=int, default=DEFAULT_MIN_LENGTH)
    parser.add_argument(
        "--mmseqs-cluster-tsv", type=Path, default=None,
        help="Optional MMseqs2 easy-cluster TSV (columns: representative, "
             "member). If given, manifest entries include cluster_id + "
             "cluster_size so the dataloader can apply §1.2.5's "
             "inverse-cluster-size sampling.",
    )
    parser.add_argument("--sample-limit", type=int, default=None,
                        help="Only process the first N chains (for smoke tests).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_manifest(
        processed_features_dir=args.processed_features_dir,
        processed_labels_dir=args.processed_labels_dir,
        max_resolution=args.max_resolution_angstroms,
        max_single_aa_fraction_threshold=args.max_single_aa_fraction,
        min_length=args.min_length,
        mmseqs_cluster_tsv=args.mmseqs_cluster_tsv,
        sample_limit=args.sample_limit,
    )
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2))

    summary = manifest["summary"]
    print(
        f"[filter] total={summary['total']} accepted={summary['accepted']} "
        f"(resolution_drop={summary['rejected_resolution']}, "
        f"single_aa_drop={summary['rejected_single_aa']}, "
        f"min_length_drop={summary['rejected_min_length']})"
    )
    print(f"[filter] manifest → {args.manifest_out}")


if __name__ == "__main__":
    main()
