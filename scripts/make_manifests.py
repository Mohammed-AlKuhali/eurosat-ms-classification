"""Generate the committed stratified split manifests and train normalisation stats.

Run once; the resulting CSVs and JSON under data/manifests and data/stats are
committed so every experiment arm uses an identical, reproducible split.

Usage
-----
    python scripts/make_manifests.py                      # uses default data root
    EUROSAT_DATA_ROOT=/path/to/EuroSAT_MS python scripts/make_manifests.py
    python scripts/make_manifests.py --skip-stats         # manifests only (fast)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eurosat_ms.data import (
    compute_train_stats,
    default_data_root,
    generate_manifests,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--stats-dir", default="data/stats")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-stats", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root) if args.data_root else default_data_root()
    print(f"data root: {data_root}")

    splits = generate_manifests(data_root, args.manifest_dir, seed=args.seed)
    for name, frame in splits.items():
        per_class = frame["class_name"].value_counts().to_dict()
        print(f"  {name:5s}: {len(frame):6d}  (per-class min={min(per_class.values())}, "
              f"max={max(per_class.values())})")

    if not args.skip_stats:
        print("computing per-band train statistics (one streaming pass)...")
        stats_dir = Path(args.stats_dir)
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats = compute_train_stats(
            data_root, splits["train"], out_path=stats_dir / "train_stats.json"
        )
        print(f"  B10 mean={stats['band_mean']['B10']:.1f} (near-empty check), "
              f"B08 mean={stats['band_mean']['B08']:.1f}")
        print(f"  wrote {stats_dir / 'train_stats.json'}")


if __name__ == "__main__":
    main()
