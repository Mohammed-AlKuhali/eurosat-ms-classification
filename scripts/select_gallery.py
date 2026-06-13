"""Pre-register the subjective-assessment sample set BEFORE any results exist.

Selecting the gallery image IDs with a fixed seed and committing them means the
qualitative comparison cannot later be attacked as cherry-picked. We sample two
patches per class plus extra draws from the documented confusion pairs (where
the RGB-vs-multispectral difference is most informative).

Run once; results/subjective_sample_ids.json is committed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eurosat_ms.data import CLASS_TO_IDX, load_manifest

# Documented EuroSAT confusion pairs to over-sample for the gallery.
CONFUSION_PAIRS = [
    ("River", "Highway"),
    ("AnnualCrop", "PermanentCrop"),
    ("PermanentCrop", "HerbaceousVegetation"),
    ("Pasture", "HerbaceousVegetation"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--out", default="results/subjective_sample_ids.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--per-pair", type=int, default=2)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    test = load_manifest(args.manifest_dir, "test")

    def sample(class_name, k):
        pool = test[test["class_name"] == class_name]["path"].tolist()
        idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return [pool[i] for i in idx]

    selected: dict[str, list[str]] = {}
    for cls in CLASS_TO_IDX:
        selected[cls] = sample(cls, args.per_class)
    # Extra draws from each side of the confusion pairs.
    pair_extra: dict[str, list[str]] = {}
    for a, b in CONFUSION_PAIRS:
        for cls in (a, b):
            pair_extra.setdefault(cls, [])
            existing = set(selected.get(cls, [])) | set(pair_extra[cls])
            pool = [p for p in test[test["class_name"] == cls]["path"] if p not in existing]
            extra = [pool[i] for i in rng.choice(len(pool), size=min(args.per_pair, len(pool)), replace=False)]
            pair_extra[cls].extend(extra)

    payload = {
        "seed": args.seed,
        "per_class": selected,
        "confusion_pair_extra": pair_extra,
        "confusion_pairs": CONFUSION_PAIRS,
        "note": "Pre-registered before any model results existed (cherry-pick guard).",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    total = sum(len(v) for v in selected.values()) + sum(len(v) for v in pair_extra.values())
    print(f"wrote {out}: {total} sample patches "
          f"({sum(len(v) for v in selected.values())} per-class + "
          f"{sum(len(v) for v in pair_extra.values())} confusion-pair extras)")


if __name__ == "__main__":
    main()
