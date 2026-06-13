"""Evaluate index-thresholding water segmentation against the hand-labelled set.

Compares NDWI vs MNDWI and fixed-threshold vs Otsu, reporting per-patch and
pixel-pooled micro-averaged IoU/Dice, and saves a qualitative panel. Run after
labelling masks with scripts/label_masks.py.

Usage
-----
    python scripts/run_segmentation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eurosat_ms.data import default_data_root, load_patch
from eurosat_ms.segmentation import evaluate_masks, fixed_mask, otsu_mask, water_index
from eurosat_ms.visualize import true_color, _save

METHODS = {
    "NDWI_fixed0": ("NDWI", lambda x: fixed_mask(x, 0.0)),
    "NDWI_otsu": ("NDWI", otsu_mask),
    "MNDWI_fixed0": ("MNDWI", lambda x: fixed_mask(x, 0.0)),
    "MNDWI_otsu": ("MNDWI", otsu_mask),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-dir", default="data/seg_labels")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    manifest = json.loads((labels_dir / "labels.json").read_text())
    data_root = Path(args.data_root) if args.data_root else default_data_root()
    items = list(manifest["labelled"].items())
    if not items:
        raise SystemExit("No labelled masks found. Run scripts/label_masks.py first.")
    print(f"Evaluating on {len(items)} hand-labelled patches.")

    imgs = [load_patch(data_root / rel) for rel, _ in items]
    trues = [np.load(labels_dir / info["mask"]).astype(bool) for _, info in items]

    rows = []
    preds_by_method = {}
    for name, (index_kind, fn) in METHODS.items():
        preds = [fn(water_index(im, index_kind)) for im in imgs]
        preds_by_method[name] = preds
        m = evaluate_masks(preds, trues)
        m["method"] = name
        rows.append(m)
        print(f"  {name:14s} micro-IoU={m['micro_iou']:.3f}  micro-Dice={m['micro_dice']:.3f}  "
              f"per-patch IoU={m['per_patch_iou_mean']:.3f}±{m['per_patch_iou_std']:.3f}")

    tables = Path(args.results_dir, "tables"); tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tables / "segmentation.csv", index=False)

    # Qualitative panel for the best method: image | true | pred (first 5 patches).
    best = max(rows, key=lambda r: r["micro_iou"])["method"]
    k = min(5, len(items))
    fig, axes = plt.subplots(k, 3, figsize=(8, 2.6 * k))
    if k == 1:
        axes = axes[None, :]
    for i in range(k):
        axes[i, 0].imshow(true_color(imgs[i])); axes[i, 0].set_title("image" if i == 0 else "", fontsize=9)
        axes[i, 1].imshow(trues[i], cmap="Blues"); axes[i, 1].set_title("hand label" if i == 0 else "", fontsize=9)
        axes[i, 2].imshow(preds_by_method[best][i], cmap="Blues"); axes[i, 2].set_title(f"{best}" if i == 0 else "", fontsize=9)
        for a in axes[i]:
            a.axis("off")
    fig.suptitle(f"Water segmentation — best method: {best}", y=1.0)
    _save(fig, Path(args.results_dir, "figures", "segmentation_panel.png"))
    print(f"\nBest by micro-IoU: {best}. Tables -> results/tables/segmentation.csv")


if __name__ == "__main__":
    main()
