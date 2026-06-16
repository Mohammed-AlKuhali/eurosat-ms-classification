"""Compact subjective gallery for the report: 5 representative patches, each shown
as true-colour, false-colour (NIR), NDVI and NDWI. Lands at a clean landscape
aspect (5 rows x 4 cols) instead of the full pre-registered set.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eurosat_ms.data import default_data_root, load_manifest, load_patch
from eurosat_ms.visualize import true_color, false_color_nir, index_map

# One patch per illustrative class (water, impervious, crop), matching the
# Grad-CAM figure so the index views and the attention maps show the same scenes.
WANT = ["River", "Highway", "PermanentCrop"]
COLS = [("true colour", lambda im: true_color(im)),
        ("false colour (NIR)", lambda im: false_color_nir(im)),
        ("NDVI", lambda im: index_map(im, "NDVI")),
        ("NDWI", lambda im: index_map(im, "NDWI"))]


def main(seed: int = 3) -> None:
    root = default_data_root()
    test = load_manifest("data/manifests", "test")
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = []
    for cls in WANT:
        pool = test[test["class_name"] == cls]["path"].tolist()
        rows.append((cls, pool[rng.integers(len(pool))]))

    fig, axes = plt.subplots(len(rows), len(COLS), figsize=(7.2, 1.7 * len(rows)))
    for r, (cls, rel) in enumerate(rows):
        img = load_patch(root / rel)
        for c, (title, fn) in enumerate(COLS):
            ax = axes[r, c]
            out = fn(img)
            ax.imshow(out, cmap=None if out.ndim == 3 else "BrBG_r",
                      vmin=None if out.ndim == 3 else -1, vmax=None if out.ndim == 3 else 1)
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(cls, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = Path("results/figures/report_gallery.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
