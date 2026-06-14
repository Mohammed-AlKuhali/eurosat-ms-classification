"""Per-class mean spectral signatures across the 13 Sentinel-2 bands.

Supports the report's 'understanding of band information': shows how the classes
separate across wavelength, why non-RGB bands (NIR/red-edge/SWIR) carry signal,
and why B10 (cirrus) is near-empty. Bands are ordered by central wavelength.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eurosat_ms.bands import BAND_INDEX
from eurosat_ms.data import CLASS_NAMES, default_data_root, load_manifest, load_patch

# Sentinel-2 central wavelengths (nm) for ordering the x-axis.
WAVELENGTH = {
    "B01": 443, "B02": 490, "B03": 560, "B04": 665, "B05": 705, "B06": 740,
    "B07": 783, "B08": 842, "B8A": 865, "B09": 945, "B10": 1375, "B11": 1610, "B12": 2190,
}
ORDER = sorted(WAVELENGTH, key=WAVELENGTH.get)


def main(n_per_class: int = 150, seed: int = 0) -> None:
    root = default_data_root()
    train = load_manifest("data/manifests", "train")
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    cmap = plt.get_cmap("tab10")
    for ci, cls in enumerate(CLASS_NAMES):
        pool = train[train["class_name"] == cls]["path"].tolist()
        pick = [pool[i] for i in rng.choice(len(pool), size=min(n_per_class, len(pool)), replace=False)]
        means = np.zeros(13)
        for rel in pick:
            img = load_patch(root / rel).astype(np.float64)
            means += img.reshape(13, -1).mean(axis=1)
        means /= len(pick)
        y = [means[BAND_INDEX[b]] for b in ORDER]
        ax.plot(range(len(ORDER)), y, marker="o", ms=3, lw=1.3, color=cmap(ci), label=cls)

    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([f"{b}\n{WAVELENGTH[b]}" for b in ORDER], fontsize=7)
    ax.set_xlabel("Sentinel-2 band (central wavelength, nm)")
    ax.set_ylabel("mean reflectance (DN = reflectance$\\times10^4$)")
    ax.set_title("Per-class spectral signatures (train split)")
    ax.axvspan(ORDER.index("B10") - 0.4, ORDER.index("B10") + 0.4, color="0.9", zorder=0)
    ax.annotate("B10 cirrus\n(near-empty)", xy=(ORDER.index("B10"), 200),
                fontsize=6.5, ha="center", color="0.4")
    ax.legend(fontsize=6.5, ncol=2, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = Path("results/figures/band_signatures.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
