"""Data loading, deterministic splits, and per-band normalisation statistics.

This module is deliberately framework-free (numpy / pandas / scikit-learn
only) so the data layer can be tested without installing PyTorch. The torch
``Dataset`` wrapper that consumes these manifests lives in ``torch_dataset.py``
and is only needed for the CNN arms.

Key guarantees
--------------
* Splits are **stratified** by class and **deterministic** given a seed.
* The brief's "80/20 train/test" is the (train+val) vs test partition; a
  validation slice is carved *out of the 80% train* for early stopping, so the
  test set is exactly 20% and is touched only once.
* Manifests are written as CSVs and committed, so every arm — classical and
  CNN — sees an identical split, which is what makes the paired McNemar test
  valid.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from .bands import BAND_ORDER, N_BANDS

# Class names in sorted (folder-enumeration) order; index == integer label.
CLASS_NAMES: list[str] = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
N_CLASSES = len(CLASS_NAMES)


def default_data_root() -> Path:
    """Resolve the EuroSAT_MS root from $EUROSAT_DATA_ROOT or the sibling download."""
    env = os.environ.get("EUROSAT_DATA_ROOT")
    if env:
        return Path(env)
    # Repo lives next to the Zenodo download folder "7711810/EuroSAT_MS".
    here = Path(__file__).resolve()
    repo = here.parents[2]
    return repo.parent / "7711810" / "EuroSAT_MS"


def load_patch(path: str | Path) -> np.ndarray:
    """Read a EuroSAT_MS GeoTIFF as a (13, H, W) uint16 array in file band order.

    tifffile returns (H, W, 13); we transpose to channels-first to match the
    PyTorch convention used downstream.
    """
    arr = tifffile.imread(str(path))
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D patch, got shape {arr.shape} for {path}")
    if arr.shape[-1] == N_BANDS:          # (H, W, 13) -> (13, H, W)
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.shape[0] != N_BANDS:
        raise ValueError(f"cannot locate 13-band axis in shape {arr.shape} for {path}")
    return np.ascontiguousarray(arr)


def scan_dataset(data_root: str | Path) -> pd.DataFrame:
    """Enumerate every .tif under data_root/<class>/ into a DataFrame.

    Returns columns: path (relative to data_root), class_name, label.
    """
    data_root = Path(data_root)
    rows: list[dict] = []
    for cls in CLASS_NAMES:
        cls_dir = data_root / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"missing class folder: {cls_dir}")
        for tif in sorted(cls_dir.glob("*.tif")):
            rows.append({
                "path": str(tif.relative_to(data_root)),
                "class_name": cls,
                "label": CLASS_TO_IDX[cls],
            })
    if not rows:
        raise FileNotFoundError(f"no .tif files found under {data_root}")
    return pd.DataFrame(rows)


def generate_manifests(
    data_root: str | Path,
    out_dir: str | Path,
    seed: int = 42,
    test_frac: float = 0.20,
    val_frac_of_train: float = 0.10,
) -> dict[str, pd.DataFrame]:
    """Create stratified train/val/test manifests and write them as CSVs.

    The split is: 80/20 (train+val)/test, then ``val_frac_of_train`` of the 80%
    is held out as validation. With the defaults this yields 72% / 8% / 20%.
    """
    from sklearn.model_selection import train_test_split

    df = scan_dataset(data_root)
    trainval, test = train_test_split(
        df, test_size=test_frac, stratify=df["label"], random_state=seed,
    )
    train, val = train_test_split(
        trainval, test_size=val_frac_of_train, stratify=trainval["label"], random_state=seed,
    )
    splits = {
        "train": train.sort_values("path").reset_index(drop=True),
        "val": val.sort_values("path").reset_index(drop=True),
        "test": test.sort_values("path").reset_index(drop=True),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    meta = {
        "seed": seed,
        "test_frac": test_frac,
        "val_frac_of_train": val_frac_of_train,
        "counts": {k: len(v) for k, v in splits.items()},
        "total": len(df),
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2))
    return splits


def load_manifest(manifest_dir: str | Path, split: str) -> pd.DataFrame:
    """Load a committed split manifest ('train' | 'val' | 'test')."""
    return pd.read_csv(Path(manifest_dir) / f"{split}.csv")


def compute_train_stats(
    data_root: str | Path,
    train_manifest: pd.DataFrame,
    out_path: str | Path | None = None,
) -> dict:
    """Per-band and per-index mean/std over the TRAIN split only (leak-free).

    Band stats drive z-score normalisation of raw reflectance channels; index
    stats standardise the engineered NDVI/NDWI/NDBI/NDRE/MNDWI channels so that,
    in arm E3, reflectance and index inputs share a comparable scale and neither
    can be accused of confounding the comparison. Uses streaming sums so memory
    stays flat regardless of train size.
    """
    from .features import INDEX_DEFS, normalized_difference

    data_root = Path(data_root)
    index_names = list(INDEX_DEFS)
    n_pix = 0
    s = np.zeros(N_BANDS, dtype=np.float64)
    ss = np.zeros(N_BANDS, dtype=np.float64)
    si = np.zeros(len(index_names), dtype=np.float64)
    ssi = np.zeros(len(index_names), dtype=np.float64)
    for rel in train_manifest["path"]:
        img = load_patch(data_root / rel)  # (13, H, W) uint16
        flat = img.reshape(N_BANDS, -1).astype(np.float64)
        s += flat.sum(axis=1)
        ss += (flat ** 2).sum(axis=1)
        for j, name in enumerate(index_names):
            idx = normalized_difference(img, *INDEX_DEFS[name]).astype(np.float64).ravel()
            si[j] += idx.sum()
            ssi[j] += (idx ** 2).sum()
        n_pix += flat.shape[1]
    mean = s / n_pix
    std = np.sqrt(np.maximum(ss / n_pix - mean ** 2, 0.0))
    imean = si / n_pix
    istd = np.sqrt(np.maximum(ssi / n_pix - imean ** 2, 0.0))
    stats = {
        "band_mean": {b: float(mean[i]) for i, b in enumerate(BAND_ORDER)},
        "band_std": {b: float(std[i]) for i, b in enumerate(BAND_ORDER)},
        "index_mean": {n: float(imean[j]) for j, n in enumerate(index_names)},
        "index_std": {n: float(istd[j]) for j, n in enumerate(index_names)},
        "n_images": int(len(train_manifest)),
        "n_pixels": int(n_pix),
    }
    if out_path is not None:
        Path(out_path).write_text(json.dumps(stats, indent=2))
    return stats


def load_stats(path: str | Path) -> dict:
    """Load committed normalisation statistics."""
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class Normalizer:
    """Per-band z-score normaliser built from committed train statistics."""

    mean: np.ndarray  # (C,)
    std: np.ndarray   # (C,)

    @classmethod
    def for_bands(cls, stats: dict, band_names: list[str]) -> "Normalizer":
        mean = np.array([stats["band_mean"][b] for b in band_names], dtype=np.float32)
        std = np.array([stats["band_std"][b] for b in band_names], dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean, std=std)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """z-score a (C, H, W) float array using per-channel stats."""
        return (x - self.mean[:, None, None]) / self.std[:, None, None]
