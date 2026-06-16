"""Tier-1 classical baseline: Random Forest on hand-crafted spectral features.

This tier exists to (a) give an interpretable, fast reference point before any
deep learning, and (b) quantify *which* bands and indices carry the land-use
signal, via permutation importance. It directly addresses the brief's "feature
extraction" requirement and its emphasis on understanding band information.

Two arms:
  C1, RGB only:    per-band statistics of B04/B03/B02.
  C2, multispectral: per-band statistics of the 12 bands (B10 excluded) plus
       statistics of the NDVI/NDWI/NDBI/NDRE indices.

Features are cached to disk so repeated runs are fast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .bands import BANDS_12, RGB_BANDS
from .data import load_patch
from .features import (
    E3_INDICES,
    band_statistics,
    band_statistics_feature_names,
    index_statistics,
    index_statistics_feature_names,
)

# Feature specs per arm: (band list, index list).
ARM_FEATURES: dict[str, tuple[list[str], list[str]]] = {
    "C1_rgb": (RGB_BANDS, []),
    "C2_multispectral": (BANDS_12, E3_INDICES),
}


def feature_names(arm: str) -> list[str]:
    band_list, index_list = ARM_FEATURES[arm]
    return band_statistics_feature_names(band_list) + index_statistics_feature_names(index_list)


def extract_features(
    data_root: str | Path,
    manifest: pd.DataFrame,
    arm: str,
    cache_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for an arm over a manifest, with optional npz caching."""
    if cache_path is not None and Path(cache_path).exists():
        d = np.load(cache_path)
        return d["X"], d["y"]

    band_list, index_list = ARM_FEATURES[arm]
    data_root = Path(data_root)
    feats, labels = [], []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc=f"features[{arm}]"):
        img = load_patch(data_root / row["path"])
        vec = band_statistics(img, band_list)
        if index_list:
            vec = np.concatenate([vec, index_statistics(img, index_list)])
        feats.append(vec)
        labels.append(int(row["label"]))
    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def train_random_forest(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """A standard RF classifier; features are tree-based so no scaling needed."""
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced",  # mild EuroSAT imbalance
    )
    clf.fit(X, y)
    return clf


def permutation_importance_df(clf, X_test, y_test, names, seed=42, n_repeats=10) -> pd.DataFrame:
    """Permutation importance on the test split, sorted descending."""
    from sklearn.inspection import permutation_importance

    r = permutation_importance(
        clf, X_test, y_test, n_repeats=n_repeats, random_state=seed, n_jobs=-1
    )
    return (
        pd.DataFrame({
            "feature": names,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
