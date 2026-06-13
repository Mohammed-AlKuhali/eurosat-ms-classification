"""Spectral indices and classical (Tier-1) feature extraction.

Indices are normalised-difference ratios computed directly from the raw
uint16 digital numbers (DN). Because EuroSAT stores L1C top-of-atmosphere
reflectance x10000, the constant scale factor cancels in a normalised
difference, so DN and reflectance give identical index values. EuroSAT
predates the Jan-2022 baseline-04.00 radiometric offset, so NO -1000 offset
is applied.

All indices are computed in float32, guarded against division-by-zero with a
small epsilon, and clipped to [-1, 1]. Output is asserted finite.
"""

from __future__ import annotations

import numpy as np

from .bands import BAND_INDEX

EPS = 1e-6

# Normalised-difference index definitions: name -> (numerator_band, denominator_band)
# NDI(a, b) = (a - b) / (a + b)
INDEX_DEFS: dict[str, tuple[str, str]] = {
    "NDVI": ("B08", "B04"),   # vegetation density
    "NDWI": ("B03", "B08"),   # open water (McFeeters 1996)
    "NDBI": ("B11", "B08"),   # built-up
    "NDRE": ("B08", "B05"),   # red-edge; crop/vegetation type
    "MNDWI": ("B03", "B11"),  # modified water index (Xu 2006); optional
}

# The engineered quartet appended to RGB in arm E3.
E3_INDICES: list[str] = ["NDVI", "NDWI", "NDBI", "NDRE"]


def normalized_difference(img: np.ndarray, num_band: str, den_band: str) -> np.ndarray:
    """Compute (num - den) / (num + den) for two named bands.

    Parameters
    ----------
    img : np.ndarray
        Image array shaped (C, H, W) in file band order.
    num_band, den_band : str
        Band names (e.g. "B08", "B04").

    Returns
    -------
    np.ndarray
        float32 array (H, W), values in [-1, 1], guaranteed finite.
    """
    a = img[BAND_INDEX[num_band]].astype(np.float32)
    b = img[BAND_INDEX[den_band]].astype(np.float32)
    denom = a + b
    out = (a - b) / np.where(np.abs(denom) < EPS, EPS, denom)
    out = np.clip(out, -1.0, 1.0).astype(np.float32)
    assert np.isfinite(out).all(), f"non-finite values in {num_band}-{den_band} index"
    return out


def compute_indices(img: np.ndarray, names: list[str]) -> np.ndarray:
    """Stack the named indices into a (len(names), H, W) float32 array."""
    return np.stack([
        normalized_difference(img, *INDEX_DEFS[n]) for n in names
    ], axis=0).astype(np.float32)


def band_statistics(img: np.ndarray, band_names: list[str]) -> np.ndarray:
    """Classical per-band summary statistics for a single patch (Tier-1 features).

    For each requested band, computes mean, std, and the 10/50/90 percentiles
    (5 features/band), capturing the spectral distribution without spatial
    structure. Operates on raw DN.

    Parameters
    ----------
    img : np.ndarray
        (C, H, W) image in file band order.
    band_names : list[str]
        Bands to summarise.

    Returns
    -------
    np.ndarray
        1-D float32 feature vector of length 5 * len(band_names).
    """
    feats: list[float] = []
    for name in band_names:
        x = img[BAND_INDEX[name]].astype(np.float32).ravel()
        p10, p50, p90 = np.percentile(x, [10, 50, 90])
        feats.extend([float(x.mean()), float(x.std()), float(p10), float(p50), float(p90)])
    return np.asarray(feats, dtype=np.float32)


def index_statistics(img: np.ndarray, names: list[str]) -> np.ndarray:
    """Mean and std of each named spectral index (2 features/index)."""
    feats: list[float] = []
    for n in names:
        idx = normalized_difference(img, *INDEX_DEFS[n])
        feats.extend([float(idx.mean()), float(idx.std())])
    return np.asarray(feats, dtype=np.float32)


def band_statistics_feature_names(band_names: list[str]) -> list[str]:
    """Human-readable names for :func:`band_statistics`, for permutation importance."""
    stats = ["mean", "std", "p10", "p50", "p90"]
    return [f"{b}_{s}" for b in band_names for s in stats]


def index_statistics_feature_names(names: list[str]) -> list[str]:
    """Human-readable names for :func:`index_statistics`."""
    return [f"{n}_{s}" for n in names for s in ("mean", "std")]
