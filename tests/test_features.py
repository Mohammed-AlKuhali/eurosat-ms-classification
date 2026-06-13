"""Spectral-index correctness and numerical-safety guards."""

from __future__ import annotations

import numpy as np

from eurosat_ms import bands
from eurosat_ms.features import (
    INDEX_DEFS,
    compute_indices,
    normalized_difference,
)


def _synthetic(nir: float, red: float, green: float = 500.0, swir: float = 500.0) -> np.ndarray:
    """A constant 13-band patch with controllable NIR/red/green/SWIR."""
    img = np.full((13, 8, 8), 500.0, dtype=np.float32)
    img[bands.BAND_INDEX["B08"]] = nir
    img[bands.BAND_INDEX["B04"]] = red
    img[bands.BAND_INDEX["B03"]] = green
    img[bands.BAND_INDEX["B11"]] = swir
    img[bands.BAND_INDEX["B05"]] = red  # red-edge ~ red for the synthetic
    return img


def test_ndvi_positive_for_vegetation_signature():
    veg = _synthetic(nir=3000, red=400)
    ndvi = normalized_difference(veg, "B08", "B04")
    assert ndvi.mean() > 0.5


def test_ndvi_negative_for_water_signature():
    water = _synthetic(nir=200, red=600)
    ndvi = normalized_difference(water, "B08", "B04")
    assert ndvi.mean() < 0


def test_indices_bounded_and_finite():
    img = _synthetic(nir=3000, red=400)
    stacked = compute_indices(img, list(INDEX_DEFS))
    assert stacked.shape == (len(INDEX_DEFS), 8, 8)
    assert np.isfinite(stacked).all()
    assert stacked.min() >= -1.0 and stacked.max() <= 1.0


def test_division_by_zero_is_guarded():
    """A band pair that sums to zero everywhere must not produce NaN/inf."""
    img = np.zeros((13, 4, 4), dtype=np.float32)
    out = normalized_difference(img, "B08", "B04")
    assert np.isfinite(out).all()


def test_scale_invariance_of_normalized_difference():
    """DN and reflectance (DN/10000) give identical index values."""
    img = _synthetic(nir=3000, red=400)
    a = normalized_difference(img, "B08", "B04")
    b = normalized_difference(img / 10000.0, "B08", "B04")
    assert np.allclose(a, b, atol=1e-5)


def test_real_forest_is_more_vegetated_than_sealake(forest_patch, sealake_patch):
    """NDVI should be substantially higher on Forest than on Sea/Lake — a real-data
    sanity check that index computation and band order agree with physics."""
    ndvi_forest = normalized_difference(forest_patch, "B08", "B04").mean()
    ndvi_water = normalized_difference(sealake_patch, "B08", "B04").mean()
    assert ndvi_forest > ndvi_water
