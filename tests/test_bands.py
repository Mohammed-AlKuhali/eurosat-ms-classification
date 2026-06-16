"""Band-order guards, the highest-impact silent-failure surface."""

from __future__ import annotations

import numpy as np

from eurosat_ms import bands


def test_band_order_internal_consistency():
    bands.validate()


def test_b8a_is_last_and_b10_is_index_9():
    assert bands.BAND_ORDER[-1] == "B8A"
    assert bands.BAND_INDEX["B8A"] == 12
    assert bands.BAND_INDEX["B10"] == 9


def test_ssl4eo_reorder_is_permutation_with_b8a_ninth():
    assert sorted(bands.SSL4EO_REORDER) == list(range(13))
    # SSL4EO order places B8A in the 9th position (index 8).
    assert bands.SSL4EO_ORDER[8] == "B8A"


def test_twelve_band_view_excludes_cirrus():
    assert "B10" not in bands.BANDS_12
    assert len(bands.BANDS_12) == 12


def test_b10_is_near_empty_on_real_data(data_root, forest_patch):
    """Verifies BOTH that B10 is near-empty AND that the file band order is correct:
    if B8A (NIR, ~3000) were at channel 9 instead of B10, this would fail."""
    b10 = forest_patch[bands.BAND_INDEX["B10"]].astype(np.float32)
    assert b10.mean() < 100, f"B10 mean {b10.mean():.1f} too high, band order is wrong"


def test_nir_exceeds_red_on_forest(forest_patch):
    """Vegetation reflects strongly in NIR and absorbs red, a semantic band check."""
    nir = forest_patch[bands.BAND_INDEX["B08"]].astype(np.float32).mean()
    red = forest_patch[bands.BAND_INDEX["B04"]].astype(np.float32).mean()
    assert nir > red, f"Forest NIR {nir:.1f} should exceed red {red:.1f}"
