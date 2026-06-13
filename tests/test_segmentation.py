"""IoU/Dice correctness and index-thresholding sanity."""

from __future__ import annotations

import numpy as np

from eurosat_ms import bands
from eurosat_ms.segmentation import dice, evaluate_masks, fixed_mask, iou, otsu_mask, water_index


def test_iou_dice_known_overlap():
    true = np.zeros((4, 4), bool); true[:2, :] = True   # 8 px
    pred = np.zeros((4, 4), bool); pred[:1, :] = True    # 4 px, all inside true
    # intersection 4, union 8 -> IoU 0.5 ; dice 2*4/(4+8)=0.667
    assert abs(iou(pred, true) - 0.5) < 1e-6
    assert abs(dice(pred, true) - 2 / 3) < 1e-6


def test_both_empty_is_perfect():
    z = np.zeros((4, 4), bool)
    assert iou(z, z) == 1.0 and dice(z, z) == 1.0


def test_micro_average_pools_pixels():
    # patch1 perfect, patch2 disjoint -> pooled differs from per-patch mean
    t1 = np.ones((2, 2), bool); p1 = np.ones((2, 2), bool)
    t2 = np.zeros((2, 2), bool); t2[0] = True
    p2 = np.zeros((2, 2), bool); p2[1] = True   # disjoint from t2
    m = evaluate_masks([p1, p2], [t1, t2])
    assert m["n_patches"] == 2
    # micro IoU = (4+0)/(4+4) = 0.5
    assert abs(m["micro_iou"] - 0.5) < 1e-6


def test_water_index_high_on_synthetic_water():
    # water: high green (B03), low NIR (B08) -> NDWI positive
    img = np.full((13, 8, 8), 500.0, dtype=np.float32)
    img[bands.BAND_INDEX["B03"]] = 1500
    img[bands.BAND_INDEX["B08"]] = 200
    ndwi = water_index(img, "NDWI")
    assert ndwi.mean() > 0
    assert fixed_mask(ndwi, 0.0).mean() == 1.0  # all flagged water


def test_otsu_guard_on_flat_index():
    flat = np.full((8, 8), 0.3, dtype=np.float32)
    # near-constant index -> guard returns x>0 (all True here), no crash
    assert otsu_mask(flat).all()
