"""Unsupervised water/vegetation segmentation via spectral-index thresholding.

This is the time-boxed extension. EuroSAT has no pixel-level ground truth, so we
hand-label a small test set and evaluate index-thresholding methods against it.
The point is not a high absolute score (a river can be ~3 px wide at 10 m/px) but
to demonstrate, with a metric, that bands beyond RGB (NIR, SWIR) localise water:
NDWI (B03,B08) and MNDWI (B03,B11) cannot be computed from RGB at all.

Methods: fixed threshold (NDWI/MNDWI > 0) and Otsu (data-driven threshold).
Metrics: per-patch IoU/Dice plus a pixel-pooled micro-average across patches,
which is the honest aggregate when foreground area varies wildly between patches.

References: McFeeters (1996); Xu (2006, MNDWI); Otsu (1979).
"""

from __future__ import annotations

import numpy as np

from .features import normalized_difference

EPS = 1e-9


def water_index(img: np.ndarray, kind: str = "NDWI") -> np.ndarray:
    """NDWI (B03,B08) or MNDWI (B03,B11), positive over open water."""
    if kind == "NDWI":
        return normalized_difference(img, "B03", "B08")
    if kind == "MNDWI":
        return normalized_difference(img, "B03", "B11")
    if kind == "NDVI":
        return normalized_difference(img, "B08", "B04")
    raise ValueError(f"unknown index {kind}")


def fixed_mask(index_arr: np.ndarray, thr: float = 0.0) -> np.ndarray:
    """Boolean mask where index exceeds a fixed threshold."""
    return index_arr > thr


def otsu_mask(index_arr: np.ndarray) -> np.ndarray:
    """Boolean mask via Otsu's data-driven threshold on the index histogram.

    Otsu assumes a bimodal histogram; for an all-water or all-land patch it can
    pick a meaningless split, so we guard by falling back to the fixed 0 cut when
    the index has near-zero spread.
    """
    from skimage.filters import threshold_otsu

    x = index_arr.astype(np.float64)
    if x.max() - x.min() < 1e-3:
        return x > 0.0
    return x > float(threshold_otsu(x))


def iou(pred: np.ndarray, true: np.ndarray) -> float:
    pred, true = pred.astype(bool), true.astype(bool)
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return float(inter / (union + EPS)) if union else 1.0  # both empty -> perfect


def dice(pred: np.ndarray, true: np.ndarray) -> float:
    pred, true = pred.astype(bool), true.astype(bool)
    inter = np.logical_and(pred, true).sum()
    denom = pred.sum() + true.sum()
    return float(2 * inter / (denom + EPS)) if denom else 1.0


def evaluate_masks(preds: list[np.ndarray], trues: list[np.ndarray]) -> dict:
    """Per-patch IoU/Dice (with spread) plus pixel-pooled micro-averaged IoU/Dice."""
    per_iou = [iou(p, t) for p, t in zip(preds, trues)]
    per_dice = [dice(p, t) for p, t in zip(preds, trues)]
    tp = sum(int(np.logical_and(p, t).sum()) for p, t in zip(preds, trues))
    pred_pos = sum(int(p.astype(bool).sum()) for p in preds)
    true_pos = sum(int(t.astype(bool).sum()) for t in trues)
    union = sum(int(np.logical_or(p, t).sum()) for p, t in zip(preds, trues))
    return {
        "n_patches": len(preds),
        "per_patch_iou_mean": float(np.mean(per_iou)),
        "per_patch_iou_std": float(np.std(per_iou)),
        "per_patch_dice_mean": float(np.mean(per_dice)),
        "micro_iou": float(tp / (union + EPS)),
        "micro_dice": float(2 * tp / (pred_pos + true_pos + EPS)),
    }
