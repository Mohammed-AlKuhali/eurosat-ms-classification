"""Figures: spectral composites, confusion matrices, per-class deltas,
data-efficiency curve, and Grad-CAM. Pure-matplotlib; saves PNGs.

The composite helpers (true-colour, false-colour NIR, NDVI/NDWI maps) double as
the report's evidence of "understanding of band information": the same patch is
shown through the human-visible bands and through the bands a model needs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bands import BAND_INDEX
from .data import CLASS_NAMES
from .features import normalized_difference

TEAL = "#0F6E56"


def _stretch(x: np.ndarray, p_lo=2, p_hi=98) -> np.ndarray:
    """Percentile contrast stretch to [0,1] for display."""
    lo, hi = np.percentile(x, [p_lo, p_hi])
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


def true_color(img: np.ndarray) -> np.ndarray:
    """RGB composite (B04,B03,B02) -> (H,W,3) display image."""
    r, g, b = (img[BAND_INDEX[x]].astype(np.float32) for x in ("B04", "B03", "B02"))
    return np.dstack([_stretch(r), _stretch(g), _stretch(b)])


def false_color_nir(img: np.ndarray) -> np.ndarray:
    """NIR-false-colour (B08,B04,B03): vegetation appears red."""
    r, g, b = (img[BAND_INDEX[x]].astype(np.float32) for x in ("B08", "B04", "B03"))
    return np.dstack([_stretch(r), _stretch(g), _stretch(b)])


def index_map(img: np.ndarray, name: str) -> np.ndarray:
    from .features import INDEX_DEFS
    return normalized_difference(img, *INDEX_DEFS[name])


def plot_confusion(cm: np.ndarray, out_path, title="Confusion matrix", normalize=True):
    m = cm.astype(np.float64)
    if normalize:
        m = m / np.clip(m.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(m, cmap="Greens", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=90, fontsize=8)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); _save(fig, out_path)


def plot_per_class_delta(delta_df, out_path, title="Per-class recall: multispectral - RGB"):
    d = delta_df.sort_values("delta_pp")
    colors = [TEAL if v >= 0 else "#B3261E" for v in d["delta_pp"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(d.index, d["delta_pp"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("recall delta (percentage points)"); ax.set_title(title)
    fig.tight_layout(); _save(fig, out_path)


def plot_data_efficiency(curve_df, out_path):
    """curve_df: columns arm_base, fraction, acc_mean, acc_std (RGB vs MS)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for arm, grp in curve_df.groupby("arm_base"):
        g = grp.sort_values("fraction")
        ax.errorbar(g["fraction"] * 100, g["acc_mean"] * 100, yerr=g["acc_std"] * 100,
                    marker="o", capsize=3, label=arm)
    ax.set_xscale("log")
    ax.set_xlabel("training labels used (%)"); ax.set_ylabel("test accuracy (%)")
    ax.set_title("Data-efficiency: RGB vs multispectral"); ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout(); _save(fig, out_path)


def gallery_row(img: np.ndarray, fig=None, axes=None, label: str = ""):
    """Render one patch as [true-colour | false-colour NIR | NDVI | NDWI]."""
    if axes is None:
        fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    axes[0].imshow(true_color(img)); axes[0].set_title(f"{label}\ntrue colour", fontsize=8)
    axes[1].imshow(false_color_nir(img)); axes[1].set_title("false colour (NIR)", fontsize=8)
    axes[2].imshow(index_map(img, "NDVI"), cmap="RdYlGn", vmin=-1, vmax=1); axes[2].set_title("NDVI", fontsize=8)
    axes[3].imshow(index_map(img, "NDWI"), cmap="BrBG_r", vmin=-1, vmax=1); axes[3].set_title("NDWI", fontsize=8)
    for a in axes:
        a.axis("off")
    return fig, axes


def gradcam_overlay(model, input_tensor, target_layer, rgb_for_display, device="cpu"):
    """Grad-CAM heatmap over a display image. Returns (cam, overlay)."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale = cam(input_tensor=input_tensor.to(device))[0]
    overlay = show_cam_on_image(rgb_for_display.astype(np.float32), grayscale, use_rgb=True)
    return grayscale, overlay


def _save(fig, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
