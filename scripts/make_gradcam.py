"""Grad-CAM explainability: where do the RGB and multispectral models look?

For a few test patches, overlay Grad-CAM heatmaps from the E1 (RGB) and E2
(12-band multispectral) ResNet-18 checkpoints on the true-colour image. Optional
extension (brief: model explainability techniques).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eurosat_ms.bands import BANDS_12, RGB_BANDS
from eurosat_ms.data import CLASS_NAMES, default_data_root, load_manifest, load_patch, load_stats
from eurosat_ms.models import build_model
from eurosat_ms.torch_dataset import EuroSATArmDataset
from eurosat_ms.visualize import gradcam_overlay, true_color

WANT = ["River", "Highway", "PermanentCrop"]


def load_model(in_chans: int, ckpt: str):
    m = build_model(in_chans, num_classes=10, pretrained="none", backbone="resnet18")
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    return m


def make_input(root, row_df, bands, stats):
    ds = EuroSATArmDataset(root, row_df.reset_index(drop=True), bands, [], stats, "zscore", train=False)
    x, _ = ds[0]
    return x.unsqueeze(0)


def main(seed: int = 5) -> None:
    root = default_data_root()
    stats = load_stats("data/stats/train_stats.json")
    test = load_manifest("data/manifests", "test")
    rng = np.random.default_rng(seed)
    rels = [test[test.class_name == c].path.iloc[int(rng.integers(len(test[test.class_name == c])))]
            for c in WANT]

    m_rgb = load_model(3, "checkpoints/E1_rgb__s0.pth")
    m_ms = load_model(12, "checkpoints/E2_multispectral__s0.pth")
    tgt_rgb, tgt_ms = m_rgb.layer4[-1], m_ms.layer4[-1]

    fig, axes = plt.subplots(len(rels), 3, figsize=(6.6, 2.3 * len(rels)))
    titles = ["true colour", "E1 RGB Grad-CAM", "E2 multispectral Grad-CAM"]
    for i, rel in enumerate(rels):
        img = load_patch(root / rel)
        disp = true_color(img).astype(np.float32)
        row = test[test.path == rel]
        _, ov_rgb = gradcam_overlay(m_rgb, make_input(root, row, RGB_BANDS, stats), tgt_rgb, disp)
        _, ov_ms = gradcam_overlay(m_ms, make_input(root, row, BANDS_12, stats), tgt_ms, disp)
        for c, im in enumerate([disp, ov_rgb, ov_ms]):
            axes[i, c].imshow(im)
            if i == 0:
                axes[i, c].set_title(titles[c], fontsize=9)
            axes[i, c].set_xticks([]); axes[i, c].set_yticks([])
        axes[i, 0].set_ylabel(rel.split("/")[0], fontsize=8)
    fig.tight_layout()
    out = Path("results/figures/gradcam.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
