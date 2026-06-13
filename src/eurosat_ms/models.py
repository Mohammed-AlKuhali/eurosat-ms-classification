"""Model construction: ImageNet- and SSL4EO-pretrained ResNet18 backbones.

Conv1 adaptation for N-channel input is delegated to timm's canonical recipe
(``in_chans=N`` repeats the pretrained RGB first-conv weights cyclically to N
channels and rescales by 3/N to preserve activation magnitude — the I3D
"inflation" idea). We do not hand-roll first-layer surgery.

Pretraining options
-------------------
* ``imagenet``        — timm ImageNet-1k weights (RGB-domain).
* ``none``            — random init (the from-scratch control, arm E5).
* ``ssl4eo_all_moco`` — torchgeo SSL4EO-S12 ResNet18, 13-band Sentinel-2 (MoCo).
* ``ssl4eo_rgb_moco`` — torchgeo SSL4EO-S12 ResNet18, RGB Sentinel-2 (MoCo).

The two SSL4EO arms complete a clean 2x2 — {ImageNet, SSL4EO} x {RGB, all-band} —
that separates the *pretraining domain* from the *spectral information*, so a
full-data "MS doesn't beat RGB" result cannot be dismissed as a pretraining
artifact.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_model(
    in_chans: int,
    num_classes: int = 10,
    pretrained: str = "imagenet",
    backbone: str = "resnet18",
) -> nn.Module:
    """Create a backbone with an `in_chans`-channel stem and a fresh head."""
    import timm

    if pretrained in ("imagenet", "none"):
        return timm.create_model(
            backbone,
            pretrained=(pretrained == "imagenet"),
            in_chans=in_chans,
            num_classes=num_classes,
        )
    if pretrained in ("ssl4eo_all_moco", "ssl4eo_rgb_moco"):
        return _build_ssl4eo(pretrained, in_chans, num_classes, backbone)
    raise ValueError(f"unknown pretrained option: {pretrained}")


def _build_ssl4eo(weights_key: str, in_chans: int, num_classes: int, backbone: str) -> nn.Module:
    """Load SSL4EO-S12 Sentinel-2 weights via torchgeo into a timm backbone.

    torchgeo exposes the SSL4EO weights as a state dict for a timm resnet; we
    create the matching timm model (no ImageNet init), load the weights with
    strict=False (only the classification head is fresh), and report what was
    left uninitialised so any silent mismatch is visible in the logs.
    """
    import timm
    from torchgeo.models import ResNet18_Weights

    weights = {
        "ssl4eo_all_moco": ResNet18_Weights.SENTINEL2_ALL_MOCO,
        "ssl4eo_rgb_moco": ResNet18_Weights.SENTINEL2_RGB_MOCO,
    }[weights_key]

    expected = weights.meta["in_chans"]
    if in_chans != expected:
        raise ValueError(
            f"{weights_key} expects in_chans={expected}, got {in_chans}. "
            f"Use the band list this weight was trained on (see docs/decisions.md)."
        )

    model = timm.create_model(backbone, pretrained=False, in_chans=in_chans, num_classes=num_classes)
    state = weights.get_state_dict(progress=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Only the classification head (fc.*) should be missing.
    non_head_missing = [k for k in missing if not k.startswith("fc.")]
    if non_head_missing:
        raise RuntimeError(f"SSL4EO load left non-head weights uninitialised: {non_head_missing[:8]}")
    print(f"  [ssl4eo] loaded {weights_key}; fresh head only "
          f"(missing={list(missing)}, unexpected={list(unexpected)[:4]})")
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
