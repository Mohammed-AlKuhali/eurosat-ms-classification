"""PyTorch Dataset that assembles per-arm input tensors from EuroSAT patches.

An arm is fully described by:
  * ``bands``        — ordered band names to take from the patch,
  * ``indices``      — spectral indices to append as extra channels,
  * ``normalization``— 'zscore' (train stats) or 'div10000' (SSL4EO weights),
plus train-time augmentation (flips + 90-degree rotations only — no photometric
jitter, which would corrupt the physical meaning of spectral values).

Because the band list is explicit and ordered, the SSL4EO channel reordering is
just ``bands = SSL4EO_ORDER`` with ``normalization='div10000'`` — there is no
separate, error-prone reindex step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .bands import BAND_INDEX
from .data import load_patch
from .features import INDEX_DEFS, normalized_difference


def build_channel_stats(stats: dict, bands: list[str], indices: list[str], mode: str):
    """Return (mean, std) vectors aligned to the [bands..., indices...] channel order."""
    if mode == "zscore":
        mean = [stats["band_mean"][b] for b in bands] + [stats["index_mean"][i] for i in indices]
        std = [stats["band_std"][b] for b in bands] + [stats["index_std"][i] for i in indices]
    elif mode == "div10000":
        # Match SSL4EO pretraining: reflectance/10000, indices (none in those arms) pass through.
        mean = [0.0] * len(bands) + [0.0] * len(indices)
        std = [10000.0] * len(bands) + [1.0] * len(indices)
    else:
        raise ValueError(f"unknown normalization mode: {mode}")
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


class EuroSATArmDataset(Dataset):
    """Patches assembled and normalised for a specific experiment arm."""

    def __init__(
        self,
        data_root: str | Path,
        manifest: pd.DataFrame,
        bands: list[str],
        indices: list[str],
        stats: dict,
        normalization: str = "zscore",
        train: bool = False,
    ):
        self.data_root = Path(data_root)
        self.paths = list(manifest["path"])
        self.labels = list(manifest["label"].astype(int))
        self.bands = bands
        self.indices = indices
        self.train = train
        mean, std = build_channel_stats(stats, bands, indices, normalization)
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
        self.band_idx = [BAND_INDEX[b] for b in bands]

    def __len__(self) -> int:
        return len(self.paths)

    def _assemble(self, img: np.ndarray) -> np.ndarray:
        chans = [img[i].astype(np.float32) for i in self.band_idx]
        for name in self.indices:
            chans.append(normalized_difference(img, *INDEX_DEFS[name]))
        return np.stack(chans, axis=0)  # (C, H, W)

    @staticmethod
    def _augment(x: np.ndarray) -> np.ndarray:
        # Dihedral group: random flips + k*90-degree rotation. Spectral-safe.
        if np.random.rand() < 0.5:
            x = x[:, ::-1, :]
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]
        k = np.random.randint(4)
        if k:
            x = np.rot90(x, k=k, axes=(1, 2))
        return np.ascontiguousarray(x)

    def __getitem__(self, i: int):
        img = load_patch(self.data_root / self.paths[i])
        x = self._assemble(img)
        if self.train:
            x = self._augment(x)
        x = (x - self.mean) / self.std
        return torch.from_numpy(x.astype(np.float32)), self.labels[i]
