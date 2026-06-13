"""Split integrity: disjointness, stratification, and patch shape."""

from __future__ import annotations

import numpy as np

from eurosat_ms.data import (
    CLASS_NAMES,
    N_BANDS,
    generate_manifests,
    load_patch,
)


def test_patch_shape_and_dtype(data_root):
    p = sorted((data_root / "Forest").glob("*.tif"))[0]
    img = load_patch(p)
    assert img.shape == (N_BANDS, 64, 64)
    assert img.dtype == np.uint16


def test_splits_disjoint_and_stratified(data_root, tmp_path):
    splits = generate_manifests(data_root, tmp_path, seed=42)
    paths = {name: set(df["path"]) for name, df in splits.items()}

    # No leakage across splits.
    assert paths["train"].isdisjoint(paths["test"])
    assert paths["train"].isdisjoint(paths["val"])
    assert paths["val"].isdisjoint(paths["test"])

    # All 27,000 images accounted for exactly once.
    total = sum(len(s) for s in paths.values())
    assert total == 27000
    assert len(paths["train"] | paths["val"] | paths["test"]) == total

    # Test split is ~20% overall.
    assert abs(len(paths["test"]) / total - 0.20) < 0.01

    # Every class present in every split, roughly in proportion.
    for name, df in splits.items():
        assert set(df["class_name"]) == set(CLASS_NAMES)


def test_split_is_deterministic(data_root, tmp_path):
    a = generate_manifests(data_root, tmp_path / "a", seed=42)["test"]
    b = generate_manifests(data_root, tmp_path / "b", seed=42)["test"]
    assert list(a["path"]) == list(b["path"])
