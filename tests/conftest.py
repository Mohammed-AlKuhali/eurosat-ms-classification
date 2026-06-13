"""Shared test fixtures.

Tests that need real imagery use a few patches sampled from the local
EuroSAT_MS download. If the data is not present, those tests are skipped
(so the pure-logic tests still run anywhere, e.g. in CI without the 2 GB set).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eurosat_ms.data import default_data_root, load_patch


@pytest.fixture(scope="session")
def data_root() -> Path:
    root = default_data_root()
    if not root.is_dir() or not any(root.glob("Forest/*.tif")):
        pytest.skip(f"EuroSAT_MS not found at {root}; skipping data-dependent tests")
    return root


@pytest.fixture(scope="session")
def forest_patch(data_root):
    p = sorted((data_root / "Forest").glob("*.tif"))[0]
    return load_patch(p)


@pytest.fixture(scope="session")
def sealake_patch(data_root):
    p = sorted((data_root / "SeaLake").glob("*.tif"))[0]
    return load_patch(p)
