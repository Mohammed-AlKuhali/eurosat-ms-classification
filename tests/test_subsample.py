"""Guards the data-efficiency subsampling (regression: pandas >=2.2 dropped 'label')."""

from __future__ import annotations

import pandas as pd

from eurosat_ms.train import subsample_train


def _toy(n_per_class=100, n_classes=10):
    rows = [{"path": f"c{c}/img_{i}.tif", "class_name": f"c{c}", "label": c}
            for c in range(n_classes) for i in range(n_per_class)]
    return pd.DataFrame(rows)


def test_subsample_preserves_all_columns():
    sub = subsample_train(_toy(), 0.1, seed=0)
    assert set(sub.columns) == {"path", "class_name", "label"}, "label column must survive"


def test_subsample_is_stratified():
    sub = subsample_train(_toy(n_per_class=100), 0.1, seed=0)
    counts = sub["label"].value_counts()
    assert len(counts) == 10                       # every class represented
    assert counts.min() == counts.max() == 10      # 10% of 100 per class


def test_subsample_is_deterministic():
    a = subsample_train(_toy(), 0.2, seed=0)
    b = subsample_train(_toy(), 0.2, seed=0)
    assert list(a["path"]) == list(b["path"])


def test_full_fraction_returns_all():
    m = _toy()
    assert len(subsample_train(m, 1.0, seed=0)) == len(m)
