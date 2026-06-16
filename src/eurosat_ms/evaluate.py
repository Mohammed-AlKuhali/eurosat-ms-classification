"""Shared evaluation used by EVERY arm, classical and CNN alike.

Producing predictions through one code path guarantees that:
  * metrics are computed identically across arms, and
  * per-sample predictions are written **keyed by image path**, so the later
    paired McNemar test can align two arms' predictions row-by-row without any
    risk of silent mis-pairing.

The brief mandates accuracy on the 80/20 split; we additionally report macro-F1
and per-class recall because EuroSAT is mildly imbalanced (2000-3000/class).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)

from .data import CLASS_NAMES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, macro-F1, balanced accuracy, and per-class recall."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(CLASS_NAMES)))
    per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_recall": {CLASS_NAMES[i]: float(per_class[i]) for i in labels},
        "n": int(len(y_true)),
    }


def save_predictions(
    out_path: str | Path,
    paths: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Write a per-sample prediction CSV keyed by image path.

    Columns: path, label, pred, one row per test image, in manifest order.
    """
    df = pd.DataFrame({
        "path": list(paths),
        "label": np.asarray(y_true).astype(int),
        "pred": np.asarray(y_pred).astype(int),
    })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def evaluate_arm(
    arm: str,
    paths: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    results_dir: str | Path,
    extra: dict | None = None,
) -> dict:
    """Save predictions + metrics for one arm and return the metrics dict.

    Writes:
      results/predictions/<arm>.csv   (ID-keyed per-sample predictions)
      results/metrics/<arm>.json      (accuracy, macro-F1, per-class recall)
    """
    results_dir = Path(results_dir)
    save_predictions(results_dir / "predictions" / f"{arm}.csv", paths, y_true, y_pred)
    metrics = compute_metrics(y_true, y_pred)
    metrics["arm"] = arm
    if extra:
        metrics.update(extra)
    mpath = results_dir / "metrics" / f"{arm}.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, indent=2))
    return metrics


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Load an arm's prediction CSV (for analysis / McNemar)."""
    return pd.read_csv(path)
