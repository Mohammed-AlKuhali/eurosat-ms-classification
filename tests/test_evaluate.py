"""Metric correctness and prediction-CSV integrity for the shared evaluator."""

from __future__ import annotations

import numpy as np

from eurosat_ms.evaluate import compute_metrics, evaluate_arm, load_predictions


def test_perfect_prediction_scores_one():
    # All 10 classes present, as in the real test split.
    y = np.arange(10)
    m = compute_metrics(y, y.copy())
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["n"] == 10


def test_metrics_match_known_confusion():
    # 3 correct, 1 wrong -> accuracy 0.75
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 0])
    m = compute_metrics(y_true, y_pred)
    assert abs(m["accuracy"] - 0.75) < 1e-9
    # class 1 recall: 1 of 2 correct
    assert abs(m["per_class_recall"]["Forest"] - 0.5) < 1e-9  # label 1 == Forest


def test_evaluate_arm_writes_id_keyed_csv(tmp_path):
    paths = ["River/River_1.tif", "Forest/Forest_2.tif", "SeaLake/SeaLake_3.tif"]
    y_true = np.array([8, 1, 9])
    y_pred = np.array([8, 1, 8])
    m = evaluate_arm("toy", paths, y_true, y_pred, tmp_path)
    df = load_predictions(tmp_path / "predictions" / "toy.csv")
    # CSV must preserve order and pair path<->label<->pred for McNemar.
    assert list(df["path"]) == paths
    assert list(df["label"]) == [8, 1, 9]
    assert list(df["pred"]) == [8, 1, 8]
    assert abs(m["accuracy"] - 2 / 3) < 1e-9
