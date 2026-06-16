"""Post-hoc analysis: results table, paired McNemar tests, per-class deltas.

All functions consume the ID-keyed prediction CSVs written by the shared
evaluator, so two arms' predictions are always aligned by image path before any
paired comparison, there is no way to silently mis-pair rows.

McNemar protocol (pre-registered, per docs/decisions.md):
  * comparison family = each multispectral arm vs the RGB baseline E1;
  * run per matched seed (seed-i model vs seed-i model), never pooled;
  * exact binomial when the discordant count b+c < 25, else corrected chi-square;
  * report the discordant counts b and c alongside every p-value;
  * Holm-correct p-values across the family.
References: Dietterich (1998); Holm (1979); Demsar (2006).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CLASS_NAMES


# --------------------------------------------------------------------------- #
# Loading / aggregation
# --------------------------------------------------------------------------- #
def load_all_metrics(results_dir: str | Path) -> pd.DataFrame:
    """Load every results/metrics/*.json into one tidy DataFrame."""
    rows = []
    for p in sorted(Path(results_dir, "metrics").glob("*.json")):
        d = json.loads(p.read_text())
        if "accuracy" not in d:  # skip e.g. permutation-importance CSV siblings
            continue
        rows.append({
            "run": d.get("arm", p.stem),
            "arm_base": d.get("arm_base", d.get("arm", p.stem)),
            "tier": d.get("tier", "?"),
            "seed": d.get("seed", -1),
            "fraction": d.get("fraction", 1.0),
            "accuracy": d["accuracy"],
            "macro_f1": d["macro_f1"],
            "balanced_accuracy": d.get("balanced_accuracy", float("nan")),
            "n": d.get("n", 0),
        })
    return pd.DataFrame(rows)


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std (and per-seed spread) over seeds, per arm_base x fraction."""
    g = metrics.groupby(["arm_base", "fraction"])
    out = g.agg(
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        f1_mean=("macro_f1", "mean"),
        f1_std=("macro_f1", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    # n=3 std is indicative only; keep per-seed min/max for honesty.
    out["acc_min"] = g["accuracy"].min().values
    out["acc_max"] = g["accuracy"].max().values
    return out.sort_values(["fraction", "acc_mean"], ascending=[True, False]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# McNemar
# --------------------------------------------------------------------------- #
def _aligned_correct(results_dir, run_a: str, run_b: str):
    """Return per-sample correctness of two runs, aligned by image path."""
    a = pd.read_csv(Path(results_dir, "predictions", f"{run_a}.csv"))
    b = pd.read_csv(Path(results_dir, "predictions", f"{run_b}.csv"))
    m = a.merge(b, on="path", suffixes=("_a", "_b"), validate="one_to_one")
    assert (m["label_a"] == m["label_b"]).all(), "label mismatch, different test splits"
    return (m["pred_a"] == m["label_a"]).to_numpy(), (m["pred_b"] == m["label_b"]).to_numpy()


def mcnemar_pair(results_dir, run_a: str, run_b: str) -> dict:
    """McNemar's test between two runs (b = a-only-correct, c = b-only-correct)."""
    from statsmodels.stats.contingency_tables import mcnemar

    ca, cb = _aligned_correct(results_dir, run_a, run_b)
    b = int(np.sum(ca & ~cb))   # a correct, b wrong
    c = int(np.sum(~ca & cb))   # a wrong, b correct
    exact = (b + c) < 25
    res = mcnemar([[int(np.sum(ca & cb)), b], [c, int(np.sum(~ca & ~cb))]],
                  exact=exact, correction=not exact)
    return {
        "run_a": run_a, "run_b": run_b,
        "b_a_only_correct": b, "c_b_only_correct": c, "discordant": b + c,
        "statistic": float(res.statistic), "pvalue": float(res.pvalue),
        "method": "exact-binomial" if exact else "chi2-corrected",
    }


def mcnemar_family(results_dir, baseline_runs, candidate_runs, alpha: float = 0.05) -> pd.DataFrame:
    """Paired McNemar for each (baseline, candidate) per matched seed, Holm-corrected.

    `baseline_runs` and `candidate_runs` are lists of run ids already matched by
    position (same seed). Returns one row per comparison with raw and Holm p-values.
    """
    from statsmodels.stats.multitest import multipletests

    rows = [mcnemar_pair(results_dir, a, b) for a, b in zip(baseline_runs, candidate_runs)]
    df = pd.DataFrame(rows)
    if len(df):
        df["pvalue_holm"] = multipletests(df["pvalue"], alpha=alpha, method="holm")[1]
        df["significant"] = df["pvalue_holm"] < alpha
    return df


# --------------------------------------------------------------------------- #
# Per-class analysis
# --------------------------------------------------------------------------- #
def per_class_recall(results_dir, run: str) -> pd.Series:
    df = pd.read_csv(Path(results_dir, "predictions", f"{run}.csv"))
    out = {}
    for i, cls in enumerate(CLASS_NAMES):
        sel = df["label"] == i
        out[cls] = float((df.loc[sel, "pred"] == i).mean()) if sel.any() else float("nan")
    return pd.Series(out, name=run)


def per_class_delta(results_dir, run_baseline: str, run_candidate: str) -> pd.DataFrame:
    """Per-class recall for two runs and the candidate-minus-baseline delta (pp)."""
    base = per_class_recall(results_dir, run_baseline)
    cand = per_class_recall(results_dir, run_candidate)
    counts = pd.read_csv(Path(results_dir, "predictions", f"{run_baseline}.csv"))["label"].value_counts()
    support = {CLASS_NAMES[i]: int(counts.get(i, 0)) for i in range(len(CLASS_NAMES))}
    return pd.DataFrame({
        "support": pd.Series(support),
        f"{run_baseline}_recall": base,
        f"{run_candidate}_recall": cand,
        "delta_pp": (cand - base) * 100,
    }).sort_values("delta_pp", ascending=False)


def confusion(results_dir, run: str) -> np.ndarray:
    from sklearn.metrics import confusion_matrix
    df = pd.read_csv(Path(results_dir, "predictions", f"{run}.csv"))
    return confusion_matrix(df["label"], df["pred"], labels=list(range(len(CLASS_NAMES))))
