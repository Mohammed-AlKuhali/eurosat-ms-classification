"""Run the Tier-1 classical baseline: RF arms C1 (RGB) and C2 (multispectral).

Trains on the committed train manifest, evaluates on the committed test manifest
via the shared evaluator, and saves a per-band/index permutation-importance
figure for the multispectral arm.

Usage
-----
    python scripts/run_classical.py
    EUROSAT_DATA_ROOT=/path/to/EuroSAT_MS python scripts/run_classical.py --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eurosat_ms.classical import (
    ARM_FEATURES,
    extract_features,
    feature_names,
    permutation_importance_df,
    train_random_forest,
)
from eurosat_ms.data import default_data_root, load_manifest
from eurosat_ms.evaluate import evaluate_arm


def plot_importance(imp_df, out_path, top=20):
    d = imp_df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color="#0F6E56")
    ax.set_xlabel("permutation importance (accuracy drop)")
    ax.set_title("Tier-1 RF — top features (C2 multispectral)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--cache-dir", default="data/features")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root) if args.data_root else default_data_root()
    train = load_manifest(args.manifest_dir, "train")
    test = load_manifest(args.manifest_dir, "test")
    cache = Path(args.cache_dir)

    summary = []
    for arm in ARM_FEATURES:
        print(f"\n=== {arm} ===")
        Xtr, ytr = extract_features(data_root, train, arm, cache / f"{arm}_train.npz")
        Xte, yte = extract_features(data_root, test, arm, cache / f"{arm}_test.npz")
        print(f"  features: train {Xtr.shape}, test {Xte.shape}")
        clf = train_random_forest(Xtr, ytr, seed=args.seed)
        metrics = evaluate_arm(
            arm, list(test["path"]), yte, clf.predict(Xte), args.results_dir,
            extra={"tier": "classical", "model": "RandomForest", "seed": args.seed},
        )
        print(f"  accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}")
        summary.append((arm, metrics["accuracy"], metrics["macro_f1"]))

        if arm == "C2_multispectral":
            imp = permutation_importance_df(clf, Xte, yte, feature_names(arm), seed=args.seed)
            imp.to_csv(Path(args.results_dir) / "metrics" / "C2_permutation_importance.csv", index=False)
            plot_importance(imp, Path(args.results_dir) / "figures" / "C2_permutation_importance.png")
            print("  top-5 features:", ", ".join(imp.head(5)["feature"]))

    print("\n=== Tier-1 summary ===")
    for arm, acc, f1 in summary:
        print(f"  {arm:20s} acc={acc:.4f}  macro_f1={f1:.4f}")


if __name__ == "__main__":
    main()
