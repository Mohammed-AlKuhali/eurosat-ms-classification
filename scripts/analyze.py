"""Produce all analysis artifacts from whatever results currently exist.

Robust to partially-complete runs: each block runs only if its prediction CSVs
are present, so this is useful both after Tier-1 alone and after the full Colab
matrix. Writes tables to results/tables/ and figures to results/figures/.

Usage
-----
    python scripts/analyze.py                      # tables + figures + gallery
    python scripts/analyze.py --no-gallery         # skip image rendering
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from eurosat_ms import analysis as A
from eurosat_ms import visualize as V
from eurosat_ms.data import default_data_root, load_manifest, load_patch


def _runs_for(results_dir, arm_base):
    preds = Path(results_dir, "predictions")
    return sorted(p.stem for p in preds.glob(f"{arm_base}__s*.csv") if "__f" not in p.stem)


def _exists(results_dir, run):
    return Path(results_dir, "predictions", f"{run}.csv").exists()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--no-gallery", action="store_true")
    args = ap.parse_args()
    rd = args.results_dir
    tables = Path(rd, "tables"); tables.mkdir(parents=True, exist_ok=True)

    # 1. Summary table over all arms/seeds/fractions.
    metrics = A.load_all_metrics(rd)
    summ = A.summary_table(metrics)
    summ.to_csv(tables / "summary.csv", index=False)
    print("=== summary ===\n", summ.to_string(index=False))

    # 2. McNemar family: each MS CNN arm vs E1 baseline, per matched seed, Holm.
    e1 = _runs_for(rd, "E1_rgb")
    mcnemar_rows = []
    for cand_base in ("E2_multispectral", "E3_indices", "E4_rgbnir", "E7_ssl4eo_all"):
        cand = _runs_for(rd, cand_base)
        pairs = [(a, b) for a, b in zip(e1, cand)]  # matched by seed order
        if pairs:
            fam = A.mcnemar_family(rd, [a for a, _ in pairs], [b for _, b in pairs])
            fam["candidate_base"] = cand_base
            mcnemar_rows.append(fam)
    # Tier-1 comparison is always available.
    if _exists(rd, "C1_rgb") and _exists(rd, "C2_multispectral"):
        t1 = pd.DataFrame([A.mcnemar_pair(rd, "C1_rgb", "C2_multispectral")])
        t1["candidate_base"] = "C2_multispectral_tier1"
        mcnemar_rows.append(t1)
    if mcnemar_rows:
        mc = pd.concat(mcnemar_rows, ignore_index=True)
        mc.to_csv(tables / "mcnemar.csv", index=False)
        print("\n=== McNemar (vs RGB baseline) ===\n", mc.to_string(index=False))

    # 3. Per-class deltas + confusion matrices for the available headline pairs.
    headline_pairs = [("C1_rgb", "C2_multispectral")]
    if e1 and _runs_for(rd, "E2_multispectral"):
        headline_pairs.append((e1[0], _runs_for(rd, "E2_multispectral")[0]))
    for base, cand in headline_pairs:
        if _exists(rd, base) and _exists(rd, cand):
            d = A.per_class_delta(rd, base, cand)
            d.to_csv(tables / f"per_class_delta__{base}__{cand}.csv")
            V.plot_per_class_delta(d, Path(rd, "figures", f"per_class_delta__{cand}.png"),
                                   title=f"Per-class recall: {cand} - {base}")
            V.plot_confusion(A.confusion(rd, cand), Path(rd, "figures", f"confusion__{cand}.png"),
                             title=f"Confusion matrix — {cand}")

    # 4. Data-efficiency curve (if E6 fractional runs exist).
    frac = metrics[metrics["fraction"] < 1.0]
    if len(frac):
        curve = (frac.groupby(["arm_base", "fraction"])
                 .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"))
                 .reset_index())
        curve.to_csv(tables / "data_efficiency.csv", index=False)
        V.plot_data_efficiency(curve, Path(rd, "figures", "data_efficiency.png"))

    # 5. Subjective gallery from the PRE-REGISTERED sample ids.
    gpath = Path(rd, "subjective_sample_ids.json")
    if not args.no_gallery and gpath.exists():
        data_root = Path(args.data_root) if args.data_root else default_data_root()
        ids = json.loads(gpath.read_text())
        picks = [(c, p) for c, ps in ids["per_class"].items() for p in ps]
        fig, axes = plt.subplots(len(picks), 4, figsize=(11, 2.5 * len(picks)))
        for row, (cls, rel) in enumerate(picks):
            V.gallery_row(load_patch(data_root / rel), axes=axes[row], label=cls)
        fig.suptitle("Pre-registered subjective gallery (true / false-colour / NDVI / NDWI)", y=1.0)
        V._save(fig, Path(rd, "figures", "subjective_gallery.png"))

    print("\nAnalysis complete. Tables in results/tables/, figures in results/figures/.")


if __name__ == "__main__":
    main()
