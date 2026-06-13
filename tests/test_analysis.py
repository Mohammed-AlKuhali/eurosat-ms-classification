"""McNemar correctness on synthetic predictions with known discordant counts."""

from __future__ import annotations

import pandas as pd

from eurosat_ms.analysis import mcnemar_pair


def _write(results_dir, run, paths, labels, preds):
    d = results_dir / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": paths, "label": labels, "pred": preds}).to_csv(d / f"{run}.csv", index=False)


def test_mcnemar_counts_and_alignment(tmp_path):
    # 6 samples. Construct a known discordant pattern between A and B.
    paths = [f"C/{i}.tif" for i in range(6)]
    labels = [0, 0, 0, 0, 0, 0]
    # A correct on 0,1,2,3 ; B correct on 0,1,4,5
    preds_a = [0, 0, 0, 0, 9, 9]
    preds_b = [0, 0, 9, 9, 0, 0]
    _write(tmp_path, "A", paths, labels, preds_a)
    # B written with SHUFFLED row order to prove alignment-by-path works.
    order = [5, 0, 3, 1, 4, 2]
    _write(tmp_path, "B", [paths[i] for i in order], [labels[i] for i in order],
           [preds_b[i] for i in order])

    r = mcnemar_pair(tmp_path, "A", "B")
    # A-only-correct: samples 2,3 -> b=2 ; B-only-correct: samples 4,5 -> c=2
    assert r["b_a_only_correct"] == 2
    assert r["c_b_only_correct"] == 2
    assert r["discordant"] == 4
    assert r["method"] == "exact-binomial"  # b+c < 25
