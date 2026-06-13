"""Guards against the smoke-contamination bug.

A 1-epoch/64-image `--smoke` run must never be mistaken for a completed full
run by the idempotent skip-if-done logic. Two invariants:

  1. A cached metrics file only counts as "complete" if it covers the full
     test set (n == len(test manifest)) and is not flagged as a smoke run.
  2. Smoke runs are redirected to an isolated results directory, so they
     cannot overwrite canonical results in the first place.
"""

from __future__ import annotations

import json

from eurosat_ms.train import _is_run_complete, _smoke_dirs


def _write(metrics_path, payload):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload))


def test_smoke_sized_result_is_not_complete(tmp_path):
    p = tmp_path / "metrics" / "E1_rgb__s0.json"
    _write(p, {"n": 32, "accuracy": 0.47})          # smoke eval size
    assert _is_run_complete(p, "data/manifests") is False


def test_smoke_flagged_result_is_not_complete(tmp_path):
    p = tmp_path / "metrics" / "E1_rgb__s0.json"
    _write(p, {"n": 5400, "smoke": True})           # full-sized but flagged smoke
    assert _is_run_complete(p, "data/manifests") is False


def test_full_result_is_complete(tmp_path):
    p = tmp_path / "metrics" / "E1_rgb__s0.json"
    _write(p, {"n": 5400, "smoke": False})
    assert _is_run_complete(p, "data/manifests") is True


def test_missing_result_is_not_complete(tmp_path):
    assert _is_run_complete(tmp_path / "nope.json", "data/manifests") is False


def test_smoke_dirs_are_isolated_from_canonical():
    res, ckpt = _smoke_dirs("results", "checkpoints")
    assert "_smoke" in str(res) and "_smoke" in str(ckpt)
    assert str(res) != "results" and str(ckpt) != "checkpoints"
