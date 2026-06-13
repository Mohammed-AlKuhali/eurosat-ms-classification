"""Run one experiment arm from its YAML config (all seeds, all label fractions).

Usage
-----
    python scripts/run_experiment.py configs/e1_rgb.yaml
    python scripts/run_experiment.py configs/e6_ms.yaml --device cuda
    python scripts/run_experiment.py configs/e1_rgb.yaml --smoke      # 1 epoch, 64 imgs
    python scripts/run_experiment.py configs/e2_multispectral.yaml --seeds 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from eurosat_ms.train import pick_device, train_arm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="override config seeds")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seeds = args.seeds if args.seeds is not None else cfg.get("seeds", [0])
    fractions = cfg.get("label_fractions", [1.0])
    device = args.device or pick_device()
    print(f"=== {cfg['arm']} | device={device} | seeds={seeds} | fractions={fractions} ===")

    for fraction in fractions:
        for seed in seeds:
            train_arm(cfg, seed=seed, fraction=fraction, data_root=args.data_root,
                      results_dir=args.results_dir, device=device, smoke=args.smoke)


if __name__ == "__main__":
    main()
