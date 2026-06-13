"""Training loop for the CNN arms — seeded, checkpointed, idempotent.

One frozen recipe is shared across all arms (AdamW + cosine schedule + early
stopping on validation macro-F1), so arms differ only in their inputs and
pretraining, never in tuning. Runs are keyed by arm/seed/fraction and skip
themselves if already complete, which makes a dropped Colab session resumable
by simply re-running.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import load_manifest, load_stats
from .evaluate import evaluate_arm
from .models import build_model
from .torch_dataset import EuroSATArmDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def subsample_train(manifest: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Stratified subsample of the train manifest for the data-efficiency curve."""
    if fraction >= 1.0:
        return manifest
    return (
        manifest.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(frac=fraction, random_state=seed))
        .reset_index(drop=True)
    )


def run_id(arm: str, seed: int, fraction: float) -> str:
    rid = f"{arm}__s{seed}"
    if fraction < 1.0:
        rid += f"__f{fraction:g}"
    return rid


def _evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds.append(out.argmax(1).cpu().numpy())
            ys.append(np.asarray(y))
    return np.concatenate(ys), np.concatenate(preds)


def train_arm(
    cfg: dict,
    seed: int,
    fraction: float = 1.0,
    data_root: str | Path = None,
    manifest_dir: str | Path = "data/manifests",
    stats_path: str | Path = "data/stats/train_stats.json",
    results_dir: str | Path = "results",
    ckpt_dir: str | Path = "checkpoints",
    device: str | None = None,
    smoke: bool = False,
) -> dict | None:
    """Train one arm at one seed (and optional label fraction); return test metrics.

    Skips and returns the cached metrics if this run already completed.
    """
    from .data import default_data_root

    data_root = Path(data_root) if data_root else default_data_root()
    device = device or pick_device()
    rid = run_id(cfg["arm"], seed, fraction)
    metrics_path = Path(results_dir) / "metrics" / f"{rid}.json"
    if metrics_path.exists() and not smoke:
        print(f"[skip] {rid} already complete")
        return json.loads(metrics_path.read_text())

    set_seed(seed)
    stats = load_stats(stats_path)
    bands, indices = cfg["bands"], cfg.get("indices", [])
    norm = cfg.get("normalization", "zscore")

    train_m = subsample_train(load_manifest(manifest_dir, "train"), fraction, seed)
    val_m = load_manifest(manifest_dir, "val")
    test_m = load_manifest(manifest_dir, "test")
    if smoke:
        train_m, val_m, test_m = train_m.head(64), val_m.head(32), test_m.head(32)

    n_workers = 0 if smoke else cfg.get("num_workers", 4)

    def make_loader(m, train):
        ds = EuroSATArmDataset(data_root, m, bands, indices, stats, norm, train=train)
        return DataLoader(ds, batch_size=cfg.get("batch_size", 64), shuffle=train,
                          num_workers=n_workers, drop_last=False)

    train_loader = make_loader(train_m, True)
    val_loader = make_loader(val_m, False)
    test_loader = make_loader(test_m, False)

    in_chans = len(bands) + len(indices)
    model = build_model(in_chans, num_classes=10,
                        pretrained=cfg.get("pretrained", "imagenet"),
                        backbone=cfg.get("backbone", "resnet18")).to(device)

    epochs = 1 if smoke else cfg.get("epochs", 30)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3),
                            weight_decay=cfg.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    best_path = Path(ckpt_dir) / f"{rid}.pth"
    best_f1, patience, bad = -1.0, cfg.get("patience", 7), 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        from sklearn.metrics import f1_score
        yv, pv = _evaluate(model, val_loader, device)
        vf1 = f1_score(yv, pv, average="macro", labels=list(range(10)), zero_division=0)
        print(f"  [{rid}] epoch {epoch+1}/{epochs} val_macroF1={vf1:.4f}")
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"  [{rid}] early stop at epoch {epoch+1}")
                break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    y_true, y_pred = _evaluate(model, test_loader, device)
    metrics = evaluate_arm(
        rid, list(test_m["path"]), y_true, y_pred, results_dir,
        extra={
            "tier": "cnn", "arm_base": cfg["arm"], "seed": seed, "fraction": fraction,
            "in_chans": in_chans, "pretrained": cfg.get("pretrained", "imagenet"),
            "best_val_macro_f1": float(best_f1), "train_seconds": round(time.time() - t0, 1),
            "n_train": int(len(train_m)),
        },
    )
    print(f"  [{rid}] TEST acc={metrics['accuracy']:.4f} macroF1={metrics['macro_f1']:.4f} "
          f"({metrics['train_seconds']}s)")
    return metrics
