"""Hand-label binary water masks on a small, pre-registered set of patches.

Run locally (needs a display). For each River/SeaLake patch it shows the
true-colour image and the NDWI map side by side; you click polygon vertices
around the water, then:
    n / enter  save mask, next patch
    c          clear the current polygon and redraw
    s          skip this patch (no water / too ambiguous)
    q          quit (progress is saved per patch)

Masks are saved as data/seg_labels/<ClassName>_<id>.npy plus a labels.json
manifest. The patch set is chosen with a fixed seed so it is reproducible and
documented.

Usage
-----
    python scripts/label_masks.py                 # label interactively
    python scripts/label_masks.py --list          # just print the chosen patches
    python scripts/label_masks.py --n 18 --seed 7 # choose a different set
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eurosat_ms.data import default_data_root, load_manifest, load_patch


def choose_patches(manifest_dir, n, seed):
    """Stratified pick from the River and SeaLake test patches (reproducible)."""
    test = load_manifest(manifest_dir, "test")
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in ("River", "SeaLake"):
        pool = test[test["class_name"] == cls]["path"].tolist()
        k = n // 2
        idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        chosen += [pool[i] for i in sorted(idx)]
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default="data/seg_labels")
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--list", action="store_true", help="print chosen patches and exit")
    args = ap.parse_args()

    patches = choose_patches(args.manifest_dir, args.n, args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "labels.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "seed": args.seed, "n_requested": args.n, "patches": patches, "labelled": {}}

    if args.list:
        for p in patches:
            done = "done" if p in manifest["labelled"] else "todo"
            print(f"  [{done}] {p}")
        return

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MplPath
    from matplotlib.widgets import PolygonSelector
    from eurosat_ms.visualize import true_color, index_map  # forces 'Agg' on import
    # visualize.py selects the headless 'Agg' backend (right for figure files, but
    # it gives no window). Restore an interactive backend so the labeller displays.
    for _backend in ("macosx", "qtagg", "tkagg"):
        try:
            matplotlib.use(_backend, force=True)
            break
        except Exception:
            continue
    else:
        raise SystemExit(
            "No interactive matplotlib backend found. Install one (pip install PyQt5) "
            "or use the no-GUI subjective evaluation instead."
        )

    data_root = Path(args.data_root) if args.data_root else default_data_root()
    ys, xs = np.mgrid[0:64, 0:64]
    grid = np.vstack([xs.ravel(), ys.ravel()]).T  # (4096, 2) as (x, y)

    todo = [p for p in patches if p not in manifest["labelled"]]
    print(f"{len(manifest['labelled'])} already labelled; {len(todo)} to go.")

    for rel in todo:
        img = load_patch(data_root / rel)
        fig, (axl, axr) = plt.subplots(1, 2, figsize=(10, 5))
        axl.imshow(true_color(img)); axl.set_title(f"{rel}\nclick water polygon vertices")
        axr.imshow(index_map(img, "NDWI"), cmap="BrBG_r", vmin=-1, vmax=1); axr.set_title("NDWI")
        for a in (axl, axr):
            a.set_xlim(-0.5, 63.5); a.set_ylim(63.5, -0.5)
        state = {"verts": []}

        def on_select(verts):
            state["verts"] = verts

        selector = PolygonSelector(axl, on_select)

        def rasterize():
            if len(state["verts"]) < 3:
                return np.zeros((64, 64), np.uint8)
            inside = MplPath(state["verts"]).contains_points(grid).reshape(64, 64)
            return inside.astype(np.uint8)

        def on_key(event):
            if event.key in ("n", "enter"):
                mask = rasterize()
                name = rel.replace("/", "_").replace(".tif", "")
                np.save(out_dir / f"{name}.npy", mask)
                manifest["labelled"][rel] = {"mask": f"{name}.npy", "water_px": int(mask.sum())}
                manifest_path.write_text(json.dumps(manifest, indent=2))
                print(f"  saved {name}.npy ({int(mask.sum())} water px)")
                plt.close(fig)
            elif event.key == "c":
                selector.disconnect_events(); state["verts"] = []
                print("  cleared — redraw the polygon")
            elif event.key == "s":
                print(f"  skipped {rel}"); plt.close(fig)
            elif event.key == "q":
                manifest_path.write_text(json.dumps(manifest, indent=2))
                print("  quit — progress saved"); plt.close(fig); raise SystemExit

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.tight_layout(); plt.show()

    print(f"Done. {len(manifest['labelled'])} masks in {out_dir}.")


if __name__ == "__main__":
    main()
