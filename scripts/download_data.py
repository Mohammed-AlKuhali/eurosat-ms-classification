"""Download and verify the EuroSAT multispectral dataset.

The phelber/EuroSAT GitHub repository hosts no data; it points to Zenodo
record 7711810, which is the official source used here. The checksum is
verified before unzipping so the reported results are provably tied to the
canonical dataset.

Usage
-----
    python scripts/download_data.py --root data/raw
    python scripts/download_data.py --root data/raw --cache /content/drive/MyDrive/eurosat_cache
    python scripts/download_data.py --root data/raw --subset 100   # 7.7 MB smoke set

On Colab, pass --cache pointing at a mounted Drive folder: the zip is copied
there after the first download and reused on subsequent sessions.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

# (filename, url, md5, extracted_dirname)
SOURCES = {
    "ms": (
        "EuroSAT_MS.zip",
        "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1",
        "091174add3c8e680a49244acf185b9f0",
        "EuroSAT_MS",
    ),
    "100": (
        "EuroSAT100.zip",
        "https://hf.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT100.zip",
        "c21c649ba747e86eda813407ef17d596",
        "EuroSAT100",
    ),
}


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/raw", help="extraction root")
    ap.add_argument("--subset", choices=["ms", "100"], default="ms",
                    help="'ms' = full 13-band set; '100' = 100-image smoke set")
    ap.add_argument("--cache", default=None, help="optional zip cache dir (e.g. Drive)")
    args = ap.parse_args()

    fname, url, md5, extracted = SOURCES[args.subset]
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    target_dir = root / extracted
    if target_dir.is_dir() and any(target_dir.rglob("*.tif")):
        print(f"[skip] {target_dir} already populated")
        return

    cache = Path(args.cache) if args.cache else root
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / fname

    if zip_path.exists() and md5sum(zip_path) == md5:
        print(f"[cache] using verified {zip_path}")
    else:
        fetch(url, zip_path)
        got = md5sum(zip_path)
        if got != md5:
            raise SystemExit(f"md5 mismatch for {fname}: expected {md5}, got {got}")
        print(f"[ok] md5 verified: {md5}")

    print(f"  extracting to {root}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    print(f"[done] dataset ready at {target_dir}")


if __name__ == "__main__":
    main()
