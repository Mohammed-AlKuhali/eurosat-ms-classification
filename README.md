# EuroSAT: RGB vs multispectral land-use classification

Does information beyond the visible RGB bands improve land-use classification on
Sentinel-2 imagery? This repository builds an RGB baseline and several
multispectral approaches on the [EuroSAT](https://github.com/phelber/EuroSAT)
dataset, compares them under an identical 80/20 split, and analyses *where and
why* spectral information helps — not just headline accuracy.

**Full report:** [`report/report.pdf`](report/report.pdf) (LaTeX source: `report/report.tex`).

## Headline result

Spectral information beyond RGB helps **conditionally** — strongly for classical
features and for in-domain-pretrained / from-scratch networks, but negligibly (or
harmfully) for ImageNet-pretrained networks, especially with few labels.

| Comparison | RGB | Multispectral | Δ |
|---|---|---|---|
| Classical RF (C1 vs C2) | 79.3% | **91.4%** | **+12.2 pp** (McNemar p≈7e-106) |
| ResNet-18 from scratch (E5a vs E5b) | 96.5% | **98.2%** | +1.7 pp |
| SSL4EO-pretrained (E7b vs E7) | 94.4% | **97.2%** | +2.8 pp |
| ImageNet-pretrained, full data (E1 vs E2) | **98.4%** | 98.3% | ~0 (n.s.) |
| ImageNet-pretrained, 1% labels | **81.4%** | 74.7% | −6.7 pp |

Best overall: **E3 (RGB + NDVI/NDWI/NDBI/NDRE), 98.69 ± 0.07%**. Water-segmentation
extension: NDWI + Otsu, micro-IoU 0.51 / Dice 0.68 on 11 hand-labelled patches.

## Quickstart

```bash
# 1. Environment (Python 3.12)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .                      # data layer + classical baseline
pip install -r requirements.txt       # adds torch/torchgeo/timm for the CNN arms

# 2. Data — downloads from Zenodo and verifies the md5 checksum
python scripts/download_data.py --root data/raw
export EUROSAT_DATA_ROOT=data/raw/EuroSAT_MS

# 3. Reproduce the committed split manifests + normalisation stats
python scripts/make_manifests.py

# 4. Run the test suite (band-order, indices, split integrity)
pytest

# 5. Reproduce the Tier-1 classical results only (fast, CPU, no torch needed)
python scripts/run_classical.py
python scripts/analyze.py

# 6. Reproduce everything (full CNN matrix + all tables/figures) — needs a GPU
DEVICE=mps bash scripts/run_all.sh    # DEVICE=cuda on NVIDIA; ~5-7 GPU-hours
```

If you already have `EuroSAT_MS/` locally, point `EUROSAT_DATA_ROOT` at it and
skip step 2. `run_all.sh` is idempotent: completed runs skip themselves, and it
ends by rebuilding every table and figure under `results/`.

## Repository layout

```
src/eurosat_ms/      core package (bands, data, features, models, train, evaluate)
scripts/             download_data.py, make_manifests.py
configs/             one YAML per experiment arm
data/manifests/      committed stratified train/val/test split (CSV)
data/stats/          committed per-band train normalisation statistics
tests/               band-order + index + split-integrity guards
docs/decisions.md    design decision log
colab/               thin driver notebook for GPU runs
results/             committed metrics CSVs and figures
report/              the written report (<=2000 words)
```

## Reproducibility notes

- Reported numbers were produced locally on Apple Silicon (M4, PyTorch MPS). The
  data layer and classical baseline run on CPU anywhere.
- Splits and statistics are committed, so results are auditable without
  re-downloading the 2 GB dataset.
- Seeds are fixed; bitwise determinism across hardware is not guaranteed, so
  per-seed metrics are committed under `results/`.

## Dataset & citation

EuroSAT (Sentinel-2, 27,000 patches, 10 classes, 13 bands), MIT-licensed under
Copernicus Sentinel data terms.

> Helber, P., Bischke, B., Dengel, A., Borth, D. *EuroSAT: A Novel Dataset and
> Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE
> JSTARS, 2019.
