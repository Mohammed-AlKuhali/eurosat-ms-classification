# EuroSAT: RGB vs multispectral land-use classification

Does information beyond the visible RGB bands improve land-use classification on
Sentinel-2 imagery? This repository builds an RGB baseline and several
multispectral approaches on the [EuroSAT](https://github.com/phelber/EuroSAT)
dataset, compares them under an identical 80/20 split, and analyses *where and
why* spectral information helps — not just headline accuracy.

> Status: work in progress. Results tables and the full report are added as the
> experiments complete.

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
```

If you already have `EuroSAT_MS/` locally, point `EUROSAT_DATA_ROOT` at it and
skip step 2.

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

- Reported numbers were produced on Google Colab (NVIDIA T4). The data layer and
  classical baseline run on CPU anywhere.
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
