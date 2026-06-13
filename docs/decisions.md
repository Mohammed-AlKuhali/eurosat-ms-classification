# Decision log

Short, defensible rationale for each design choice — written as the work
happens so it can be explained later. One entry per decision.

## Data

**D1 — Use the all-bands (13-band) GeoTIFF set, not the JPEG RGB archive.**
The RGB baseline is built from bands B04/B03/B02 of the *same* GeoTIFFs as the
multispectral arms. Using the separate lossy JPEG RGB archive would introduce a
compression confound into the RGB-vs-MS comparison. Source: Zenodo record
7711810, `EuroSAT_MS.zip`, md5 `091174add3c8e680a49244acf185b9f0` (verified).

**D2 — Band order is B01..B12, B8A LAST; every access is by name.**
EuroSAT GeoTIFFs store B8A as the final channel (index 12), not in Sentinel-2
SAFE order. Indexing positionally silently corrupts NDVI/NDRE and the SSL4EO
arm. Guarded by a single `BAND_ORDER` constant + unit tests. Verified
empirically: channel 9 (B10) has train-set mean 12.1 DN; if the order were
SAFE, channel 9 would be NIR (~2300).

**D3 — Exclude B10 (cirrus) from the multispectral CNN/RF arms; label "12-band".**
B10 train-set mean 12.1, std ~5, vs means 410–3100 for surface bands — it
carries no land-cover signal over EuroSAT's cloud-screened L1C patches (matches
Helber Fig. 9, where B10 is the worst single band). The arm is labelled
"12-band (B10 excluded)", never "all bands". The SSL4EO-pretrained arm is
exempt — its weights expect all 13 channels in SSL4EO order.

**D4 — Stratified 80/20 train/test with a committed manifest; val carved from train.**
The brief mandates 80/20. We commit the split manifest (CSV of image IDs) so
every arm — classical and CNN — sees an identical split, which is what makes the
paired McNemar test valid. A 10%-of-train validation slice (→ 72/8/20 overall)
is held out for early stopping; the test set is touched once. This reproduces
Helber et al.'s own random class-wise 80/20 protocol with better reproducibility.

**D5 — Compute normalisation statistics on the train split only.**
Per-band z-score mean/std from train images only (leak-free), committed to
`data/stats/train_stats.json`. Normalisation is a *per-arm* property: arms we
train ourselves use this z-score; the SSL4EO arm instead uses DN/10000 to match
its pretraining distribution (recorded when that arm is built).

**D6 — Compute spectral indices from raw DN, before normalisation.**
Normalised-difference indices (NDVI/NDWI/NDBI/NDRE) are invariant to the ×10000
scale factor, so DN and reflectance give identical values; computing pre-norm
keeps them physically interpretable. EuroSAT predates the Jan-2022 baseline-04.00
offset, so no −1000 correction is applied. Division guarded by ε, output clipped
to [−1, 1] and asserted finite.
