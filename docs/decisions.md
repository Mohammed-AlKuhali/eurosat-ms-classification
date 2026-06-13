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

## Modelling — Tier 1 (classical)

**D7 — Random Forest on per-band + index statistics as the interpretable tier.**
Two arms on the committed split: C1 (RGB band statistics, 15 features) and C2
(12-band statistics + NDVI/NDWI/NDBI/NDRE statistics, 68 features). RF needs no
feature scaling, handles mixed-scale features, and gives permutation importance
for free — directly serving the brief's "feature extraction" and band-understanding
criteria.

**Result (Tier-1, 5,400-image test):** C1 RGB = 79.3% acc / 78.5% macro-F1;
C2 multispectral = 91.4% acc / 91.1% macro-F1 — a **+12.2 pp** gain from spectral
information. Top permutation-importance features include the NDWI and NDRE
indices, confirming the engineered indices carry real signal. *Interview framing:*
this large Tier-1 gain contrasts with the small/near-zero gain expected for
ImageNet-pretrained CNNs at full data — evidence that spectral information matters
most when there is no powerful pretrained RGB representation to lean on, which is
the same mechanism behind the data-efficiency hypothesis (E6).

## Modelling — Tier 2 (CNN)

**D8 — ResNet18 backbone, one frozen recipe across all arms.**
ResNet18 (not ResNet50/ViT) because at 64×64 it fine-tunes in minutes, letting us
afford 8 arms × 3 seeds + a data-efficiency curve; the ~0.3–0.9 pp a bigger
backbone would add is worth far less than that breadth, and the brief grades
analysis over raw accuracy. One recipe — AdamW (lr 1e-3, wd 1e-4), cosine
schedule, batch 64, ≤30 epochs, early stop (patience 7) on val macro-F1,
flips+90° augmentation, 64×64 — is selected once on E1's val slice and frozen for
every arm, so arms differ only in inputs and pretraining, never in tuning.
(Planned: an LR sanity sweep on E2's val, run at GPU time, to confirm the MS arm
isn't disadvantaged by RGB-tuned settings.)

**D9 — Conv1 adaptation via timm `in_chans=N`, not hand-rolled surgery.**
timm repeats the pretrained RGB first-conv weights cyclically to N channels and
rescales by 3/N to preserve activation magnitude (the I3D "inflation" recipe).
Citable and bug-free; replication is known to beat random-init for extra channels.

**D10 — Augmentation is flips + 90° rotations only (no photometric jitter).**
Colour/brightness jitter would corrupt the physical meaning of reflectance values
and the spectral indices derived from them — unacceptable for a study whose whole
point is spectral information. The dihedral group is label- and spectrum-preserving
for nadir satellite patches.

**D11 — Add SSL4EO-S12 arms (E7 all-band, E7b RGB) to complete a 2×2.**
With only ImageNet (RGB-domain) pretraining, a "MS ties/loses to RGB" result at
full data could be dismissed as a pretraining artifact. Adding torchgeo's
SSL4EO-S12 ResNet18 weights — 13-band (E7) and RGB (E7b), both Sentinel-2-domain —
gives a {ImageNet, SSL4EO} × {RGB, all-band} 2×2 that separates *pretraining
domain* from *spectral information*. Verified: weights load with only the
classification head fresh. SSL4EO arms keep all 13 bands in SSL4EO order (B8A
9th) and use DN/10000 normalisation to match their pretraining distribution —
normalisation is therefore a per-arm property, not global.

## Extension — water segmentation

**D12 — NDWI/MNDWI thresholding vs a small hand-labelled set; pixel-pooled IoU/Dice.**
EuroSAT has no pixel ground truth, so (as the brief suggests) we hand-label ~18
River/SeaLake patches and evaluate index thresholding against them. We compare
NDWI (B03,B08) vs MNDWI (B03,B11) and fixed-threshold vs Otsu, and report a
pixel-pooled micro-average alongside per-patch spread — the honest aggregate when
water area varies from a thin river to a full lake. Framed as illustrative and
resolution-limited (a river is ~3 px wide at 10 m/px); the point is to show, with
a metric, that NIR/SWIR bands localise water where RGB alone cannot. Sample set
chosen by fixed seed and documented. Citations: McFeeters 1996, Xu 2006, Otsu 1979.

## Reproducibility / pipeline robustness

**D13 — Smoke runs are isolated; skip-if-done validates completeness (not mere existence).**
A bug surfaced during the first full run: the RGB baseline (E1) reported ~27%
accuracy while the multispectral arm hit 98%. Root cause (found by systematic
debugging, not guesswork) was *not* the model — a prior `--smoke` sanity run
(1 epoch, 64 images) had written E1 metrics into the canonical `results/` dir
under the same run-id, and the idempotent skip-if-done logic, which only checked
*file existence*, then skipped E1's real training. Fixed at the source with
defense-in-depth: (1) smoke runs now write to an isolated `results/_smoke/` dir
so they can never overwrite real results; (2) a run counts as "complete" only if
its metrics cover the full test set (`n == |test|`) and carry `smoke=False` — a
leftover smoke result (n=32) no longer masquerades as done. Guarded by unit tests
(`tests/test_run_completeness.py`). *Interview framing:* the diagnostic signal was
that the on-disk "result" had n=32 (=15/32 acc) not n=5400 — committing per-sample
prediction CSVs and an `n` field made the contamination detectable in seconds. The
RGB and multispectral arms differ only in input bands, never in correctness.
