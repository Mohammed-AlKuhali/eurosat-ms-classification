"""Sentinel-2 band definitions for the EuroSAT multispectral GeoTIFFs.

CRITICAL — band order
=====================
EuroSAT_MS ``.tif`` files store their 13 bands in this channel order::

    index:  0    1    2    3    4    5    6    7    8    9    10   11   12
    band:   B01  B02  B03  B04  B05  B06  B07  B08  B09  B10  B11  B12  B8A

i.e. **B8A is the LAST channel (index 12), not in Sentinel-2 "SAFE" order**
(where B8A sits between B08 and B09). This is a well-known footgun that
silently corrupts every spectral index if you index bands positionally.

This was verified empirically on the actual download: channel 9 (B10,
cirrus) has a per-image mean of ~8-12 digital numbers (near-empty), whereas
if the order were SAFE, channel 9 would be B8A/NIR with a mean of ~3000.
Every consumer in this codebase therefore looks bands up *by name* through
:data:`BAND_INDEX`, never by a hard-coded integer.

References
----------
Helber et al., "EuroSAT: A Novel Dataset and Deep Learning Benchmark for
Land Use and Land Cover Classification", IEEE JSTARS 2019.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# File channel order (0-indexed) exactly as stored in EuroSAT_MS .tif files.
# ---------------------------------------------------------------------------
BAND_ORDER: list[str] = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B09", "B10", "B11", "B12", "B8A",
]
BAND_INDEX: dict[str, int] = {name: i for i, name in enumerate(BAND_ORDER)}
N_BANDS = len(BAND_ORDER)  # 13

# True-colour RGB channels (red, green, blue).
RGB_BANDS: list[str] = ["B04", "B03", "B02"]

# Near-infrared band (used by the RGB+NIR ablation, arm E4).
NIR_BAND = "B08"

# Cirrus band — near-empty over EuroSAT's cloud-screened L1C patches.
CIRRUS_BAND = "B10"

# 12-band "multispectral" view (B10 excluded) — the E2 arm input.
BANDS_12: list[str] = [b for b in BAND_ORDER if b != CIRRUS_BAND]

# ---------------------------------------------------------------------------
# SSL4EO-S12 / torchgeo Sentinel-2 weights expect a different 13-band order,
# with B8A in the 9th position (after B08, before B09):
#     B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12
# To feed a file-order tensor to those weights, reindex with SSL4EO_REORDER.
# ---------------------------------------------------------------------------
SSL4EO_ORDER: list[str] = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]
# SSL4EO_REORDER[i] = file-order channel that belongs at SSL4EO position i.
SSL4EO_REORDER: list[int] = [BAND_INDEX[b] for b in SSL4EO_ORDER]

# ---------------------------------------------------------------------------
# Physical meaning and native resolution of each band, for the report's
# "understanding of band information" table. (description, native_res_m,
# land-cover relevance)
# ---------------------------------------------------------------------------
BAND_INFO: dict[str, tuple[str, int, str]] = {
    "B01": ("Coastal aerosol", 60, "atmosphere / haze; little surface signal"),
    "B02": ("Blue", 10, "true colour; water, built-up"),
    "B03": ("Green", 10, "true colour; vegetation vigour, water (NDWI)"),
    "B04": ("Red", 10, "true colour; chlorophyll absorption (NDVI)"),
    "B05": ("Red edge 1", 20, "crop/vegetation type (NDRE)"),
    "B06": ("Red edge 2", 20, "vegetation structure"),
    "B07": ("Red edge 3", 20, "vegetation structure"),
    "B08": ("NIR", 10, "biomass; vegetation vs water (NDVI, NDWI)"),
    "B09": ("Water vapour", 60, "atmosphere; little surface signal"),
    "B10": ("Cirrus", 60, "high cloud; near-empty over EuroSAT (excluded)"),
    "B11": ("SWIR 1", 20, "moisture, built-up (NDBI, MNDWI)"),
    "B12": ("SWIR 2", 20, "soil/mineral, burn; built-up vs bare"),
    "B8A": ("NIR narrow", 20, "vegetation; red-edge plateau"),
}

# Native spatial resolution (m) per band; bands coarser than 10 m were
# cubic-spline upsampled to the 10 m / 64x64 grid in EuroSAT (note in report).
BAND_RES_M: dict[str, int] = {b: BAND_INFO[b][1] for b in BAND_ORDER}


def select_indices(band_names: list[str]) -> list[int]:
    """Return file-order channel indices for a list of band names."""
    return [BAND_INDEX[b] for b in band_names]


def validate() -> None:
    """Internal consistency checks (also exercised by the test suite)."""
    assert len(BAND_ORDER) == 13, "EuroSAT MS has 13 bands"
    assert BAND_ORDER[-1] == "B8A", "B8A must be the last file channel"
    assert BAND_ORDER[9] == "B10", "B10 must be file channel index 9"
    assert len(set(BAND_ORDER)) == 13, "band names must be unique"
    assert sorted(SSL4EO_REORDER) == list(range(13)), "SSL4EO reorder must be a permutation"
    assert CIRRUS_BAND not in BANDS_12 and len(BANDS_12) == 12
    assert all(b in BAND_INFO for b in BAND_ORDER)
