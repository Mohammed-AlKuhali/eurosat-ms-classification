#!/usr/bin/env bash
# Run the full experiment matrix in priority order. Idempotent: completed
# arm/seed/fraction runs skip themselves, so a dropped Colab session resumes by
# simply re-running this script.
#
# Priority (per docs/decisions.md): the non-negotiable core first, so that if
# time/compute runs out, the brief's mandatory comparison is already complete.
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-}"
ARG=""
[ -n "$DEVICE" ] && ARG="--device $DEVICE"

run () { python scripts/run_experiment.py "$1" $ARG; }

echo "########## Tier 1 (classical) ##########"
python scripts/run_classical.py ${DATA_ROOT_ARG:-}

echo "########## CORE: RGB baseline + multispectral + indices ##########"
run configs/e1_rgb.yaml
run configs/e2_multispectral.yaml
run configs/e3_indices.yaml

echo "########## Pretraining-confound controls ##########"
run configs/e5a_rgb_scratch.yaml
run configs/e5b_ms_scratch.yaml

echo "########## Data-efficiency curve ##########"
run configs/e6_rgb.yaml
run configs/e6_ms.yaml

echo "########## SSL4EO 2x2 completion ##########"
run configs/e7_ssl4eo_all.yaml
run configs/e7b_ssl4eo_rgb.yaml

# E4 (RGB+NIR ablation) dropped from the default run: its question — does adding
# spectral info to RGB help — is answered more thoroughly by E2 (12-band) and E3
# (RGB+indices). Run manually if desired: run configs/e4_rgbnir.yaml
# (Documented as a scope cut in docs/decisions.md, D14.)

echo "All experiments complete. Metrics in results/metrics/."
