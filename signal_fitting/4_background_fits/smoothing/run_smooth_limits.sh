#!/usr/bin/env bash
# Smoothing study, combine side: build the production cards
# (8_combine_limits/production/make_refined_workspaces.py) with a smoothed
# background and run combine on a mass subset, without touching the
# production cards/ and results/.
#
#   ./run_smooth_limits.sh <kernel|hybrid|param|kde|raw> [same|raw] [masses...]
#
#   arg 2 = --anchor-source: fit the anchors on the smoothed spectrum (same,
#           default) or on the raw MC (raw: isolates the data_obs effect;
#           anchored cards then reproduce production exactly)
#   raw   = regression check (production inputs through the same path)
#
# Outputs under run2/ee_resolved/: cards_<label>/ee_resolved/, results_<label>/
# Environment: MRW_EXTRA (extra make_refined_workspaces.py args) + LABEL_SUFFIX
# (e.g. MRW_EXTRA="--anchor-norm window" LABEL_SUFFIX=_normwin: anchored cards
# normalized to the smoothed window yield instead of B_env -- the 'mcmax' test),
# NTOYS (HybridNew toys/point, default 500), SKIP_HYBRID=1
# (asymptotic only), EXTRA_BINDS (apptainer binds; on the Mac OrbStack VM:
#   orb -m combine bash -c 'cd <this dir> && EXTRA_BINDS="-B /Users/billy/work/WrCoffea" SKIP_HYBRID=1 ./run_smooth_limits.sh kernel'
# on cmslpc just ./run_smooth_limits.sh kernel).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD="$(cd "$HERE/../../8_combine_limits/production" && pwd)"
M="${1:?method (kernel|hybrid|param|kde|raw)}"
ANCH="${2:-same}"
shift $(( $# >= 2 ? 2 : $# ))
MASSES="${*:-1000 2000 2600 3200 3400 4000 4600}"
OUT="$HERE/run2/ee_resolved"
LABEL="$M"
[[ "$ANCH" == raw && "$M" != raw ]] && LABEL="${M}_anchraw"
LABEL="${LABEL}${LABEL_SUFFIX:-}"      # e.g. LABEL_SUFFIX=_normwin with MRW_EXTRA="--anchor-norm window"
if [[ "$M" == raw ]]; then SRC="raw"; else SRC="smooth:$M"; fi
export MRW_ARGS="--bkg-source $SRC --anchor-source $ANCH --masses $MASSES --output-dir $OUT/cards_$LABEL ${MRW_EXTRA:-}"
export CARDS_DIR="$OUT/cards_$LABEL/ee_resolved"
export RES_DIR="$OUT/results_$LABEL/ee_resolved"
echo ">>> $LABEL: $MRW_ARGS"
cd "$PROD" && ./run_refined.sh ee resolved "${NTOYS:-500}"
