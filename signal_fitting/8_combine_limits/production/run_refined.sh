#!/usr/bin/env bash
# Stage 10.8, step 2 -- build the refined run2 workspaces and run combine.
# Re-execs itself under apptainer if not already inside.
#
#   ./run_refined.sh [channel] [topology] [hybrid_toys]
#   ./run_refined.sh ee resolved 500
#
# Environment overrides (smoothing study, 4_background_fits/smoothing):
#   MRW_ARGS    extra arguments for make_refined_workspaces.py
#               (e.g. "--bkg-source smooth:kernel --masses 2000 4600 --output-dir X")
#   CARDS_DIR   where the manifest is  (default cards/<tag>)
#   RES_DIR     where combine writes    (default results/<tag>)
#   SKIP_HYBRID=1  AsymptoticLimits only (no HybridNew toys in the sparse tail)
#   EXTRA_BINDS    extra apptainer binds (e.g. "-B /Users/billy/work/WrCoffea")
#
# float + anch_low cards        -> AsymptoticLimits (valid: B >= 12)
# anch_sparse cards             -> AsymptoticLimits (diagnostic) and, for the
#                                  CENTRAL member only, HybridNew expected
#                                  quantiles (the analysis-grade band at
#                                  B_window < 7 where asymptotics under-cover)
# model-spread members          -> AsymptoticLimits only
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:latest

if [[ -z "${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-}" ]]; then
    # shellcheck disable=SC2086
    exec apptainer exec -B /uscms_data -B /cvmfs ${EXTRA_BINDS:-} "$IMG" \
        /bin/bash "${BASH_SOURCE[0]}" "$@"
fi

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /home/cmsusr/CMSSW_* && eval "$(scramv1 runtime -sh)" && cd "$HERE"

CH="${1:-ee}"; TOPO="${2:-resolved}"; NTOYS="${3:-500}"
TAG="${CH}_${TOPO}"

# shellcheck disable=SC2086
python3 make_refined_workspaces.py --channel "$CH" --topology "$TOPO" ${MRW_ARGS:-}

CARDS="${CARDS_DIR:-$HERE/cards/$TAG}"
RES="${RES_DIR:-$HERE/results/$TAG}"
mkdir -p "$RES"
cd "$RES"
while IFS=$'\t' read -r card mass variant regime rmax; do
    echo ">>> m=$mass $variant ($regime, rMax=$rmax)"
    combine -M AsymptoticLimits "$card" -m "$mass" -n "_${variant}" \
        --rMax "$rmax" | grep -E "Observed|Expected" || true
    if [[ "$regime" == anch_sparse && "$variant" == anchored && -z "${SKIP_HYBRID:-}" ]]; then
        for q in 0.025 0.16 0.5 0.84 0.975; do
            echo "    HybridNew expected q=$q (${NTOYS} toys/point)"
            combine -M HybridNew "$card" -m "$mass" -n "_${variant}" \
                --LHCmode LHC-limits --expectedFromGrid="$q" \
                -T "$NTOYS" --rMax "$rmax" \
                | grep -E "Limit:" || true
        done
    fi
done < "$CARDS/manifest.txt"
echo "Done. Results in $RES"
