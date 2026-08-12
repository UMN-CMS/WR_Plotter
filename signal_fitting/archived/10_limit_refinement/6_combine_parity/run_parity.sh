#!/usr/bin/env bash
# Stage 10.6, step 2 -- build the parity workspaces and run combine in the
# container. Re-execs itself under apptainer if not already inside (same
# recipe as 8_combine_limits/baseline/run_limits.sh).
#
#   ./run_parity.sh [channel] [topology] [hybrid_toys]
#   ./run_parity.sh ee resolved 500
#
# fit-regime cards  (fixed + prior)  -> AsymptoticLimits (fast)
# sparse cards      (anchored)       -> AsymptoticLimits + HybridNew expected
#                                       median & +-1sigma (toys; the reference
#                                       method where asymptotics break down)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:latest

if [[ -z "${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-}" ]]; then
    exec apptainer exec -B /uscms_data -B /cvmfs "$IMG" \
        /bin/bash "${BASH_SOURCE[0]}" "$@"
fi

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /home/cmsusr/CMSSW_14_1_0_pre4 && eval "$(scramv1 runtime -sh)" && cd "$HERE"

CH="${1:-ee}"; TOPO="${2:-resolved}"; NTOYS="${3:-500}"
TAG="${CH}_${TOPO}"

python3 make_workspaces_parity.py --channel "$CH" --topology "$TOPO"

RES="$HERE/results/$TAG"
mkdir -p "$RES"
cd "$RES"
while IFS=$'\t' read -r card mass variant regime rmax; do
    echo ">>> m=$mass $variant ($regime, rMax=$rmax)"
    combine -M AsymptoticLimits "$card" -m "$mass" -n "_${variant}" \
        --rMax "$rmax" | grep -E "Observed|Expected" || true
    if [[ "$regime" == sparse ]]; then
        for q in 0.16 0.5 0.84; do
            echo "    HybridNew expected q=$q (${NTOYS} toys/point)"
            combine -M HybridNew "$card" -m "$mass" -n "_${variant}" \
                --LHCmode LHC-limits --expectedFromGrid="$q" \
                -T "$NTOYS" --rMax "$rmax" \
                | grep -E "Limit:" || true
        done
    fi
done < "$HERE/cards/$TAG/manifest.txt"
echo "Done. Results in $RES"
