#!/usr/bin/env bash
# Stage 10.8, step 2 -- build the refined run2 workspaces and run combine.
# Re-execs itself under apptainer if not already inside.
#
#   ./run_refined.sh [channel] [topology] [hybrid_toys]
#   ./run_refined.sh ee resolved 500
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
    exec apptainer exec -B /uscms_data -B /cvmfs "$IMG" \
        /bin/bash "${BASH_SOURCE[0]}" "$@"
fi

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /home/cmsusr/CMSSW_* && eval "$(scramv1 runtime -sh)" && cd "$HERE"

CH="${1:-ee}"; TOPO="${2:-resolved}"; NTOYS="${3:-500}"
TAG="${CH}_${TOPO}"

python3 make_refined_workspaces.py --channel "$CH" --topology "$TOPO"

RES="$HERE/results/$TAG"
mkdir -p "$RES"
cd "$RES"
while IFS=$'\t' read -r card mass variant regime rmax; do
    echo ">>> m=$mass $variant ($regime, rMax=$rmax)"
    combine -M AsymptoticLimits "$card" -m "$mass" -n "_${variant}" \
        --rMax "$rmax" | grep -E "Observed|Expected" || true
    if [[ "$regime" == anch_sparse && "$variant" == anchored ]]; then
        for q in 0.025 0.16 0.5 0.84 0.975; do
            echo "    HybridNew expected q=$q (${NTOYS} toys/point)"
            combine -M HybridNew "$card" -m "$mass" -n "_${variant}" \
                --LHCmode LHC-limits --expectedFromGrid="$q" \
                -T "$NTOYS" --rMax "$rmax" \
                | grep -E "Limit:" || true
        done
    fi
done < "$HERE/cards/$TAG/manifest.txt"
echo "Done. Results in $RES"
