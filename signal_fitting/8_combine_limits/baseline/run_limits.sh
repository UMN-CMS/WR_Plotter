#!/usr/bin/env bash
# Stage 9, step 3 -- build workspaces and run AsymptoticLimits in the combine
# container. Re-execs itself under apptainer if not already inside.
#
#   ./run_limits.sh [channel] [topology] [functions] [masses] [run2|run3] [mc|data]
#   ./run_limits.sh ee resolved expo 2000                  # Phase-1 single point
#   ./run_limits.sh ee resolved expo,powlaw all            # full production (run3)
#   ./run_limits.sh ee resolved expo,powlaw all run2 data  # run2, real-data observed
#
# Outputs: {run}/results/{ch}_{topo}/higgsCombine_{fn}.AsymptoticLimits.mH{m}.root
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:latest

if [[ -z "${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-}" ]]; then
    exec apptainer exec -B /uscms_data -B /cvmfs "$IMG" \
        /bin/bash "${BASH_SOURCE[0]}" "$@"
fi

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /home/cmsusr/CMSSW_* && eval "$(scramv1 runtime -sh)" && cd "$HERE"

CH="${1:-ee}"; TOPO="${2:-resolved}"; FNS="${3:-expo,powlaw}"; MASSES="${4:-all}"
RUN="${5:-run3}"; OBS="${6:-mc}"
TAG="${CH}_${TOPO}"

MARGS=()
[[ "$MASSES" != all ]] && MARGS=(--masses ${MASSES//,/ })
python3 make_workspaces.py --channel "$CH" --topology "$TOPO" --run "$RUN" \
    --observed "$OBS" --functions ${FNS//,/ } "${MARGS[@]}"

RES="$HERE/$RUN/results/$TAG"
mkdir -p "$RES"
cd "$RES"
while IFS=$'\t' read -r card mass fn; do
    echo ">>> $fn m=$mass"
    combine -M AsymptoticLimits "$card" -m "$mass" -n "_${fn}" --rMax 50 \
        | grep -E "Observed|Expected" || true
done < "$HERE/$RUN/cards/$TAG/manifest.txt"
echo "Done. Results in $RES"
