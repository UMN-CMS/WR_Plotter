#!/usr/bin/env bash
# Stage 10.9, step 3 -- run AsymptoticLimits over the variant cards.
# Re-execs itself under apptainer if not already inside.
#   ./run_opt.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:latest

if [[ -z "${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-}" ]]; then
    exec apptainer exec -B /uscms_data -B /cvmfs "$IMG" \
        /bin/bash "${BASH_SOURCE[0]}" "$@"
fi

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /home/cmsusr/CMSSW_* && eval "$(scramv1 runtime -sh)" && cd "$HERE"
TAG="ee_resolved"

python3 make_opt_workspaces.py

RES="$HERE/results/$TAG"
mkdir -p "$RES"
cd "$RES"
while IFS=$'\t' read -r card mass label; do
    echo ">>> m=$mass $label"
    combine -M AsymptoticLimits "$card" -m "$mass" -n "_${label}" --rMax 20 \
        | grep -E "Expected 50" || true
done < "$HERE/cards/$TAG/manifest.txt"
echo "Done. Results in $RES"
