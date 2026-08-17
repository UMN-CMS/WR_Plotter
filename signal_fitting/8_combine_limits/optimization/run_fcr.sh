#!/usr/bin/env bash
# Flavor-CR variant, step 3 -- build the CR-anchored workspaces and run
# AsymptoticLimits over them (re-execs itself under apptainer if needed).
#   ./run_fcr.sh                # all masses in manifest_fcr.txt
#   ./run_fcr.sh 2000           # only this mass
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
ONLY_MASS="${1:-}"

python3 make_fcr_workspaces.py

RES="$HERE/results/$TAG"
mkdir -p "$RES"
cd "$RES"
while IFS=$'\t' read -r card mass label; do
    [[ -n "$ONLY_MASS" && "$mass" != "$ONLY_MASS" ]] && continue
    echo ">>> m=$mass $label"
    combine -M AsymptoticLimits "$card" -m "$mass" -n "_${label}" --rMax 20 \
        | grep -E "Expected 50" || true
done < "$HERE/cards/$TAG/manifest_fcr.txt"
echo "Done. Results in $RES"
