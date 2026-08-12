#!/usr/bin/env bash
# Stage 10.9, step 5 -- (a) the signal-shape constraint retest on the winner
# configuration, and (b) the FitDiagnostics toy validation of the winner.
# Re-execs itself under apptainer if not already inside.
#
#   ./run_sigscan_and_toys.sh [ntoys_null] [ntoys_inj]
#
# (a) AsymptoticLimits for the sig-mode variant cards (mu/sigma param-
#     constrained on top of k5_bw50_bconstr, + the baseline sig030 point).
# (b) toys, all on card configurations that already exist:
#     null (expectSignal 0): winner at every mass + baseline at 3 masses
#       -> spurious r-hat distribution, pull width (coverage)
#     injection (expectSignal ~ the median expected limit): winner at 3
#       masses -> recovery of an injected signal at the sensitivity edge
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:latest

if [[ -z "${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-}" ]]; then
    exec apptainer exec -B /uscms_data -B /cvmfs "$IMG" \
        /bin/bash "${BASH_SOURCE[0]}" "$@"
fi

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /home/cmsusr/CMSSW_* && eval "$(scramv1 runtime -sh)" && cd "$HERE"
NT_NULL="${1:-500}"; NT_INJ="${2:-300}"
TAG="ee_resolved"
WINNER="k5_bw50_bconstr"
BASE="k3_bw100_float"

python3 make_opt_workspaces.py

RES="$HERE/results/$TAG"
mkdir -p "$RES"
cd "$RES"

# (a) asymptotics for the new signal-mode cards only
grep -E "sig030|sig015|mu030|mu010|both030" "$HERE/cards/$TAG/manifest.txt" \
| while IFS=$'\t' read -r card mass label; do
    echo ">>> asymptotic m=$mass $label"
    combine -M AsymptoticLimits "$card" -m "$mass" -n "_${label}" --rMax 20 \
        | grep -E "Expected 50" || true
done

# NB: FitDiagnostics output names do NOT include -m, so the mass must be part
# of -n or the per-mass files overwrite each other.
# (b1) null toys: winner at every mass, baseline at three
for m in 2000 2200 2400 2600 2800 3000 3200; do
    echo ">>> null toys m=$m $WINNER"
    combine -M FitDiagnostics "$HERE/cards/$TAG/card_${WINNER}_m${m}.txt" \
        -m "$m" -n "_null_${WINNER}_m${m}" -t "$NT_NULL" --expectSignal 0 \
        -s 12345 --rMin -20 --rMax 20 2>/dev/null | grep -E "Best fit" || true
done
for m in 2000 2600 3200; do
    echo ">>> null toys m=$m $BASE"
    combine -M FitDiagnostics "$HERE/cards/$TAG/card_${BASE}_m${m}.txt" \
        -m "$m" -n "_null_${BASE}_m${m}" -t "$NT_NULL" --expectSignal 0 \
        -s 12345 --rMin -20 --rMax 20 2>/dev/null | grep -E "Best fit" || true
done

# (b2) injection toys: winner, r_inj ~ the median expected limit
declare -A RINJ=( [2000]=0.78 [2600]=0.37 [3200]=0.19 )
for m in 2000 2600 3200; do
    r="${RINJ[$m]}"
    echo ">>> injection toys m=$m $WINNER r_inj=$r"
    combine -M FitDiagnostics "$HERE/cards/$TAG/card_${WINNER}_m${m}.txt" \
        -m "$m" -n "_inj_${WINNER}_m${m}" -t "$NT_INJ" --expectSignal "$r" \
        -s 54321 --rMin -20 --rMax 20 2>/dev/null | grep -E "Best fit" || true
done
echo "Done. Results in $RES"
