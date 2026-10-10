#!/usr/bin/env bash
# MC-only comparison cards (make_mc_cards.py, TH1 shapes): AsymptoticLimits
# over the manifest. Run inside the combine container:
#   FCR_CARDS=cards_2024_mconly FCR_RESULTS=results_2024_mconly ./run_mc.sh [mass]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="ee_resolved"
ONLY_MASS="${1:-}"
CARDS="${FCR_CARDS:-cards_2024_mconly}"
RESULTS="${FCR_RESULTS:-results_2024_mconly}"
RES="$HERE/$RESULTS/$TAG"
mkdir -p "$RES"
while IFS=$'\t' read -r card mass label; do
    [[ -n "$ONLY_MASS" && "$mass" != "$ONLY_MASS" ]] && continue
    echo ">>> m=$mass $label"
    [[ "$card" = /* ]] || card="$HERE/../../../$card"      # manifest paths are repo-relative
    cdir="$(dirname "$card")"
    ( cd "$cdir" && combine -M AsymptoticLimits "$(basename "$card")" -m "$mass" -n "_${label}" --rMax 50 2>&1 \
        | grep -E "Expected 50|rror|ERROR|Traceback" ) || true
    mv -f "$cdir/higgsCombine_${label}.AsymptoticLimits.mH${mass}.root" "$RES/" 2>/dev/null || echo "   (no output for m=$mass)"
done < "$HERE/$CARDS/$TAG/manifest_mc.txt"
echo "Done. Results in $RES"
