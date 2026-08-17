# optimization — the float-card configuration scan + toy validation

(Previously `10_limit_refinement/9_float_region_optimization/`.) The evidence
behind the production float card: which combine configuration gives the best
*valid* expected limit at 1400–3200, all on the MC-Asimov (no-data) run2
background.

## Scan

| step | script | env |
|---|---|---|
| 1 | `prepare_opt.py` | LCG_106 — per-variant windows (k3/k4/k5/a53 × 100/50/20 GeV) with efficiencies recomputed per window/binning |
| 2 | `make_opt_workspaces.py` | container — the card matrix: background float / bconstr / bfixed / anch × signal fix / µ-, σ-, both-constrained |
| 3 | `run_opt.sh` | container — AsymptoticLimits over every card |
| 4 | `collect_opt.py` | LCG_106 — `opt_table_ee_resolved.csv` + `opt_ratios.*`, ranked by geometric-mean median ratio to the k3_bw100_float baseline |

Result: **k5_bw50_bconstr** = 0.60× the baseline (window width ~0.74 and
anchor-slope constraint ~0.72 compound; binning alone does nothing,
1.015–1.035); fully-anchored bound 0.59. Signal-shape retest: the 0.3σ₀
priors cost +3.5% here vs +22% on the floating-background card — the old
10.6 cost was the signal↔background degeneracy.

## Toy validation

`run_sigscan_and_toys.sh` (container) + `collect_toys.py` (LCG) →
`toy_validation_table.csv`: FitDiagnostics null toys (500/mass; the toys
sample the slope constraint) + injections at the sensitivity edge (300/mass).
The winner passes everywhere (null spurious < 2% of the medians, injection
recovered 1–5%, pull RMS ≈ 1). The **both030** re-validation passes at
1400–2600 but collapses toy convergence at 2800–3200 (84/51/32% vs 99/88/66%)
— which is why the production card applies the shape priors only up to 2600
(`SIG_PRIOR_MAX` in `../production/make_refined_workspaces.py`).

Gotchas: FitDiagnostics output names don't include `-m` (put the mass in
`-n`); cards whose only nuisances are `flatParam` need `--toysNoSystematics`
to generate toys.

## Flavor-CR anchored variant (`k5_bw50_fcr`, Aug 2026)

First bite of the CR-anchored background plan (defense priority 1): the summed
floating background is split into **tt+tW** (expo, slope fixed to the MC
component fit; norm = shared `mu_tt rateParam`) and **rest** (DY + Nonprompt +
Other, floating expo — the float-card treatment), plus a one-bin **flavor-CR
counting channel** (`shapes * fcr FAKE`) over the same mass window whose tt
rate carries the same `mu_tt`. Stat-only: no lnN on the SR/CR transfer factor.

  python prepare_fcr.py -v         # LCG: tt/rest split + CR yields + b_tt fits
  ./run_fcr.sh [mass]              # container: workspaces + AsymptoticLimits

Inputs: `inputs/ee_resolved_fcr.{root,json}` (b_tt stored there is the single
source of truth for both the workspace and the homemade card model). CR purity
80–93%, mu_tt constrained to ~5% at m=2000 (√N of the ~500-event CR window).
Result: **0.95–0.98× the float card at 1400–2200**, no gain ≥2400 (statistics
dominate); orthogonal to (and stackable with) the bconstr slope anchor. Main
value is defensive — the tt norm becomes CR-data-anchored instead of free,
addressing the step-11a LO-DY-tail risk for the tt half of the background.

### CR-data observation (`k5_bw50_fcrd`)

Same card with the fcr channel observing the unblinded flavor-CR DATA
(EGamma count; `--cr-dataset Muon` for the alternative) instead of the MC
Asimov — the SR observation stays MC (blind). At m=2000 the data anchors
mu_tt = (480 − 51.7)/452.6 = 0.946 ± 0.050. The EXPECTED limit is within
~1% of the all-MC fcr variant at every mass: the floating rest norm
re-absorbs the tt shift against the unchanged MC observation in the SR, so
the anchor's value is the data-backed normalization (defense), not expected
sensitivity. Homemade twin: `card_model_toys.py --cr-obs EGamma` (rows
`card_fcrd`, outputs `run2/card_fcrd/`) with the toy truth made consistent
(SR tt scaled by the measured mu_tt); homemade/combine parity 0.90–1.03
over 1400–2800.
