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
