# The limits, in one page

**The run2 2018 ee result** = `8_combine_limits/production/`:
`refined_limit_table_ee_resolved.csv` + `plots/ee_resolved/refined.*`
(data-free expected σ·B(eeqq̄′), 1000–6000 GeV, expected exclusion ~5.04 TeV).

There are exactly **three limit configurations** in this repository; everything
else is validation of them or history.

| config | what | where | status |
|---|---|---|---|
| **A — homemade** | Stage-6 toys → closed-form CLs band (the original framework) | `7_limit_plots/` | frozen cross-check; reach ≤ 4 TeV (toys die in empty windows) |
| **B — combine baseline** | the same model as A translated into combine cards (k3 window, floating expo, fixed signal) | `8_combine_limits/baseline/` | reference point; deliberately un-optimized |
| **C — combine production** | regime-split cards: anchored (1000–1200, ≥3400, HybridNew for the sparse tail) + optimized float (1400–3200: k5 window, 50 GeV bins, anchor-constrained slope, µ/σ priors ≤2600) | `8_combine_limits/production/` | **the quoted result** |

## Directory map

| directory | role | status |
|---|---|---|
| `0_signal_samples/` … `6_spurious_signal_toys/` | the input chain: signal shapes → widths → windows → background plots/fits → spurious toys. Feed A, B and C alike (`run2/`\|`run3/` outputs) | active |
| `7_limit_plots/` | config A + the step2–5 statistics explainers | frozen cross-check |
| `8_combine_limits/baseline/` | config B (`run_limits.sh CH TOPO FNS MASSES RUN [mc\|data]`) | reference |
| `8_combine_limits/production/` | config C (`run_refined.sh`, `collect_refined.py`) | **production** |
| `8_combine_limits/optimization/` | the scan + FitDiagnostics toy validation that fixed C's float card | evidence for C |
| `8_combine_limits/explainer/` | 29-script step-by-step walkthrough of how C is built, data-free — steps 1–5 the inputs, 6–9 combine's internals rebuilt from scratch (absorbed `internals_explainer/`, was 10.7), 10–12 the production choices | documentation |
| `archived/10_limit_refinement/` 10.1–10.5 | the studies that fixed the homemade machinery (run3-era); conclusions absorbed into C; its README is the study log | archived |
| `archived/10_limit_refinement/6_combine_parity/` | proved combine ≡ homemade (0.90–1.04) in the trusted regime; asymptotics under-cover below B≈5 | historical (load-bearing conclusions) |
| `archived/8_signal_injection_study/` | injection tests for config A; superseded by the FitDiagnostics toys in `8_combine_limits/optimization/` | archived |

## Which table supersedes which (run2 ee resolved)

1. `8_combine_limits/production/refined_limit_table_ee_resolved.csv` — **quote this.**
2. `8_combine_limits/baseline/run2/combine_limit_table_ee_resolved.csv` — baseline (data-observed run); for method comparisons only.
3. `7_limit_plots/run2/xsec_limit_table_ee_resolved.csv` — homemade cross-check (expected-only, ≤ 4 TeV).

Conventions everywhere: no data in the expected bands (MC Asimov; the baseline
also has a data-observed variant), centre = zero, per-channel σ·B via
`bfrac = 0.5` (flavor-mixed signal samples), lumi 59.83 fb⁻¹.

History note (Jul 2026): `8_combine_limits/{baseline,production,optimization,
explainer}` were previously `9_combine_limits/`, `10_limit_refinement/
8_run2_refined_limit/`, `10_limit_refinement/9_float_region_optimization/`,
and `9_combine_limits/run2/no_data/` — older notes may use those paths.
Later in Jul 2026 `8_combine_limits/internals_explainer/` (was
`10_limit_refinement/7_combine_explainer/`) was merged into
`8_combine_limits/explainer/` as its steps 6–9 (analysis steps 6/7/8
renumbered to 10/11/12).
