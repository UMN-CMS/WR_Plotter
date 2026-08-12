# Stage 10.8 — refined run2 (2018 ee) expected limit, combine-only

> **STATUS: PRODUCTION (config C)** — the quoted run2 limit: `refined_limit_table_ee_resolved.csv` + `plots/ee_resolved/refined.*`. (Previously `10_limit_refinement/8_run2_refined_limit/`.)

Applies the Stage-10 findings to the run2 2018 ee limit using **nothing but
combine cards** — no homemade toy machinery. The Stage-9 run2 baseline
(floating background + AsymptoticLimits everywhere) has two known invalid
regions, mapped by 10.1/10.6 and confirmed on the run2 background:

| regime | masses | B_window | problem | fix here |
|---|---|---|---|---|
| anch_low | 1000–1200 | ~900 | window left-clamped at the 800 GeV threshold → no left sideband, floating background collinear with the signal | background **anchored** (fixed) to a trusted-spectrum fit; AsymptoticLimits (B is large, asymptotics fine) |
| float | 1400–3200 | 37–1036 (k5) | the Stage-9 k3/floating card was *valid* (10.6 parity 0.90–1.04) but far from optimal | the **Stage-10.9 optimized card**: k5 window, 50 GeV bins, expo slope `param`-constrained to the anchor (norm free), signal µ and σ `param`-constrained at 0.3σ₀ ('both030'); 0.44–0.85× the baseline, toy-validated in `../optimization/` |
| anch_sparse | ≥ 3400 | < 7 | free background norm runs away in near-empty windows **and** AsymptoticLimits under-covers at a few events (10.6: 2.3 vs HybridNew 3.25 at 4600) | anchored background + **HybridNew** expected quantiles (toys); signal shape stays fixed here |

Float-regime notes (Jul 2026 promotion of the 10.9 winner): windows and
efficiencies come from `../optimization/inputs` (`k5_bw50`);
1400–1800 adopt the winner by extrapolation (the scan covered 2000–3200) and
their k5 windows clamp against the 800 GeV floor, which is why the gain
shrinks toward 1400 (0.76–0.85× vs 0.40–0.69× at 2000–3200). A small seam at
the 3200→3400 regime boundary (0.197 float vs 0.234 anchored-HybridNew) is
expected — the background treatment and method change there, and the 3400
HybridNew quantiles carry toy noise.

## The anchor

Binned Poisson-ML fits (`TH1::Fit "L"`) of the summed run2 background MC,
done inside `make_refined_workspaces.py` (no dependency on the run3-era 10.4
outputs), pivot 2000 GeV:

| member | function | range | role |
|---|---|---|---|
| central | expo | [1000, 3500] | the quoted card |
| tail | expo | [1000, 6000] | model spread |
| expo2 | expo + curvature | [1000, 3500] | model spread |
| powexp | power law × expo | [1000, 3500] | model spread |

At each anchored mass the member is transported into the Stage-6 window
(`B_env` = Σ f(bin centre)) and fixed — only r floats. Model spread =
max |asymptotic-median shift| across members, reported as a column and drawn
as grey bars (it belongs ON the band, not inside the CLs σ — the 10.4 rule).

Signal rate = lumi × eff (Stage-9 run2 inputs) → **r = σ·B(eeqq̄′) in fb**.

## Run

```bash
./run_refined.sh ee resolved 500        # container; HybridNew toys/point = 500
# then, LCG_106:
python collect_refined.py -v
```

Outputs: `refined_limit_table_ee_resolved.csv`,
`plots/ee_resolved/refined.*` (official-style band) and
`refined_vs_baseline.*` (vs the Stage-9 run2 baseline, with ratio panel).

## Caveats

- Expected-limit study: the observation is the MC Asimov everywhere; anchored
  cards have no background freedom, so the observed column is only a spurious
  diagnostic there.
- The anchor is fit to the same MC that plays data — the model spread grey
  bars, not the width of the band, carry the "is expo the right family"
  uncertainty.
- Signal shape: µ and σ are `param`-constrained at 0.3σ₀ ('both030') on the
  float cards **up to 2600 GeV only** (`SIG_PRIOR_MAX`) — the 10.9 retest
  showed the priors are nearly free (+3.5%) once the slope is constrained
  (the 10.6 "+35% cost" was the floating-background degeneracy), and the
  toy re-validation passes there (median spurious < 3% of the limit, core
  width +10%). At 2800–3200 the two shape nuisances collapse toy convergence
  (84/51/32% vs 99/88/66% fixed) and drift the survivor median while
  *costing* limit, so those cards keep the fixed shape; same for the
  anchored regimes (untested + HybridNew toy cost). The width variation at
  ≥2800 stays an offline bias systematic.
