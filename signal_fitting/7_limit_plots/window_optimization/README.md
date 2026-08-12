# Window-width optimization study

How the signal **efficiency** and the resulting **sensitivity** (expected
`sigma_UL`) depend on the window half-width `k` (window = `[m_c - k*sigma, m_c +
k*sigma]`, clamped to `[fit_min, fit_max]`). Answers: *is the current fixed k=3
leaving sensitivity on the table, and what k should we use?*

The window enters sensitivity twice, and both improve with wider `k` in the
reliable region:

    sigma_UL(m,k) = N_UL(m,k) / (1000 * L * eff(m,k))
      eff(m,k)  UP  with k  -- wider window captures more of the (tailed) signal
      N_UL(m,k) DOWN with k -- more sideband pins the background -> smaller RMS(N_sp)

...until the window is so wide the toy S+B fit destabilizes and `RMS(N_sp)` (the
band width) blows up. So there is a **reliability-capped optimum** per mass.

## Two parts

**Part A -- `efficiency_vs_k.py`** (cheap, no fitting)
Signal containment `f_win = S_win/S_tot` and `eff = S_win/(bfrac*sumw)` vs `k`,
per grid `m_WR` (m_N = m_WR/2), reusing `xsec_limit.signal_efficiency` and the
Stage-2 window params. Key finding: containment at k=3 is only ~90-95% (the
median-over-M_N `sigma` is narrower than the real signal), so widening recovers
+2.5..+8.6% signal; the low-mass clamp caps that gain.

**Part B -- `sensitivity_vs_k.py`**
Combines `eff(k)` (Part A) with `N_UL(k) = cls_band(0, RMS(N_sp), alpha)` from a
Stage-6 toy table at each `k`, giving `sigma_UL(m,k)`. Tags each `(m,k)`
reliable/unreliable from the Stage-6 CSV via THREE cuts -- convergence
(`n_ok/ntoys >= --conv-min`, survivor bias), runaway tail
(`RMS/q95 <= --tail-max`), and mismodeling **bias**
(`|mean(N_sp)|/RMS <= --bias-max`, the spurious-signal acceptance the centre-zero
band omits) -- then picks the k that minimizes `sigma_UL` among the reliable k.
Reports per-mass optimal `k(m)`, the gain vs k=3, and the reach.

## Running (ee resolved, run2)

    source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

    # Part A (seconds)
    python efficiency_vs_k.py --channel ee --topology resolved

    # Stage-6 toys at each k (the sweep Part B reads) -- ~1 min each
    for K in 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0; do
      ( cd ../../6_spurious_signal_toys && python spurious_signal_toys.py \
          --era RunIISummer20UL18 --dir 20260714_run2_bkgs \
          --channel ee --topology resolved --k $K --functions expo powlaw \
          --no-toy-plots --mass-max 4000 \
          --output-dir ../7_limit_plots/window_optimization/run2/stage6/k${K/.0/} )
    done

    # Part B (seconds)
    python sensitivity_vs_k.py --channel ee --topology resolved --functions expo powlaw

## Result (run2 ee-resolved, expo & powlaw agree; grid k=2.0..7.0)

The k=7 extension is the important test -- it shows the centre-zero `sigma_UL`
optimization is **gameable**:

- **Efficiency saturates at k=5** (~99% containment; k5->k7 adds <1% signal). No
  signal-side reason to widen past 5.
- **Naive centre-zero `sigma_UL` keeps falling to k=7** (optimum -> 7 for most
  masses <=3 TeV, mean gain x1.6-2 vs k=3). But eff is flat past 5, so this is
  purely `RMS(N_sp)` shrinking as the huge sideband over-constrains the
  background -- an artifact.
- **The catch: growing spurious BIAS.** Widening past the signal makes the fixed
  expo/powlaw mismodel the ~2 TeV-wide window and fake a signal: at m=3000,
  `|mean(N_sp)|/RMS` grows 0.33 -> 1.6 as k 3.5 -> 7. Centre-zero ignores this
  (uses only RMS), so it rewards the most-mismodeled windows; the RMS/q95 proxy
  misses it (checks width, not centring).
- **Fix: the `--bias-max` cut** (default 0.5 = standard spurious-signal
  acceptance). With it, wide windows are rejected -- contiguous safe window is
  typically **k ~ 3.5-4.5**, some masses fail even at k=2, and the clean gain
  evaporates. Set `--bias-max 99` to recover the (misleading) RMS-only view.
- **Bottom line: k=7 is not a real improvement.** The useful window is k~3-5,
  bracketed by the efficiency ceiling (5) and the bias ceiling (~4). The gain is
  limit **depth** not reach anyway (theory crossing is beyond the reliability
  ceiling regardless of k).

## Outputs (`run2/`)

    efficiency_vs_k_ee_resolved.csv     containment/eff per (mass,k)
    containment_vs_k/, eff_vs_k/, eff_map/, mn_spread/    Part A plots
    stage6/k{K}/                         per-k Stage-6 toy tables (Part B input)
    sensitivity_vs_k_ee_resolved.csv    k_opt, sigma_UL, gain per (mass,fn)
    sigma_vs_k/, kopt_vs_mass/, reach/, gain_vs_mass/     Part B plots

## Caveats

- Median expected limit, centre-zero. The spurious-**bias** systematic is not
  folded in; doing so (see `xsec_limit --center mean` / a quadrature term) would
  temper the low-mass gains where the bias is ~1 sigma.
- Reliability proxy uses the Stage-6 CSV columns (`n_ok`, `RMS`, `q95`); it
  reproduces the RMS/robust-sigma verdict from the k=3/k=5 study without
  re-reading the raw toys.
