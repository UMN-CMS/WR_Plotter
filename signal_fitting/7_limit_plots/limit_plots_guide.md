# From `nsp_hist` to limit plots — a plain-language guide

This explains how the Stage-6 toy distributions become the Stage-7 expected-limit
plots, and how those become cross-section limits. It assumes you know everything
up to Stage 6 (windows, S+B fits, toys) and nothing about limit-setting.

Throughout, one worked example is followed: **ee resolved, expo, m_WR = 1200**,
whose Stage-6 numbers are `mean_Nsp = 21.4`, `rms_Nsp = 47.9`.

---

## 1. What Stage 6 handed us

For each mass, the toys answer one question: *when there is no signal at all,
what does our fit report for the signal yield?* The answer is `nsp_hist` — to a
very good approximation a Gaussian with

* **mean `mu0`** — the spurious-signal bias (the fit invents this much signal
  from background mismodeling alone), and
* **RMS `sigma`** — the statistical noise of the yield measurement.

Read `sigma` as the *resolution* of our instrument: even with zero true signal,
any single dataset will hand us a fitted N_sig somewhere in `mu0 ± 2 sigma`.
For the example mass: `21.4 ± 2×47.9`, i.e. anywhere from about −74 to +117
events, just from Poisson noise.

Everything in Stage 7 is computed from those two numbers per (mass, function).
**No new fits happen after Stage 6** — Stage 7 is a formula applied to the
Stage-6 CSV.

---

## 2. What an upper limit is

Since we see no signal, the question flips from "how much signal is there?" to
**"how much signal could still be hiding?"** The logic:

> Suppose the true signal yield were N. Then our fitted yield would come out
> near N, give or take the noise `sigma`. If N is so large that a fitted value
> as small as the one we actually got would happen less than 5% of the time,
> that N is not believable — it is **excluded at 95% confidence**.

The **95% CL upper limit** `UL` is the largest N that survives this test. Every
yield above `UL` is excluded; everything below is still possible. For a Gaussian
measurement `N_hat` with noise `sigma`, the naive version of this is

    UL = N_hat + 1.645 * sigma        (one-sided 95%)

— "the truth can't sit more than ~1.6 noise-widths above what we measured."

## 3. The CL_s tweak (why 1.96 and not 1.645)

The naive formula misbehaves when the data fluctuates *low*. If `N_hat` comes
out at −2 sigma, the naive limit is `N_hat + 1.645 sigma < 0`: we would claim to
exclude *zero* signal, which is nonsense — that lucky downward fluctuation says
nothing about signals we were never sensitive to.

**CL_s** is the standard fix: the exclusion p-value is divided by the
background-compatibility p-value, which automatically weakens limits exactly
when the data fluctuates below the background expectation. In the Gaussian
(asymptotic) form used in `expected_limit.py`:

    UL(N_hat) = N_hat + sigma * Phi^-1( 1 - 0.05 * Phi(N_hat / sigma) )

Two anchor values are worth memorizing:

* if the data lands exactly on expectation (`N_hat = mu0`, and `mu0` small):
  `UL ≈ mu0 + 1.96 sigma`;
* if the data fluctuates 2 sigma *low*, the limit still comes out at
  `mu0 + 1.05 sigma` — the CL_s "floor". The limit never dives toward zero.

So: **a 95% CL_s limit is roughly "two noise-widths of signal", never less than
about one.**

---

## 4. Stage 6 → Stage 7: the expected limit on N_sig

We have not fitted data yet, so there is no `N_hat` and no limit — but we can
ask: **"when we do, what limit should we expect?"** Every Stage-6 toy is a
stand-in for a possible dataset. Feed each toy's fitted yield through `UL(.)`
and you get a *distribution of limits* — one thousand possible outcomes of the
analysis at each mass. The plot summarizes that distribution:

* **dashed line ("median expected")** — the middle outcome: half of all
  possible datasets give a stronger limit, half weaker. This is the headline
  sensitivity number.
* **green band ("68% expected")** — the middle 68% of outcomes (16th to 84th
  percentile). When real data arrives and is boring (no signal), the observed
  limit should land inside the green band 68% of the time.
* **yellow band ("95% expected")** — the middle 95% (2.5th to 97.5th). Landing
  outside the yellow band is a once-in-forty event — either a real fluctuation
  worth staring at, or a sign something is wrong.
* **red points (event-level plot only)** — the spurious yield from the
  unfluctuated MC fit. If a red point climbs toward the dashed line, the fit
  bias is competing with the statistical sensitivity at that mass.

A practical shortcut makes this cheap: `UL(.)` is monotonic (a larger fitted
yield always gives a weaker limit), so the percentiles of the *limit*
distribution are just `UL(.)` evaluated at the percentiles of the *yield*
distribution — `mu0`, `mu0 ± sigma`, `mu0 ± 2 sigma`. That is the whole of
`expected_limit.py`: five evaluations of a closed-form formula per mass.

**Worked example** (m = 1200): `mu0 = 21.4`, `sigma = 47.9` →

| −2σ | −1σ | median | +1σ | +2σ |
|---|---|---|---|---|
| 57.2 | 78.1 | **109.1** | 149.9 | 196.2 |

"We expect to exclude any signal above ~109 events; on a lucky dataset ~57, on
an unlucky one ~196." Note the asymmetry of the band around the median
(−52/+87): that is the CL_s floor at work on the low side.

Two reading notes for the vs-mass plot:

* the curve *falls with mass* because `sigma` tracks the background under the
  window, and the background falls steeply;
* the curve is *jagged* because each mass is its own independent set of 1000
  toys — that is honest statistical scatter, not a bug.

`--center mean` (default) bakes the spurious bias `mu0` into the band;
`--center zero` shows the pure-statistical band and leaves the bias to be
quoted as a separate systematic.

---

## 5. Stage 7 → 7b: from events to cross section

The event-level limit answers "how many signal events could be hiding in the
window?" To compare with theory we need "how big could the signal *cross
section* be?" The bridge is the standard counting relation — a signal of cross
section sigma×B leaves this many events in our window:

    N = (sigma*B) × L × eff
        eff = S_fit / genEventSumw

* `S_fit` — raw signal-MC yield inside the S+B fit range `[fit_lo, fit_hi]`
  (straight off the Stage-6 CSV, so it is the same window by construction);
* `genEventSumw` — the genWeight sum over *all generated* events, from the
  analyzer config. Because the signal histograms are raw genWeight fills, this
  single ratio contains the whole funnel: trigger + selection efficiency,
  acceptance, and the fraction of the peak inside the window;
* `L` — the luminosity of the background the band was built from.

Solving for the cross section turns the event limit into a cross-section limit
by **dividing by one number per mass**:

    sigma_UL = N_UL / (1000 × L[fb^-1] × eff)          [pb]

Because that is a fixed positive scale factor at each mass, *all five band
percentiles divide through unchanged* — the median stays the median, the 68%
band stays the 68% band. Nothing statistical happens in Stage 7b; it is a unit
conversion.

**Worked example continued** (m = 1200): `eff = 0.136`, `L = 109.8 fb^-1` →
one pb of signal would leave `1000 × 109.8 × 0.136 ≈ 15,000` events in the
window. The 109.1-event median limit becomes

    sigma_UL = 109.1 / 14958 = 7.3e-3 pb = 7.3 fb

---

## 6. Reading the cross-section plot

Same skeleton as before — dashed median, green 68%, yellow 95%, all now in
femtobarns on a log axis — plus two new elements:

* **red "Theory (g_R = g_L)" curve** — the *predicted* sigma×B versus mass for
  the benchmark model (from the config, m_N = m_WR/2). It falls steeply because
  the parton luminosity for making a multi-TeV object collapses with mass.
* **the crossing point** — the money feature. Where the theory curve is *above*
  the limit band, the predicted signal would have been excluded: **every mass
  to the left of the crossing is expected to be ruled out.** The crossing of
  theory with the dashed median (~4.5 TeV here) is the *expected mass reach*;
  its crossings with the band edges say how that reach could fluctuate.
* **"Observed limit (MC Asimov)" (solid black)** — a placeholder for data: the
  limit evaluated on the *unfluctuated background MC treated as if it were
  data* (`nsp_asimov`, with the same toy-RMS sigma as the band). With no
  fluctuations it hugs the median by construction; when a real dataset is
  fitted, its N_sp replaces `nsp_asimov` and this line becomes the actual
  observed limit, wiggling in and out of the green band.

At the example mass the theory prediction is 2.39 pb while the median limit is
7.3 fb — the limit sits a factor ~330 *below* the prediction, i.e. m_WR = 1200
is overwhelmingly excluded (as expected: this region was ruled out long ago).

Why the shapes differ between the two plots: the *event* limit falls with mass
(background, hence noise, dies away), but the *cross-section* limit flattens
into a plateau at high mass. Once the window is essentially background-free the
limit saturates at the few-event Poisson floor, and dividing a constant few
events by a flat `L × eff` gives a flat sigma_UL — the same ~0.1 fb plateau you
see in the official Run-2 plots.

**The mu = sigma/sigma_theory companion plot** is the same information with the
theory curve divided out: the red line becomes the horizontal `mu = 1`, and the
expected reach is where the band crosses it. Reading `mu_UL` directly tells you
the margin: at m = 1200, `mu_UL = 0.003` means we could exclude a signal 300×
smaller than predicted; `mu_UL = 2` at 4.8 TeV means the prediction there is
twice too small to exclude.

---

## 7. Caveats that carry over

* ee-resolved background closure is trusted only up to ~3.4 TeV (Stage 4), so
  the ~4.5 TeV crossing is indicative, not a claim.
* The band folds in *only* the background statistics and the spurious-signal
  bias — no lumi, JES, lepton-SF or signal-shape systematics yet.
* Signal inputs are the RunII (13 TeV) samples standing in for Run3: `eff` and
  the theory curve inherit that approximation.
* `eff` counts signal *inside the fit range*, while the fitted N_sp is a full
  Gaussian normalization; a Stage-8-style injection of the true MC shape would
  measure the small residual recovery factor between the two conventions.
