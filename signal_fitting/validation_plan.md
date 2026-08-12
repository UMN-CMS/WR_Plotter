# Data-driven background estimation — standard procedure & checks (W_R → ℓℓjj)

**Scope:** validation of the **background estimate only** — the standard checks a
data-driven background method must pass before unblinding. (Signal modeling,
trigger/object efficiency, the full statistical model, look-elsewhere, and
expected limits are *out of scope* here.) Everything runs on **MC and unblinded
control regions / sidebands** — no signal-region (SR) data.

---

## What is being estimated

- **Backgrounds:** DY+jets (dominant), tt̄/tW, nonprompt, other.
- **Observables:** resolved SR fits **m_ℓℓjj** (four-object dilepton+dijet mass;
  codebase `mass_fourobject`); boosted SR fits **m_ℓJ** (lead lepton + AK8 jet,
  two-object mass; `mass_twoobject`, SR for m_ℓJ > 0.8 TeV).
- **Data-driven pieces:**
  - **tt̄/tW** — eμ **flavor-symmetry** from a flavor control region (boosted uses
    lepton-in-jet topologies: lead μ + e-in-jet for the ee search, lead e +
    μ-in-jet for the μμ search; separate CRs per channel). Normalization enters
    via a **simultaneous SR + flavor-CR fit** (transfer factor in the likelihood).
  - **DY** — MC shape with data constraints: DY-CR (60 < m_ℓℓ < 150 GeV),
    pT(Z) NLO/LO K-factor + nNLO EW corrections, a data-driven jet-pT **reshape**,
    and a **floating normalization** in the fit. *Our pipeline currently uses LO
    DY with k-factors = 1.0 and no reshape — to be added.*
  - **Functional in-window fit** (`4_background_fits`, `5_signal_injection_study`)
    — a smooth analytic function fit in the signal window; the data-driven version
    fits **data sidebands** and interpolates under the peak.

> **Method vs precedent.** The flagship CMS W_R → ℓℓjj search (arXiv:2112.03949,
> JHEP 04 (2022) 047) estimates the background with a CR-constrained binned-ML
> **template shape fit** (simultaneous SR + flavor-CR + DY-CR), *not* an analytic
> SR function. The functional in-window/sideband approach here is a deviation —
> justify it (limited high-mass MC stats, robustness to template mismodeling) or
> frame it as a cross-check of the template fit.

---

## The standard data-driven checks

Status: ✅ done · 🔶 in progress · ⬜ to do

| # | check | what it must show | acceptance | status |
|---|---|---|---|---|
| 1 | **Background-shape adequacy** (`in_window_fit`) | the function/template describes the background shape | toy-based GoF (saturated / Baker–Cousins) p-value above threshold — not χ²/ndf alone; fit valid + accurate covariance | ✅ / 🔶 |
| 2 | **Closure (MC)** (`sideband_closure`) | the method predicts the in-window / SR background **without looking under the peak** | predict-from-sidebands vs MC truth; non-closure compatible with zero | ⬜ |
| 2b | **Closure validity range** | where the method is trustworthy | above ~3.6 TeV (resolved) the peak piles toward the spectrum endpoint and the upper sideband vanishes → the sideband/window prediction breaks (our closure degrades to −50…−90 % by ~4.6 TeV). State the valid range; the endpoint needs a shape fit / counting | ⬜ |
| 3 | **Validation in unblinded data** (`flavor_cr_fit`, DY-CR) | the method works on **real data** where you can look | prediction vs observation agree in signal-free regions: eμ flavor CR (tt̄/tW), DY CR, off-peak sidebands | ⬜ |
| 4 | **Spurious signal** (`5_signal_injection_study` B1) | the background model does **not fake** a signal | fit S+B with N_inj = 0; \|N_sp\| small — analysis-dependent (ATLAS form: 10 % of expected signal **or** 20 % of stat uncertainty, ~0.2 σ; 20–50 % range). Report vs mass | 🔶 |
| 5 | **Signal injection / non-absorption** (B2) | a real signal is recovered and does **not bias** the background | inject known N (real signal MC), recover slope ≈ 1, intercept ≈ spurious. **Toy version** (bin-wise `Poisson(bkg + N·shape)`) for pull/coverage — separate from Asimov | 🔶 |
| 6 | **Signal contamination** | signal leaking into the sidebands/CRs does not bias the estimate | quantify leakage vs injected μ at a plausible max signal, or argue negligible. *MC closure is clean (no W_R in the background template); the exposure is the **data** sidebands/CRs. The in-window S+B fit floating N_sig absorbs contamination, the sideband-only prediction does not* | ⬜ |
| 7 | **Bias / non-closure systematic** (B3) | the residual bias is quantified and propagated | from non-closure / spurious / function choice, **as a function of mass**, per function → the background systematic | 🔶 |
| 8 | **Stability under method choices** (C1, C3) | the estimate doesn't depend on arbitrary choices | vary window half-width `k` (1.5/2/2.5/3 σ), fit range, sideband width, and **bin width** (50/100/200 GeV — at m2000, σ~140 → only ~3 bins/FWHM at 100 GeV); B_fit, GoF, and spurious stable | ⬜ |
| 9 | **Function-choice uncertainty** | the "which function?" choice doesn't bias the result | discrete-profiling / envelope (Dauncey et al.) or one function + the bias systematic. F-test only for **nested** orders (expo → expo2 → expo3); non-nested choices → envelope / AIC | ⬜ |

---

## Where we are

- ✅ **(1)** adequacy (Stage 4). 🔶 **(4)(5)(7)** spurious + injection + bias-vs-mass
  (Stage 5, **resolved only** — boosted and the toy/pull study still to run).
- **Parked but central:** **(2)** sideband closure and **(3)** flavor-CR / DY-CR
  validation — in `signal_fitting/archived/`. These are the heart of a data-driven
  estimate (closure + unblinded-data validation).
- **⚠ B is method-validation only, not the deliverable systematic yet:** the
  current inputs are a Run-2 signal (RunIISummer20UL18) injected on a Run-3
  background (RunIII2024Summer24) — a placeholder — and a LO-DY template with
  k-factors = 1.0 that misses CR data/MC by ~20–30 %. Checks (4)(5)(7) are **not
  final** until the eras match and the DY K-factor + reshape are applied.

## The data-driven validation logic, in one line

**Assumption → closure (MC) → validation region (data) → spurious / contamination
→ bias systematic → stability.** That sequence is the standard package; checks
(4)–(6) are the functional-fit-specific way of asking "does the method fake or
absorb signal," which the closure/VR tests (2)(3) complement on the
background-only side.

## References

- Flagship CMS W_R → ℓℓjj: arXiv:2112.03949 (JHEP 04 (2022) 047).
- Discrete profiling: Dauncey, Kenzie, Wardle, Davies, *JINST* **10** (2015) P04015 [arXiv:1408.6865].
- Earlier CMS W_R: arXiv:1803.11116, arXiv:1407.3683.
