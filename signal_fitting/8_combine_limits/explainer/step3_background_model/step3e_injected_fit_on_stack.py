#!/usr/bin/env python3
"""Step 3e -- the step-3d picture with a signal injected and the window fit run.

Pseudo-data = Poisson draw of (summed 2018 background MC + N_inj events of the
m_WR = 2000 signal shape) in the 31 window bins. Two fits to those counts:
  S+B:    nu_i = r S_i + N e_i(b), with r, N, b free (mu, sigma fixed to the
          step-2 values), and
  B-only: the same with r = 0.
Drawn over the background stack: the pseudo-data points, the S+B fit (solid) and
the B-only fit (dashed), both inside the window only.

    python step3e_injected_fit_on_stack.py [--ninj 300] [--seed S]

Made 2026-10-03 for the HEP seminar (research-slides/talks/20261007_hep_seminar).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.patches import Patch
from scipy.optimize import minimize
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common as C

HERE = Path(__file__).resolve().parent
hep.style.use("CMS")
M_C, SIGMA, LO, HI = C.window_2000()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ninj", type=float, default=300.0, help="injected signal events in the window")
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--xmax", type=float, default=6000.0, help="right edge of the x axis")
    ap.add_argument("--binned", action="store_true",
                    help="window-only plot of the counts n_i with the fitted nu_i in 31 bins, plus the\n"
                         "profile uncertainty on r (homemade version of explainer step 6b)")
    ap.add_argument("--sigline", action="store_true",
                    help="seminar variant: no pseudo-data points and no B-only line; the injected signal\n"
                         "is drawn as a dashed line stacked on the background and the S+B fit runs on\n"
                         "the unfluctuated (Asimov) sum")
    args = ap.parse_args()

    comps = C.load_bkg_components(rebin=5)
    edges = comps["DYJets"][0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    colors = {"Other": "#94a4a2", "Nonprompt": "#964a8b", "tt_tW": "#f89c20", "DYJets": "#5790fc"}
    order = ["Other", "Nonprompt", "tt_tW", "DYJets"]
    total = np.maximum(sum(comps[s][1] for s in C.BKG_SAMPLES), 0.0)
    sel = (centers >= LO) & (centers <= HI)
    c, lo_e, hi_e = centers[sel], edges[:-1][sel], edges[1:][sel]

    # signal template: Gaussian integrated per bin, unit sum in the window
    cdf = lambda x: 0.5 * (1 + erf((x - M_C) / (SIGMA * np.sqrt(2))))
    S = cdf(hi_e) - cdf(lo_e); S /= S.sum()
    rng = np.random.default_rng(args.seed)
    data = (total[sel] + args.ninj * S) if (args.sigline or args.binned) else rng.poisson(total[sel] + args.ninj * S).astype(float)

    def expo(b):
        e = np.exp(b * (c - M_C) / 1000.0); return e / e.sum()

    def nll(par, with_sig):
        r, N, b = (par if with_sig else (0.0, *par))
        nu = np.clip(r * S + N * expo(b), 1e-9, None)
        return float((nu - data * np.log(nu)).sum())

    sb = minimize(nll, x0=[args.ninj, total[sel].sum(), -2.9], args=(True,), method="Nelder-Mead",
                  options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 20000})
    bo = minimize(nll, x0=[data.sum(), -2.9], args=(False,), method="Nelder-Mead",
                  options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 20000})
    r_hat, N_hat, b_hat = sb.x
    N0, b0 = bo.x
    q0 = 2 * (bo.fun - sb.fun)
    print(f"injected {args.ninj:.0f}: S+B r={r_hat:.1f}, N={N_hat:.1f}, b={b_hat:.3f}; "
          f"B-only N={N0:.1f}, b={b0:.3f}; q0={q0:.1f} (Z~{np.sqrt(max(q0, 0)):.1f})")

    # profile scan in r for the uncertainty: -2 ln L rises by 1 (used by the
    # --binned plot and by the S+B legend of the --sigline plots)
    def prof(r):
        return 2 * minimize(lambda q: nll([r, *q], True), x0=[N_hat, b_hat], method="Nelder-Mead",
                            options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 20000}).fun
    base = prof(r_hat)
    rs = np.linspace(r_hat - 60, r_hat + 60, 241)
    dq = np.array([prof(r) for r in rs]) - base
    inside = rs[dq <= 1.0]
    err_lo, err_hi = r_hat - inside.min(), inside.max() - r_hat
    print(f"profile: r = {r_hat:.1f} -{err_lo:.1f} +{err_hi:.1f} events")
    if args.binned:
        nu = r_hat * S + N_hat * expo(b_hat)
        fig, ax = plt.subplots(figsize=(9, 7.5))
        ax.errorbar(c, data, yerr=np.sqrt(data), fmt="o", color="black", markersize=4, zorder=7,
                    label=f"counts $n_i$ (summed {C.MC_LABEL.split()[-2]} MC)")
        ax.stairs(nu, np.append(lo_e, hi_e[-1]), color="#1565c0", linewidth=2.2, zorder=8,
                  label=r"fitted $\nu_i = \hat r S_i + \hat N e_i(\hat b)$")
        ax.set_yscale("log"); ax.set_xlim(LO, HI); ax.set_ylim(3e-1, 3e2)
        C.log_yaxis_one_ten(ax)
        ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)"); ax.set_ylabel("Events / 50 GeV")
        ax.legend(loc="upper right", fontsize=15, frameon=False)
        ax.text(0.03, 0.05, "ee\nResolved SR\n" + rf"$m_{{W_R}}=2000$ GeV" + "\n\n"
                + rf"$\hat r = {0.0 + round(r_hat):.0f}\,^{{+{err_hi:.0f}}}_{{-{err_lo:.0f}}}$ events" + "\n"
                + rf"$\hat N = {N_hat:.0f}$,  $\hat b = {b_hat:.2f}$/TeV",
                transform=ax.transAxes, fontsize=15, va="bottom")
        C.cms_label(ax)
        C.savefig(fig, HERE / "step3e_homemade_likelihood")
        return
    mfine = np.linspace(LO, HI, 500)
    bw = edges[1] - edges[0]
    def curve(r, N, b):
        e = np.exp(b * (mfine - M_C) / 1000.0) / np.exp(b * (c - M_C) / 1000.0).sum()
        g = (cdf(mfine + bw / 2) - cdf(mfine - bw / 2)) / (cdf(hi_e) - cdf(lo_e)).sum()
        return r * g + N * e

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.stackplot(centers, *[comps[s][1] for s in order], step="mid",
                 colors=[colors[s] for s in order], labels=[C.BKG_LABELS[s] for s in order])
    handles, labels = ax.get_legend_handles_labels()
    ax.axvspan(LO, HI, color="tab:red", alpha=0.12, zorder=4)
    win = Patch(facecolor="tab:red", alpha=0.12, label=rf"window $[{LO:.0f}, {HI:.0f}]$ GeV")
    extra = []
    if args.sigline and args.ninj > 0:
        sig_full = np.zeros_like(total)
        sig_full[sel] = args.ninj * S
        (l_sig,) = ax.step(centers[sel], (total + sig_full)[sel], where="mid", color="#b2182b", linewidth=2.2,
                           linestyle="--", zorder=7, label=f"+ {args.ninj:.0f} signal events (simulation)")
        extra.append(l_sig)
    elif not args.sigline:
        pts = ax.errorbar(c, data, yerr=np.sqrt(data), fmt="o", color="black", markersize=4, zorder=7,
                          label=f"pseudo-data: background + {args.ninj:.0f} signal events")
        extra.append(pts)
    (l_sb,) = ax.plot(mfine, curve(r_hat, N_hat, b_hat), color="#1b5e20", linewidth=2.6, zorder=8,
                      label=rf"S+B fit: $\hat r = {0.0 + round(r_hat):.0f} \pm {int(np.floor(0.5 * (err_lo + err_hi) + 0.5))}$ events")
    extra.append(l_sb)
    if not args.sigline:
        (l_b,) = ax.plot(mfine, curve(0.0, N0, b0), color="#e65100", linewidth=2.4, linestyle="--", zorder=8,
                         label="background-only fit ($r = 0$)")
        extra.append(l_b)
    ax.set_yscale("log"); ax.set_xlim(800, args.xmax); ax.set_ylim(1e-3, 1e4)
    C.log_yaxis_one_ten(ax)
    ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)"); ax.set_ylabel("Events / 50 GeV")
    ax.legend(handles[::-1] + [win] + extra, labels[::-1] + [win.get_label()] + [h.get_label() for h in extra],
              loc="upper right", fontsize=13, frameon=False)
    ax.text(0.03, 0.96, f"ee\nResolved SR\n{C.MC_LABEL}", transform=ax.transAxes, fontsize=16, va="top")
    C.cms_label(ax)
    C.savefig(fig, HERE / (f"step3e_injected_fit_on_stack_n{args.ninj:.0f}" + ("_sigline" if args.sigline else "")))


if __name__ == "__main__":
    main()
