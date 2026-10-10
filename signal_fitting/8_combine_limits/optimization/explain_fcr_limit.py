#!/usr/bin/env python3
"""From the two-region likelihood to the expected limit -- by hand, at one mass.

Same model, same inputs as explain_fcr.py (fcr_shape CardSB: SR = r S_i +
R T t_i(b_tt) + B e_i(b); e-mu CR = T t_i(b_tt) + o_i; SR pseudo-data = summed
MC, CR = real EGamma data).  Everything is in EVENTS; the conversion to a
cross section happens only when windows at different masses are compared.

  limit_0_forced      three fits to the same pseudo-data with r forced to 0,
                      r_95 and 2 r_95; the background refit each time; q_mu
                      printed on each panel.
  limit_1_profile     q~_mu vs mu on the (Asimov) pseudo-data: force r = mu,
                      refit T, b_tt, b, B, record -2 Delta lnL.
  limit_2_qmu_toys    at mu = r_95: q~_mu under the S+B and B-only universes
                      (Poisson toys of BOTH regions), q_obs, CLs+b, CLb, CLs.
  limit_3_cls_scan    CLs vs mu: asymptotic curve (sigma_A from the profile
                      curve) with toy points; the 0.05 crossing is r_95;
                      the +-1, +-2 sigma expected band from the same sigma_A.
  limit_numbers.json  every number quoted on the slides.

  PYTHONPATH=/opt/homebrew/lib/root .venv-fitting/bin/python explain_fcr_limit.py
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from explain_fcr import (ERA, MASS, VAR, INPUTS, OUT_DEFAULT, _cmslabel,  # noqa: E402
                         _save, setup_logging, load_lumi, rebin_snap, CardSB,
                         pseudo_data_sr)
import explain_fcr as X                                                 # noqa: E402
import uproot                                                           # noqa: E402
from scipy.stats import norm                                            # noqa: E402
from scipy.special import gammaln                                       # noqa: E402

logger = logging.getLogger("explain_fcr_limit")
ALPHA = 0.05
C_SB, C_B = "#e42536", "#5790fc"


# ---------------------------------------------------------------------------
# the test statistic
# ---------------------------------------------------------------------------

def qmu_tilde(sb, mu, data, data_cr):
    """CCGV q~_mu: -2 ln L(mu, theta^^) / L(r^, theta^), with r^ clipped at 0
    and q~ = 0 when r^ > mu (one-sided, for an upper limit)."""
    g = sb.fit(data, data_cr=data_cr)
    r_raw, nll_hat = g["nsig"], g["nll2"]
    r_hat = r_raw
    if r_hat < 0.0:
        r_hat, nll_hat = 0.0, sb.fit_profiled(0.0, data, data_cr=data_cr)["nll2"]
    if r_hat > mu:
        return 0.0, r_raw
    nll_mu = sb.fit_profiled(mu, data, data_cr=data_cr)["nll2"]
    return max(nll_mu - nll_hat, 0.0), r_raw


def toys_qmu(sb, mu_gen, mu_test, data, data_cr, ntoys, seed):
    """q~_{mu_test} on ntoys Poisson toys of BOTH regions, generated from the
    model at r = mu_gen with the background parameters profiled on the data."""
    prof = sb.fit_profiled(mu_gen, data, data_cr=data_cr)
    nu_sr = sb.nu(mu_gen, prof["params"])
    nu_cr = sb.nu_cr(prof["params"])
    rng = np.random.default_rng(seed)
    out, rh = np.empty(ntoys), np.empty(ntoys)
    for i in range(ntoys):
        d = rng.poisson(nu_sr).astype(float)
        c = rng.poisson(nu_cr).astype(float)
        out[i], rh[i] = qmu_tilde(sb, mu_test, d, c)
    return out, rh


# ---------------------------------------------------------------------------
# asymptotics (CCGV, arXiv:1007.1727, q~_mu)
# ---------------------------------------------------------------------------

def p_sb_asym(q, mu, sigma):
    a = (mu / sigma) ** 2
    if q <= a:
        return 1.0 - norm.cdf(math.sqrt(q))
    return 1.0 - norm.cdf((q + a) / (2.0 * mu / sigma))


def p_b_asym(q, mu, sigma):
    a = (mu / sigma) ** 2
    if q <= a:
        return 1.0 - norm.cdf(math.sqrt(q) - mu / sigma)
    return 1.0 - norm.cdf((q - a) / (2.0 * mu / sigma))


def cls_asym(q, mu, sigma):
    clsb = p_sb_asym(q, mu, sigma)
    clb = p_b_asym(q, mu, sigma)
    return clsb / clb if clb > 0 else 1.0


def expected_ul(sigma, nsig):
    """Median (nsig=0) and band edges of the expected 95% CL upper limit."""
    return sigma * (norm.ppf(1.0 - ALPHA * norm.cdf(nsig)) + nsig)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _info(ax, extra=""):
    ax.text(0.03, 0.97,
            f"{X.CH_LABEL}\nResolved SR + $e\\mu$ CR\n" f"{ERA}\n"
            fr"$m_{{W_R}}={MASS}$" + ("\n" + extra if extra else ""),
            transform=ax.transAxes, va="top", fontsize=15)


def plot_profile(mus, qs, sigma_A, r95, out, com, lumi):
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(mus, qs, color="black", lw=2.5, label=r"$q_\mu$ on the pseudo-data")
    ax.axhline(1.0, color="gray", lw=1.2, ls="--")
    ax.text(mus[-1], 1.0, r"$q_\mu = 1$: $\pm 1\sigma$ ", ha="right",
            va="bottom", fontsize=13, color="gray")
    ax.axvline(0.0, color=C_B, lw=1.5, ls=":")
    ax.text(0.5, 0.2, r"$\hat r = 0$", color=C_B, fontsize=14, ha="left")
    ax.axvline(sigma_A, color="gray", lw=1.0, ls=":")
    ax.text(sigma_A + 0.5, 0.2, fr"$\sigma = {sigma_A:.1f}$ events",
            fontsize=13, color="gray", ha="left")
    ax.set_xlabel(r"$\mu$: the forced value of $r$ [events]")
    ax.set_ylabel(r"$q_\mu = -2\,\ln\,[L(\mu)\,/\,L(\hat r)]$")
    ax.set_xlim(mus[0], mus[-1])
    ax.set_ylim(0, max(qs) * 1.08)
    _info(ax, "SR: simulation as pseudo-data\nCR: data")
    ax.legend(fontsize=14, loc="upper left", bbox_to_anchor=(0.03, 0.72))
    _cmslabel(ax, com, lumi)
    _save(fig, out)


def plot_forced(sb, mu_sr, cr_data, mus, qs, out, com, lumi):
    """Three panels: SR pseudo-data with the total fit for r forced to each
    value in `mus` (background parameters refit, CR included), the fitted
    signal part dashed, and q_mu on the panel."""
    hep.style.use("CMS")
    fig, axes = plt.subplots(1, len(mus), figsize=(8 * len(mus), 8),
                             sharey=True)
    xs = np.linspace(sb.lo, sb.hi, 400)
    cols = [C_B, "black", C_SB]
    # CardSB drops the ln n! constants; add them back so the printed value is
    # the full binned-Poisson -2 ln L of both regions (q_mu is unchanged)
    const = 2.0 * (float(np.sum(gammaln(mu_sr + 1.0)))
                   + float(np.sum(gammaln(np.asarray(cr_data) + 1.0))))

    def curve(norm, b):
        k = b / 1000.0
        cn = ((math.exp(k * (sb.hi - sb.m_c))
               - math.exp(k * (sb.lo - sb.m_c))) / k) if abs(b) > 1e-9 \
            else (sb.hi - sb.lo)
        return norm * np.exp(k * (xs - sb.m_c)) * sb.width / cn

    def sig(r):
        return r * sb.width / (math.sqrt(2 * math.pi) * sb.sigma) * np.exp(
            -0.5 * ((xs - sb.m_c) / sb.sigma) ** 2)

    for ax, mu, q, col in zip(axes, mus, qs, cols):
        prof = sb.fit_profiled(mu, mu_sr, data_cr=cr_data)
        T, b_tt, b, B = prof["params"][:4]
        rho = prof["params"][4] if sb.has_rho else 1.0
        bkg = curve(rho * sb.R * T, b_tt) + curve(B, b)
        ax.errorbar(sb.centers, mu_sr, yerr=np.sqrt(mu_sr), fmt="ko", ms=5,
                    label="pseudo-data (MC)")
        ax.plot(xs, bkg + sig(mu), color=col, lw=2.5,
                label=fr"fit with $r = {mu:.0f}$ held fixed")
        ax.plot(xs, bkg, color=col, lw=1.5, ls="--", label="its background part")
        ax.set_yscale("log")
        ax.set_ylim(0.3, 300.0)
        ax.set_xlim(sb.lo, sb.hi)
        ax.set_xlabel(r"$m_{lljj}$ [GeV]")
        ax.text(0.04, 0.96, fr"$r = {mu:.0f}$ events" "\n" fr"$q_\mu = {q:.1f}$",
                transform=ax.transAxes, va="top", fontsize=20, color=col)
        ax.text(0.04, 0.05, fr"$-2\ln L = {prof['nll2'] + const:.2f}$",
                transform=ax.transAxes, va="bottom", fontsize=17, color=col)
        ax.legend(fontsize=14, loc="upper right")
        _cmslabel(ax, com, lumi)
    axes[0].set_ylabel("Events / 50 GeV")
    fig.subplots_adjust(wspace=0.08)
    _save(fig, out)


def plot_qmu_toys(q_sb, q_b, q_obs, mu, res, out, com, lumi):
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 8))
    hi = max(np.percentile(q_b, 99.0), q_obs * 1.3, 6.0)
    bins = np.linspace(0, hi, 40)
    for arr, col, lab in ((q_sb, C_SB, fr"signal of {mu:.0f} events + background"),
                          (q_b, C_B, "background only")):
        ax.hist(np.clip(arr, bins[0], bins[-1]), bins=bins, histtype="step",
                lw=2.2, color=col, label=lab)
        ax.hist(np.clip(arr[arr >= q_obs], bins[0], bins[-1]), bins=bins,
                color=col, alpha=0.25)
    ax.axvline(q_obs, color="black", lw=2, ls="--",
               label=fr"pseudo-data: $q_\mu = {q_obs:.2f}$")
    ax.set_yscale("log")
    ax.set_ylim(0.5, None)
    ax.set_xlabel(fr"$q_\mu$ at $\mu = {mu:.0f}$ events")
    ax.set_ylabel("Toys")
    ax.set_xlim(bins[0], bins[-1])
    _info(ax, f"{len(q_sb)} toys per hypothesis")
    ax.text(0.97, 0.60,
            fr"CL$_{{s+b}}$ = {res['clsb']:.3f}" "\n"
            fr"CL$_b$ = {res['clb']:.3f}" "\n"
            fr"CL$_s$ = {res['cls']:.3f}",
            transform=ax.transAxes, fontsize=15, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1.0))
    ax.legend(fontsize=13, loc="upper right")
    _cmslabel(ax, com, lumi)
    _save(fig, out)


def plot_rhat_toys(r_sb, r_b, r_obs, mu, res, out, com, lumi):
    """The CLs test on the fitted-r axis: r-hat under the two hypotheses,
    the pseudo-data value, and the two lower tails."""
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 8))
    bins = np.linspace(-40, 70, 56)
    for arr, col, lab in ((r_b, C_B, "no signal injected"),
                          (r_sb, C_SB, fr"signal of {mu:.0f} events injected")):
        ax.hist(np.clip(arr, bins[0], bins[-1]), bins=bins, histtype="step",
                lw=2.2, color=col, label=lab)
        ax.hist(np.clip(arr[arr <= r_obs], bins[0], bins[-1]), bins=bins,
                color=col, alpha=0.25)
    ax.axvline(r_obs, color="black", lw=2, ls="--",
               label=fr"pseudo-data: $\hat r = {r_obs:.0f}$")
    ax.set_xlabel(r"fitted $\hat r$ [events]")
    ax.set_ylabel("Toys")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)   # headroom: labels and legend clear the histograms
    _info(ax, f"{len(r_sb)} toys per hypothesis")
    ax.text(0.97, 0.62,
            fr"CL$_{{s+b}}$ = {res['clsb']:.3f}" "\n"
            fr"CL$_b$ = {res['clb']:.3f}" "\n"
            fr"CL$_s$ = {res['cls']:.3f}",
            transform=ax.transAxes, fontsize=15, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1.0))
    ax.legend(fontsize=13, loc="upper right")
    _cmslabel(ax, com, lumi)
    _save(fig, out)


def plot_cls_scan(mus, cls_curve, toy_pts, r95, band, out, com, lumi, per_fb=None):
    """CLs vs the hypothesized r. per_fb (= L*eps, events per fb of sigma*B): draw the
    identical scan with the x axis converted to sigma*B = r / (L*eps) in fb."""
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 8))
    k = 1.0 / per_fb if per_fb else 1.0
    mus, r95 = np.asarray(mus) * k, r95 * k
    band = {key: x * k for key, x in band.items()}
    ax.plot(mus, cls_curve, color="black", lw=2.5,
            label=r"CL$_s$, combine's closed form")
    if toy_pts:
        xs, ys, es = zip(*toy_pts)
        ax.errorbar(np.asarray(xs) * k, ys, yerr=es, fmt="o", color=C_SB, ms=8, capsize=3,
                    label=r"CL$_s$, toys (fitted $r$ tails)", zorder=5)
    ax.axhline(ALPHA, color="gray", lw=1.5, ls="--")
    ax.text(mus[-1], ALPHA, " 0.05 ", ha="right", va="bottom", fontsize=13,
            color="gray")
    ax.axvline(r95, color=C_SB, lw=1.8, ls="--",
               label=(fr"$(\sigma\mathcal{{B}})_{{95}} = {r95:.2f}$ fb" if per_fb
                      else fr"$r_{{95}} = {r95:.0f}$ events"))
    m2, m1, p1, p2 = band["-2"], band["-1"], band["+1"], band["+2"]
    ax.axvspan(m2, p2, color="#f5d800", alpha=0.35, lw=0,
               label=r"expected $\pm 2\sigma$")
    ax.axvspan(m1, p1, color="#00cc00", alpha=0.35, lw=0,
               label=r"expected $\pm 1\sigma$")
    ax.set_xlabel(r"hypothesized $\sigma\mathcal{B} = r\,/\,L\varepsilon$ [fb]" if per_fb
                  else r"hypothesized $r$ [events]")
    ax.set_ylabel(r"CL$_s$")
    ax.set_xlim(mus[0], mus[-1])
    ax.set_ylim(0, 1.05)
    _info(ax)
    ax.legend(fontsize=13, loc="upper right")
    _cmslabel(ax, com, lumi)
    _save(fig, out)


def plot_cls_band(sigma, band, out, com, lumi, mu_max=65.0):
    """How the expected band is built. The CLs curve for a fit that lands at
    r-hat = N sigma (Gaussian tails of the fitted r, the toy-slide construction:
    CLs = Phi(N - mu/sigma) / Phi(N)), for N = -2..+2. N = 0 is the median (the
    Asimov); each curve crosses 0.05 exactly at one band edge."""
    from matplotlib.lines import Line2D
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 8))
    mus = np.linspace(0.0, mu_max, 500)
    ax.axvspan(band["-2"], band["+2"], color="#f5d800", alpha=0.35, lw=0)
    ax.axvspan(band["-1"], band["+1"], color="#00cc00", alpha=0.35, lw=0)
    c1, c2 = "#007a00", "#a88d00"
    for n, key, col, ls, lw in ((-2, "-2", c2, "--", 2.2), (-1, "-1", c1, "--", 2.2),
                                (0, "0", "black", "-", 2.8),
                                (1, "+1", c1, "--", 2.2), (2, "+2", c2, "--", 2.2)):
        ax.plot(mus, norm.cdf(n - mus / sigma) / norm.cdf(n), color=col, ls=ls, lw=lw)
        ax.plot([band[key]], [ALPHA], "o", color=col, ms=10, zorder=6,
                markeredgecolor="white", markeredgewidth=1.0)
    ax.axhline(ALPHA, color="gray", lw=1.5, ls="--")
    ax.text(mu_max, ALPHA, " 0.05 ", ha="right", va="bottom", fontsize=13, color="gray")
    handles = [Line2D([], [], color="black", lw=2.8,
                      label=fr"fit at the median, $\hat r = 0$: $r_{{95}} = {band['0']:.0f}$"),
               Line2D([], [], color=c1, ls="--", lw=2.2,
                      label=fr"fit at $\hat r = \pm 1\sigma$ ($\pm {sigma:.0f}$): green band"),
               Line2D([], [], color=c2, ls="--", lw=2.2,
                      label=fr"fit at $\hat r = \pm 2\sigma$ ($\pm {2 * sigma:.0f}$): yellow band")]
    ax.legend(handles=handles, fontsize=14, loc="upper right")
    ax.set_xlabel(r"hypothesized $r$ [events]")
    ax.set_ylabel(r"CL$_s$")
    ax.set_xlim(0.0, mu_max)
    ax.set_ylim(0, 1.05)
    _info(ax)
    _cmslabel(ax, com, lumi)
    _save(fig, out)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--ntoys", type=int, default=5000,
                   help="toys per hypothesis for limit_2 (default 5000)")
    p.add_argument("--scan-toys", type=int, default=1000,
                   help="toys per family per CLs scan point (default 1000)")
    p.add_argument("--scan-mus", type=float, nargs="+",
                   default=[10.0, 15.0, 20.0, 25.0, 30.0])
    p.add_argument("--mu-max", type=float, default=45.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--tt-to-cr", action="store_true",
                   help="SR pseudo-data = MC with tt+tW scaled to the "
                        "flavor-CR data (as explain_fcr --tt-to-cr)")
    p.add_argument("--replot-toys", action="store_true",
                   help="reuse the toys saved in <output-dir>/limit_toys.npz "
                        "(from a previous run) for limit_2 instead of "
                        "generating them again")
    p.add_argument("--forced-only", action="store_true",
                   help="only the limit_0_forced panels (no toys)")
    p.add_argument("--band-only", action="store_true",
                   help="only redraw limit_3_cls_band from <output-dir>/limit_numbers.json")
    p.add_argument("--channel", choices=["ee", "mumu"], default="ee",
                   help="lepton channel of the signal region (inputs <channel>_resolved*)")
    p.add_argument("--asimov", action="store_true",
                   help="replace the pseudo-data (SR and CR) by the expectation of the "
                        "background-only fit to it, so the fit returns r = 0 exactly "
                        "(combine's expected-limit construction)")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    info = load_lumi(ERA)
    lumi, com = info["lumi"], info.get("com", 13.6)

    X.set_channel(args.channel)
    if args.band_only:
        with open(args.output_dir / "limit_numbers.json") as fh:
            nums = json.load(fh)
        plot_cls_band(nums["sigma_A"], nums["band_events"], args.output_dir / "limit_3_cls_band",
                      com, lumi, mu_max=args.mu_max)
        logger.info("wrote %s", args.output_dir / "limit_3_cls_band")
        return
    tag = X.TAG
    with open(INPUTS / f"{tag}.json") as fh:
        meta = json.load(fh)["masses"][str(MASS)]
    with open(INPUTS / f"{tag}_fcr.json") as fh:
        fcr_all = json.load(fh)
    f = fcr_all["masses"][str(MASS)]["vars"][VAR]
    v = meta["vars"][VAR]
    edges, mu_sr, _ = pseudo_data_sr(INPUTS, v, f, args.tt_to_cr, tag=tag)
    hcr = uproot.open(INPUTS / f"{tag}_fcr.root")
    _, cr_data = rebin_snap(hcr["fcr_data_EGamma_native"].axes[0].edges(),
                            hcr["fcr_data_EGamma_native"].values(),
                            v["bw"], v["fit_lo"], v["fit_hi"])
    _, cr_other = rebin_snap(hcr["fcr_rest_native"].axes[0].edges(),
                             hcr["fcr_rest_native"].values(),
                             v["bw"], v["fit_lo"], v["fit_hi"])
    sb = CardSB(edges, meta["m_c"], meta["sigma"], mode="fcr_shape",
                fcr=dict(R=fcr_all["R_global"], T_cr=f["T_cr"],
                         b_tt=f["b_tt"], cr_other=cr_other,
                         R_err_rel=f.get("R_err_rel", 0.0)))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.asimov:
        # background-only fit to the pseudo-data, then take the fitted model's
        # expectation in every bin (SR and CR) as the data: r-hat = 0 exactly
        gb = sb.fit_profiled(0.0, mu_sr, data_cr=cr_data)
        mu_sr = np.asarray(sb.nu(0.0, tuple(gb["params"])), dtype=float)
        cr_data = np.asarray(sb.nu_cr(tuple(gb["params"])), dtype=float)
        logger.info("Asimov pseudo-data from the b-only fit: SR %.1f events, CR %.1f events",
                    mu_sr.sum(), cr_data.sum())
    # 1 -- the profile curve on the pseudo-data (= the b-only Asimov dataset)
    g = sb.fit(mu_sr, data_cr=cr_data)
    logger.info("global fit: r = %.2f +- %.2f", g["nsig"], g["nsig_err"])
    mus = np.arange(0.0, args.mu_max + 0.5, 1.0)
    qs = np.array([qmu_tilde(sb, m, mu_sr, cr_data)[0] for m in mus])
    # sigma_A at the limit itself (CCGV: sigma = mu / sqrt(q~_mu,A)); iterate
    sigma_A = g["nsig_err"]
    for _ in range(4):
        r95 = expected_ul(sigma_A, 0.0)
        qA, _ = qmu_tilde(sb, r95, mu_sr, cr_data)
        sigma_A = r95 / math.sqrt(qA)
    r95 = expected_ul(sigma_A, 0.0)
    band = {k: expected_ul(sigma_A, n) for k, n in
            (("-2", -2.0), ("-1", -1.0), ("0", 0.0), ("+1", 1.0), ("+2", 2.0))}
    logger.info("sigma_A = %.2f  r95(median expected) = %.2f  band %s",
                sigma_A, r95, {k: round(x, 1) for k, x in band.items()})
    plot_profile(mus, qs, sigma_A, r95, out / "limit_1_profile", com, lumi)
    forced = [0.0, float(round(r95)), 2.0 * float(round(r95))]
    forced_q = [qmu_tilde(sb, m, mu_sr, cr_data)[0] for m in forced]
    logger.info("forced r %s -> q_mu %s", forced, [round(q, 2) for q in forced_q])
    plot_forced(sb, mu_sr, cr_data, forced, forced_q, out / "limit_0_forced",
                com, lumi)
    if args.forced_only:
        return

    # 3 -- CLs vs mu: asymptotic curve from the profile curve, toy points
    mu_fine = np.linspace(0.5, args.mu_max, 200)
    q_fine = np.interp(mu_fine, mus, qs)
    cls_curve = np.array([cls_asym(q, m, sigma_A)
                          for q, m in zip(q_fine, mu_fine)])
    toy_pts = []
    scan = {}
    for m in args.scan_mus:
        q_obs = float(np.interp(m, mus, qs))
        q_sb, r_sb = toys_qmu(sb, m, m, mu_sr, cr_data, args.scan_toys, args.seed + int(m))
        q_b, r_b = toys_qmu(sb, 0.0, m, mu_sr, cr_data, args.scan_toys, args.seed + 1000 + int(m))

        def _cls(a_sb, a_b):
            clsb, clb = float(np.mean(a_sb)), float(np.mean(a_b))
            cls = clsb / clb if clb > 0 else 1.0
            err = cls * math.sqrt(max(clsb * (1 - clsb), 1e-9) / args.scan_toys / clsb**2
                                  + max(clb * (1 - clb), 1e-9) / args.scan_toys / clb**2) \
                if clsb > 0 else 0.0
            return clsb, clb, cls, err
        clsb, clb, cls, err = _cls(q_sb >= q_obs, q_b >= q_obs)
        # the seminar version of the same test: lower tails of the fitted r
        clsb_r, clb_r, cls_r, err_r = _cls(r_sb <= g["nsig"], r_b <= g["nsig"])
        toy_pts.append((m, cls_r, err_r))
        scan[str(m)] = dict(q_obs=q_obs, clsb=clsb, clb=clb, cls=cls, err=err,
                            clsb_r=clsb_r, clb_r=clb_r, cls_r=cls_r, err_r=err_r,
                            cls_asym=float(cls_asym(q_obs, m, sigma_A)))
        logger.info("  mu=%4.0f  CLs toys(q) %.3f +- %.3f  toys(r) %.3f +- %.3f"
                    "  asymptotic %.3f", m, cls, err, cls_r, err_r,
                    scan[str(m)]["cls_asym"])
    plot_cls_scan(mu_fine, cls_curve, toy_pts, r95, band,
                  out / "limit_3_cls_scan", com, lumi)
    plot_cls_scan(mu_fine, cls_curve, toy_pts, r95, band,
                  out / "limit_3_cls_scan_xsec", com, lumi, per_fb=v["rate_per_fb"])
    # the same two figures without the toy points (seminar slides 42 and 43)
    plot_cls_scan(mu_fine, cls_curve, None, r95, band,
                  out / "limit_3_cls_scan_notoys", com, lumi)
    plot_cls_scan(mu_fine, cls_curve, None, r95, band,
                  out / "limit_3_cls_scan_xsec_notoys", com, lumi, per_fb=v["rate_per_fb"])
    plot_cls_band(sigma_A, band, out / "limit_3_cls_band", com, lumi, mu_max=args.mu_max)

    # 2 -- the two q~ distributions at mu = r95 (rounded to an integer)
    mu_t = float(round(r95))
    q_obs = float(np.interp(mu_t, mus, qs))
    toy_file = out / "limit_toys.npz"
    if args.replot_toys and toy_file.exists():
        tz = np.load(toy_file)
        assert float(tz["mu"]) == mu_t, "saved toys are for a different mu"
        q_sb, r_sb, q_b, r_b = tz["q_sb"], tz["r_sb"], tz["q_b"], tz["r_b"]
        logger.info("reusing %d saved toys per hypothesis from %s", len(q_sb), toy_file)
    else:
        q_sb, r_sb = toys_qmu(sb, mu_t, mu_t, mu_sr, cr_data, args.ntoys, args.seed + 7)
        q_b, r_b = toys_qmu(sb, 0.0, mu_t, mu_sr, cr_data, args.ntoys, args.seed + 8)
        np.savez(toy_file, mu=mu_t, q_sb=q_sb, r_sb=r_sb, q_b=q_b, r_b=r_b)
    clsb = float(np.mean(q_sb >= q_obs))
    clb = float(np.mean(q_b >= q_obs))
    res = dict(mu=mu_t, q_obs=q_obs, clsb=clsb, clb=clb,
               cls=clsb / clb if clb > 0 else 1.0,
               cls_asym=float(cls_asym(q_obs, mu_t, sigma_A)))
    logger.info("mu=%.0f: q_obs=%.3f CLs+b=%.3f CLb=%.3f CLs=%.3f (asym %.3f)",
                mu_t, q_obs, clsb, clb, res["cls"], res["cls_asym"])
    plot_qmu_toys(q_sb, q_b, q_obs, mu_t, res, out / "limit_2_qmu_toys",
                  com, lumi)
    # the same test on the fitted-r axis: "at or below the pseudo-data"
    r_obs = g["nsig"]
    clsb_r = float(np.mean(r_sb <= r_obs))
    clb_r = float(np.mean(r_b <= r_obs))
    res_r = dict(mu=mu_t, r_obs=r_obs, clsb=clsb_r, clb=clb_r,
                 cls=clsb_r / clb_r if clb_r > 0 else 1.0,
                 mean_sb=float(np.mean(r_sb)), rms_sb=float(np.std(r_sb)),
                 mean_b=float(np.mean(r_b)), rms_b=float(np.std(r_b)))
    logger.info("mu=%.0f on r-hat: CLs+b=%.3f CLb=%.3f CLs=%.3f",
                mu_t, clsb_r, clb_r, res_r["cls"])
    plot_rhat_toys(r_sb, r_b, r_obs, mu_t, res_r, out / "limit_2_rhat_toys",
                   com, lumi)

    numbers = dict(mass=MASS, var=VAR, window=[sb.lo, sb.hi],
                   r_hat=g["nsig"], r_hat_err=g["nsig_err"],
                   sigma_A=sigma_A, r95_events=r95,
                   band_events=band, rate_per_fb=v["rate_per_fb"],
                   eff=v["eff"], lumi=lumi,
                   r95_fb=r95 / v["rate_per_fb"],
                   band_fb={k: x / v["rate_per_fb"] for k, x in band.items()},
                   qmu_toys=res, rhat_toys=res_r, cls_scan=scan,
                   profile=dict(mu=mus.tolist(), q=qs.tolist()))
    with open(out / "limit_numbers.json", "w") as fh:
        json.dump(numbers, fh, indent=1)
    logger.info("r95 = %.1f events = %.3f fb (rate %.2f ev/fb); wrote %s",
                r95, numbers["r95_fb"], v["rate_per_fb"], out)


if __name__ == "__main__":
    main()
