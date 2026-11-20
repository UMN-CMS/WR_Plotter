#!/usr/bin/env python3
"""The bias test made visible, one cell at a time: one truth, one channel, one signal mass, the
candidate functions. Figures in results/<channel>/11_bias_explainer/<variant>_<truth>_m<mass>/:

  mcfit_<cand>                the expected spectrum with the candidate's background-only fit over the
                              fit range (the Asimov S+B N_sig in the legend), pull (expected - fit) / sqrt(fit)
  nsig_pull_<cand>            N_sig / sigma(N_sig) over the toys at zero injection with the Gaussian fitted to it
                              (its mean = the bias, its sigma = the pull width: lib/pulls.py)
  toyfit_<cand>_inj<k>        one Poisson toy of the truth (k x the expected limit injected) over
                              the full fit range with its S+B fit: the fitted background alone
                              (dashed) and background + signal (solid), the expected spectrum as a
                              step histogram, the fit start, the signal's mu and sigma and N_sig in
                              the legend title, the pull (data - fit) / sqrt(fit) per bin below
  nsig_dist_<cand>            N_sig over the 500 toys of the study at zero injection, in events:
                              the median, the mean, the +-RMS band (S_spur = median / RMS is in numbers.md)
  nsig_dist_compare           the candidates' N_sig distributions and medians on one axis
  nsig_err_<cand>             the per-toy fit error sigma(N_sig) against the RMS of N_sig and the
                              Asimov sigma_fit: the RMS is the typical fit error
  nsig_inject_<cand>          N_sig at 0, 1x and 2x the expected limit with the injected values:
                              the bias at injection = (mean - injected) / RMS
  numbers.md                  every number drawn

The toys of the distributions are the study's own (results/<channel>/<toys-dir>/<variant>/
nsig_<truth>.npz) unless --ntoys asks for more: then they are thrown and fitted here, in parallel
(--workers), per candidate and injection. The single toys are drawn here with the study's seed, one
toy fitted by every candidate so the functions are compared on the same pseudo-data.

  cd WR_Plotter && python background_fit/11_bias_explainer.py
      [--channel ee] [--variant lo_ht] [--truth raw] [--mass 1600] [--candidates expo powexp]
      [--lo 1200] [--toys-dir 4_toys_lo1200] [--ntoys 100000 --workers 8]
      [--k 5 --lo 1000 --toys-dir 6_windows/k5]      # the +-5 sigma window of the window study (folder tag _k5)
      [--truth kde --truth-dir 3_truths_kde --ntoys 100000]   # the RooKeysPdf-alone truth (no study toys: thrown here)
"""
from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))
import config as C                                                       # noqa: E402

logger = logging.getLogger("11_bias_explainer")
EXPL_COLOR = {"expo": "#5790fc", "powexp": "#f89c20", "atlasemu": "#e42536", "wprime7": "#964a8b", "dilep8": "#7a21dd"}
FIT_COLOR = "#000000"      # the Gaussian fitted to the pulls
SB_COLOR = "#e42536"          # the background + signal fit, every candidate


def toy_job(job):
    """One chunk of toys of one (candidate, injection): throw, fit, return N_sig, its error, the checks."""
    import ROOT
    from sbfit import SBModel
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    model = SBModel(job["cand"], job["mu"], job["sigma"], job["lo"], job["hi"])
    rng = ROOT.TRandom3(job["seed"])
    expect, e = job["expect"], job["edges"]
    ns, es, oks = [], [], []
    for _ in range(job["n"]):
        toy = np.array([rng.Poisson(max(x, 0.0)) for x in expect], float)
        r = model.fit(e, toy, job["bkg_seed"], job["n_inj"], job["nsig_bounds"])
        if r is None:
            continue
        ns.append(r["nsig"])
        es.append(r["nsig_err"])
        oks.append(r["passed"] and r["nsig_err"] > 0)
    return job["key"], np.array(ns), np.array(es), np.array(oks, bool)


def load_rows(path, cand, mass):
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["candidate"] == cand and abs(float(r["mass"]) - mass) < 1 and r.get("status") == "ok":
                out[float(r["inject"])] = {k: (float(v) if v not in ("", "None") else float("nan"))
                                           for k, v in r.items() if k in ("n_inj", "asimov_err", "asimov_nsig", "nsig_median",
                                                                           "nsig_mean", "nsig_rms", "bias_median", "bias_mean",
                                                                           "pull_width", "n_conv", "n_passed", "ul_expected")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", default="ee")
    ap.add_argument("--variant", default="lo_ht")
    ap.add_argument("--truth", default="raw")
    ap.add_argument("--mass", type=float, default=1600.0)
    ap.add_argument("--candidates", nargs="+", default=["expo", "powexp"])
    ap.add_argument("--lo", type=float, default=1200.0, help="fit start (full range) or the floor of the window")
    ap.add_argument("--k", type=float, default=0.0,
                    help="window half-width in sigma around the signal mean, as 6_windows.py: the fit range becomes "
                         "[max(mu - k sigma, --lo), min(mu + k sigma, 8000)]; 0 = the full range from --lo")
    ap.add_argument("--toys-dir", default="4_toys_lo1200", help="results/<channel>/<toys-dir>/ with the study's toys")
    ap.add_argument("--truth-dir", default="3_truths", help="results/<channel>/<truth-dir>/ to take the truth from (3_truths_kde for the RooKeysPdf-alone truth)")
    ap.add_argument("--tag", default="", help="extra folder-name tag (e.g. split9 for a truth built from the nine processes)")
    ap.add_argument("--truth-label", default=None, help="the smoother line of the figures (default: by truth name)")
    ap.add_argument("--inject-figs", nargs="+", type=float, default=[0.0, 1.0], help="single-toy figures at these injections")
    ap.add_argument("--ntoys", type=int, default=0, help="throw and fit this many toys per candidate and injection here (0: the study's)")
    ap.add_argument("--toy-max-pull", type=float, default=None,
                    help="the single-toy figure shows the first passing toy with |N_sig - injected| / sigma at most this; "
                         "the first passing toy of all is saved beside it as toyfit_<cand>_inj<k>_first")
    ap.add_argument("--toy-edge-pull", type=float, default=None,
                    help="also save, at zero injection, the first passing toy whose pull is below minus this as "
                         "toyfit_<cand>_inj0_edge (a fit that ends where S+B reaches zero)")
    ap.add_argument("--toy-sb-only", action="store_true",
                    help="the single-toy figures draw one fitted curve, labelled 'S+B fit', without the separate background curve")
    ap.add_argument("--toy-ratio", choices=("pull", "data_over_fit"), default="pull",
                    help="the lower panel of the single-toy figures: (data - fit) / sqrt(fit) per bin (default), or data / fit "
                         "for the bins with toy events, saved beside the default as toyfit_..._ratio")
    ap.add_argument("--toys-only", action="store_true",
                    help="stop after the single-toy figures and write their numbers to toys.md: no distributions, numbers.md "
                         "untouched (--ntoys as in the cell's own run puts them in that cell's folder; without it the folder "
                         "has no toy-count tag)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2500, help="toys per worker job")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    from wrplotter.cli_utils import setup_logging
    setup_logging(args.verbose)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import ROOT
    import published_functions as pf
    import plots as P
    from sbfit import FIT_HI, SBModel, bkg_only, signal_pars, signal_shape
    from spectrum import centers, rebin
    from truth_sets import load_truth
    from pulls import gauss_curve, gauss_fit, pull_stats
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning

    ch, variant, truth_name, m = args.channel, args.variant, args.truth, args.mass
    tag = ((f"_k{args.k:g}" if args.k else "") + (f"_{args.tag}" if args.tag else "")
           + (f"_{args.ntoys // 1000}k" if args.ntoys else ""))
    out = C.RESULTS / ch / "11_bias_explainer" / (f"{variant}_{truth_name}_m{m:.0f}" + tag)
    out.mkdir(parents=True, exist_ok=True)
    edges10, parts, _ = load_truth(C.RESULTS / ch / args.truth_dir / variant / f"truth_{truth_name}")
    e, t, var = rebin(edges10, parts["sum"], np.maximum(parts["sum"], 0.0))
    c = centers(e)
    mu, sigma = signal_pars(ch, m)
    if args.k:                                                   # the window of 6_windows.py
        lo, hi = max(mu - args.k * sigma, args.lo), min(mu + args.k * sigma, FIT_HI)
        range_txt = rf"$\pm{args.k:g}\sigma$ window: {lo:.0f}" + "\u2013" + f"{hi:.0f}"      # short: it sits beside the label block
        range_md = f"window +-{args.k:g} sigma = [{lo:.0f}, {hi:.0f}] GeV"
    else:
        lo, hi = args.lo, FIT_HI
        range_txt = f"fit range {lo:.0f}" + "\u2013" + f"{hi:.0f} GeV"
        range_md = f"fits over [{lo:.0f}, {hi:.0f}] GeV"
    XMAX = 5000.0 if hi <= 5000.0 else FIT_HI        # the spectrum figures end at 5 TeV unless the window reaches past it
    leg_title = rf"$\pm{args.k:g}\sigma$ window" if args.k else None   # the window named on every legend (short: it must fit beside the label)
    # the spectrum figures: a short legend entry for the expectation, the smoother named in the region text
    expected_lab = "expected (raw MC)" if truth_name == "raw" else "expected (smoothed MC)"
    # the label block of every figure: channel, region, the background model
    # ("Raw MC Bkg" / "RooKeysPdf Bkg"), the candidate at the mass; no DY-sample or process-count line
    TRUTH_BKG = {"raw": "Raw MC Bkg", "kde": "RooKeysPdf Bkg", "kde_expo": "RooKeysPdf + tail Bkg",
                 "kde_powexp": "RooKeysPdf + tail Bkg", "kernel_expo": "Kernel + tail Bkg", "kernel_powexp": "Kernel + tail Bkg"}
    CAND_SHORT = {"expo": "Expo", "powexp": r"Power $\times$ expo", "atlasemu": "Running power law",
                  "powlaw": "Power law", "dilep8": "Power $\times$ exp(quadratic)", "wprime7": "W' function"}
    bkg_label = args.truth_label or TRUTH_BKG.get(truth_name, f"{truth_name} Bkg")

    # the functions written out for the label of the fit figure (m0 = the pivot, 2 TeV; c per TeV)
    m0_txt = rf"$m_0$ = {C.PIVOT / 1000:g} TeV"
    EQUATION = {"atlasemu": r"$N\,(m/m_0)^{\,b\,+\,c\,\ln(m/m_0)}$, " + m0_txt,
                "powexp": r"$N\,(m/m_0)^{\,b}\,e^{\,c\,(m - m_0)}$, " + m0_txt,
                "expo": r"$N\,e^{\,c\,(m - m_0)}$, " + m0_txt}

    def cand_at(cand, wrap=False):
        """The function at the mass; wrap=True breaks a long name before the mass (the histograms' label block
        sits beside the lines at zero)."""
        short = CAND_SHORT.get(cand, cand)
        return f"{short}{chr(10) if wrap and len(short) > 14 else ' '}at {m / 1000:g} TeV"
    spec_extra = bkg_label
    shape = signal_shape(e, mu, sigma)
    core = (c >= mu - C.SIG_CORE_K * sigma) & (c <= mu + C.SIG_CORE_K * sigma)
    b_core = float(t[core].sum())
    from sbfit import nsig_limits
    nsig_bounds = nsig_limits(b_core)                         # None: N_sig is free (config.NSIG_LIMITED for the old guard)
    seed = C.SEED + 1000 * int(m) + zlib.crc32(f"{ch}/{variant}/{truth_name}".encode()) % 997
    region = bkg_label
    cands = sorted(args.candidates, key=lambda n: pf.FUNCS[n][0])
    npz_path = C.RESULTS / ch / args.toys_dir / variant / f"nsig_{truth_name}.npz"
    if npz_path.exists():
        npz = np.load(npz_path)
        rows = {cand: load_rows(npz_path.with_name(f"toys_{truth_name}.csv"), cand, m) for cand in cands}
    elif args.ntoys > 0 or args.toys_only:                     # a truth the study has no toys for: all toys thrown here
        npz = type("NoToys", (), {"files": []})()
        rows = {cand: {} for cand in cands}
        logger.info("no study toys at %s: the distributions use the %d toys thrown here", npz_path, args.ntoys)
    else:
        raise SystemExit(f"no toys at {npz_path}; pass --ntoys to throw them here")
    L = [f"# The bias test, one cell: {ch} {C.REGION_NAME}, {C.DY_LABEL[variant]}, truth {truth_name} ({args.truth_dir}), "
         f"m_WR = {m:.0f} GeV, {range_md}", "",
         f"Signal: Gaussian mu = {mu:.0f} GeV, sigma = {sigma:.0f} GeV (the window map); core +-{C.SIG_CORE_K:g} sigma holds "
         f"{b_core:.1f} background events; " + ("N_sig free (no limits)." if nsig_bounds is None else f"N_sig bounds [{nsig_bounds[0]:.0f}, {nsig_bounds[1]:.0f}]."), ""]

    # ---- the fits: background-only seeds, Asimov, then single toys ---------------------------------
    parents, models, asimov = {}, {}, {}
    for cand in cands:
        fb = bkg_only(cand, e, t, var, lo, hi, parents)
        parents[cand] = fb
        models[cand] = SBModel(cand, mu, sigma, lo, hi)
        asimov[cand] = models[cand].fit(e, t, fb["params"], 0.0, nsig_bounds)
        a = asimov[cand]
        L.append(f"- {cand}: Asimov N_sig = {a['nsig']:+.2f} +- {a['nsig_err']:.2f} (sigma_fit); expected limit 1.96 sigma_fit = "
                 f"{1.96 * a['nsig_err']:.1f} events")
    L.append("")
    def mcfit_title(cand, a):
        """The label of the background-only fit figure. Over the full range the fit does not depend on the signal
        mass, so the label names the function and its equation only (no signal mass and no S+B
        number on a background-only fit); in a window the mass defines the fit range and the S+B fit of the same
        spectrum is quoted with it."""
        eq = EQUATION[cand] + "\n" if cand in EQUATION else ""
        if not args.k:
            return (CAND_SHORT.get(cand, cand) + "\n" + eq).rstrip("\n")
        return f"{cand_at(cand)}\n" + eq + rf"Asimov S+B: $N_{{\rm sig}}$ = {a['nsig']:+.1f} $\pm$ {a['nsig_err']:.1f}"

    # ---- the expected spectrum with the background-only fit (what the fit is asked to describe) -----
    sel = c >= C.THRESHOLD
    infit = (c >= lo) & (c <= hi)
    for cand in cands:
        fb, model, a = parents[cand], models[cand], asimov[cand]
        for i, pv in enumerate(fb["params"]):
            model.fb.SetParameter(i, float(pv))
        fit_vals = np.array([model.fb.Eval(x) for x in c])
        col = EXPL_COLOR.get(cand, pf.COLOR[cand])
        pull = np.where(fit_vals > 0, (t - fit_vals) / np.sqrt(np.where(fit_vals > 0, fit_vals, 1.0)), np.nan)
        chi2 = float(np.nansum(pull[infit] ** 2))
        ndf = int(infit.sum()) - pf.FUNCS[cand][0]
        top_t = float(np.nanmax(t[sel & (c <= 5000.0)]))
        fig, (ax, axr) = P.spectrum_figure(12)
        P.step_hist(ax, e, t, var, "0.55", expected_lab, errors=False)
        ax.plot(c[infit], np.maximum(fit_vals[infit], P.YMIN), color=col, lw=2.2, label="background-only fit")
        for xe in ((lo, hi) if args.k else ()):               # a window's edges; the full range has no marker
            if xe <= 5000.0:
                ax.plot([xe, xe], [P.YMIN, 3.0 * top_t], color="0.3", lw=1.2, ls=(0, (2, 2)), label=range_txt if xe == lo else None)
        axr.plot(c[infit], pull[infit], "o", ms=4, color="k")
        for yv in (-2, 2):
            axr.axhline(yv, color="0.5", lw=0.6, ls="--")
        P.finish(ax, axr, channel=ch, xmax=XMAX, top=top_t, log=True, ymin=1e-2, extra=spec_extra,
                 ratio_label=r"(MC $-$ fit) / $\sqrt{\rm fit}$", ratio_lim=(-4, 4), ratio_ref=0.0,
                 legend_title=mcfit_title(cand, a))
        P.save(fig, out / f"mcfit_{cand}")
        L.append(f"- {cand}: background-only fit to the expected spectrum, sum of (expected - fit)^2 / fit over the "
                 f"{int(infit.sum())} bins of the range = {chi2:.1f} for {ndf} degrees of freedom")
    L.append("")
    rng = ROOT.TRandom3(seed)
    for k in args.inject_figs:
        ref = cands[0]
        n_inj = k * 1.96 * asimov[ref]["nsig_err"]
        expect = t + n_inj * shape
        for cand in cands:
            model = models[cand]
            # the toy shown is the first one whose fit passes the checks: at 6 TeV most
            # toys leave N_sig on its bound, and a frame should show a fit that counts
            # the toy shown: the first whose fit passes the checks; with --toy-max-pull, the first such toy whose pull
            # |N_sig - injected| / sigma is at most that (a typical outcome), the very first passing toy being kept
            # as toyfit_<cand>_inj<k>_first
            toy, r, tries, first, typical, edge = None, None, 0, None, None, None
            want_edge = args.toy_edge_pull is not None and k == 0
            for tries in range(1, 3001):
                toy = np.array([rng.Poisson(max(x, 0.0)) for x in expect], float)
                r = model.fit(e, toy, asimov[cand]["bkg"], n_inj, nsig_bounds)
                if r is None or not r["passed"]:
                    continue
                pull_ = (r["nsig"] - n_inj) / r["nsig_err"] if r["nsig_err"] > 0 else float("inf")
                if first is None:
                    first = (toy, r, tries)
                if typical is None and (args.toy_max_pull is None or abs(pull_) <= args.toy_max_pull):
                    typical = (toy, r, tries)
                # a toy whose fit ends where S+B reaches zero: a pull far below the histogram's range (--toy-edge-pull)
                if want_edge and edge is None and pull_ < -args.toy_edge_pull:
                    edge = (toy, r, tries)
                if typical is not None and (edge is not None or not want_edge):
                    break
            if first is None:
                logger.warning("%s: no toy passed the checks", cand)
                continue
            toy, r, tries = typical or first
            shown = [(toy, r, tries, "")]
            if args.toy_max_pull is not None:
                shown.append((*first, "_first"))
            if edge is not None:
                shown.append((*edge, "_edge"))
            for toy, r, tries, tsuffix in shown:
                for i_, pv_ in enumerate(r["bkg"]):
                    model.fb.SetParameter(i_, float(pv_))
                fb_vals = np.array([model.fb.Eval(x) for x in c])
                total = fb_vals + r["nsig"] * shape
                sel = c >= C.THRESHOLD
                infit = (c >= lo) & (c <= hi)
                col = EXPL_COLOR.get(cand, pf.COLOR[cand])
                inj_txt = f"{n_inj:.0f} injected" if k else "no injection"
                for xmin_, xmax, log, suffix in ((None, XMAX, True, ""),):       # drawn to XMAX; the fit runs to FIT_HI
                    fig, (ax, axr) = P.spectrum_figure(12)                    # room for the legend and its title
                    P.step_hist(ax, e, t, var, "0.55", expected_lab, errors=False)      # a histogram: steps, not a line
                    ax.errorbar(c[sel], toy[sel], yerr=np.sqrt(np.maximum(toy[sel], 0.0)), fmt="o", ms=4, color="k",
                                label="pseudo-data (Poisson toy)")
                    fbv = fb_vals[infit] if not log else np.maximum(fb_vals[infit], P.YMIN)
                    tot = total[infit] if not log else np.maximum(total[infit], P.YMIN)
                    if args.toy_sb_only:                               # one curve, "S+B fit"
                        ax.plot(c[infit], tot, color=SB_COLOR, lw=2.2, label="S+B fit")
                    else:
                        ax.plot(c[infit], fbv, color=col, lw=1.8, ls="--", label="fit: background")
                        ax.plot(c[infit], tot, color=SB_COLOR, lw=2.2, label="fit: background + signal")
                    # the fitted signal is the gap between the two curves, as CMS fit plots draw it: no shading
                    top_toy = float(np.nanmax(toy[sel & (c <= xmax)]))
                    for xe in ((lo, hi) if args.k else ()):           # a window's edges; the full range has no marker
                        if xe <= xmax:
                            ax.plot([xe, xe], [0.0 if not log else P.YMIN, (0.5 if not log else 3.0) * top_toy], color="0.3",
                                    lw=1.2, ls=(0, (2, 2)), label=range_txt if xe == lo else None)
                    if args.toy_ratio == "data_over_fit":
                        # data / fit: the bins with toy events only, as in the upper panel; the
                        # bins where S+B is not positive have no ratio
                        has = infit & (toy > 0) & (total > 0)
                        axr.errorbar(c[has], toy[has] / total[has], yerr=np.sqrt(toy[has]) / total[has], fmt="o", ms=4, color="k")
                        ratio_kw = {"ratio_label": "data / fit", "ratio_lim": (0.0, 3.0), "ratio_ref": 1.0}
                    else:
                        ratio = np.where(total > 0, (toy - total) / np.sqrt(np.where(total > 0, total, 1.0)), np.nan)
                        axr.plot(c[infit], ratio[infit], "o", ms=4, color="k")             # one pull per bin, as for data
                        for yv in (-2, 2):
                            axr.axhline(yv, color="0.5", lw=0.6, ls="--")
                        ratio_kw = {"ratio_label": r"(data $-$ fit) / $\sqrt{\rm fit}$", "ratio_lim": (-4, 4), "ratio_ref": 0.0}
                    P.finish(ax, axr, channel=ch, xmax=xmax, xmin=xmin_, top=top_toy, log=log, ymin=1e-2,
                             extra=spec_extra,                                  # the fit range is in the legend
                             **ratio_kw,
                             # short title lines: a line wider than the room beside the CMS label block sends the
                             # legend under the label and fit_header stretches the axis
                             legend_title=(f"{cand_at(cand)}\n"
                                           rf"Gaussian $\mu$ = {mu:.0f}, $\sigma$ = {sigma:.0f} GeV" + "\n"
                                           + (f"injected: {n_inj:.0f} events" if k else "injected: none") + "\n"
                                           rf"fitted: $N_{{\rm sig}}$ = {r['nsig']:+.1f} $\pm$ {r['nsig_err']:.1f}"))
                    P.save(fig, out / (f"toyfit_{cand}_inj{k:g}{suffix}{tsuffix}" + ("_ratio" if args.toy_ratio == "data_over_fit" else "")))
                L.append(f"- toy{tsuffix} at {k:g}x ({n_inj:.1f} injected), {cand}: N_sig = {r['nsig']:+.2f} +- {r['nsig_err']:.2f}, "
                         f"N_sig / sigma = {(r['nsig'] - n_inj) / r['nsig_err']:+.2f} against the injection; checks {r['checks']}; "
                         f"toy #{tries}" + (" (passes the checks)" if r["passed"] else " (none of the toys drawn passed)"))
    L.append("")
    if args.toys_only:
        (out / "toys.md").write_text("\n".join(L) + "\n")       # the single toys' numbers (numbers.md is the full cell's)
        print("\n".join(L))
        return

    # ---- the distributions: the study's toys, or --ntoys thrown and fitted here -----------------------
    own = {}
    if args.ntoys > 0:
        jobs = []
        for ci, cand in enumerate(cands):
            for ki, k in enumerate((0.0, 1.0, 2.0)):
                n_inj = k * 1.96 * asimov[cand]["nsig_err"]
                expect = t + n_inj * shape
                for j, start in enumerate(range(0, args.ntoys, args.chunk)):
                    jobs.append({"key": (cand, k), "cand": cand, "mu": mu, "sigma": sigma, "lo": lo, "hi": hi,
                                 "edges": e, "expect": expect, "bkg_seed": asimov[cand]["bkg"], "n_inj": n_inj,
                                 "nsig_bounds": nsig_bounds, "n": min(args.chunk, args.ntoys - start),
                                 "seed": seed + 1_000_000 * (1 + ci) + 100_000 * ki + j + 1})
        logger.info("%d toys per candidate and injection in %d jobs on %d workers", args.ntoys, len(jobs), args.workers)
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for key, n_, e_, ok_ in pool.imap_unordered(toy_job, jobs):
                old = own.get(key)
                own[key] = (n_, e_, ok_) if old is None else tuple(np.concatenate([o, x]) for o, x in zip(old, (n_, e_, ok_)))
        for cand in cands:
            for k in (0.0, 1.0, 2.0):
                rows[cand][k] = {"n_inj": k * 1.96 * asimov[cand]["nsig_err"], "asimov_err": asimov[cand]["nsig_err"]}
                n_, e_, ok_ = own[(cand, k)]
                if ok_.sum() >= 10:
                    rows[cand][k].update(pull_stats((n_[ok_] - rows[cand][k]["n_inj"]) / e_[ok_]))

    def arrays(cand, k):
        if (cand, k) in own:
            return own[(cand, k)]
        key = f"{cand}_{int(m)}_{k:g}"
        return npz[key], npz[key + "_err"], npz[key + "_ok"].astype(bool)

    # a candidate whose fits almost never converge (three parameters on an empty window at 6 TeV) has no
    # distributions to draw: it is recorded and left out of the toy figures
    dropped = [c for c in cands if len(arrays(c, 0.0)[0]) < 10]
    for c in dropped:
        L += [f"## {c}: only {len(arrays(c, 0.0)[0])} converged fits at zero injection: no distributions", ""]
    cands = [c for c in cands if c not in dropped]
    stats, gfits = {}, {}
    for cand in cands:
        n, err, ok = arrays(cand, 0.0)
        # the distribution of N_sig is that of the fits passing the checks, the toys of the pulls: with N_sig free
        # (no limits since 2026-11-16) a fit that does not converge can return any value
        npass = n[ok] if ok.any() else n
        med, mean, rms = float(np.median(npass)), float(npass.mean()), float(npass.std())
        # with N_sig free, a toy with no events under the peak is fitted with a large negative N_sig and a large
        # error (its pull stays moderate): the mean and the RMS in events are then set by that tail, so the figure
        # and the numbers use the median and the central 68 % of the fits
        q16, q84, q025, q975 = (float(x) for x in np.percentile(npass, [16.0, 84.0, 2.5, 97.5]))
        wid68 = 0.5 * (q84 - q16)
        free = nsig_bounds is None
        if free:
            rms = wid68
        stats[cand] = (n, err, ok, med, mean, rms)
        gfits[cand] = g = gauss_fit(n[ok] / err[ok])             # the Gaussian fitted to the pulls (CMS)
        row = rows[cand].get(0.0, {})
        a_err = row.get("asimov_err", asimov[cand]["nsig_err"])
        L += [f"## {cand}: N_sig over {len(n)} toys, zero injection", "",
              f"median {med:+.2f}, mean {mean:+.2f}, RMS {rms:.2f}, Asimov sigma_fit {a_err:.2f}; S_spur = median / RMS = "
              f"{med / rms:+.2f}; converged {int(ok.sum())} of {len(n)}; pull Gaussian fit: mean {g['mean']:+.2f}, "
              f"sigma {g['sigma']:.2f}", "",
              f"robust: central 68 % of the passing fits [{q16:+.2f}, {q84:+.2f}], half-width {wid68:.2f}; "
              f"{100.0 * float((npass < q025 - 2.0 * wid68).mean()):.2f} % of them below {q025 - 2.0 * wid68:.1f} events"
              + (" (N_sig free: the RMS above is this half-width)" if free else ""), ""]
        # N_sig distribution: median, mean, +-RMS band
        col = EXPL_COLOR.get(cand, pf.COLOR[cand])
        fig, ax = P.scan_figure()
        # a plain histogram of the fitted N_sig with the injected value marked, as the pull figure (no
        # median, band or off-axis text on it). The range only: with N_sig free it covers the central 95 % of the
        # fits and two half-widths of their central 68 % beyond; with the old limits, four RMS around the median.
        if free:
            bins = np.linspace(q025 - 2.0 * wid68, q975 + 2.0 * wid68, 41 if len(n) < 5000 else 121)
        else:
            bins = np.linspace(med - 4.0 * rms, med + 4.0 * rms, 41 if len(n) < 5000 else 121)
        ax.hist(npass, bins=bins, histtype="step", lw=2.0, color=col)
        ax.axvline(0.0, color="k", lw=1.0)
        handles = [Line2D([], [], color=col, lw=2.0, label="toys"),
                   Line2D([], [], color="k", lw=1.0, label="injected: 0")]
        ax.set_xlabel(r"fitted $N_{\rm sig}$ [events]")
        ax.set_ylabel(f"toys / {bins[1] - bins[0]:.1f} events")
        cms_label(ax, C.LUMI, C.COM, data=False,
                  text=P.region_text(ch, f"{region}\n{cand_at(cand, wrap=True)}"), year=C.YEAR)
        ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE_LEGEND, title=leg_title,
                  title_fontsize=FONT_SIZE_LEGEND, alignment="left")
        fit_header(ax)
        P.save(fig, out / f"nsig_dist_{cand}")
        # the fit error per toy against the RMS
        fig, ax = P.scan_figure()
        e_ok = err[ok]
        bins = np.linspace(0.0, max(2.0 * rms, float(np.percentile(e_ok, 99))), 41 if len(e_ok) < 5000 else 121)
        ax.hist(e_ok, bins=bins, histtype="step", lw=2.0, color=col)
        ax.axvline(rms, color="k", lw=2.0)
        ax.axvline(a_err, color="0.5", lw=1.6, ls="--")
        handles = [Line2D([], [], color=col, lw=2.0, label=rf"$\sigma(N_{{\rm sig}})$ per toy"),
                   Line2D([], [], color="k", lw=2.0, label=f"RMS of $N_{{\\rm sig}}$: {rms:.1f}"),
                   Line2D([], [], color="0.5", lw=1.6, ls="--", label=rf"Asimov $\sigma_{{\rm fit}}$: {a_err:.1f}")]
        ax.set_xlabel(r"$\sigma(N_{\rm sig})$ of the fit [events]")
        ax.set_ylabel(f"toys / {bins[1] - bins[0]:.1f} events")
        cms_label(ax, C.LUMI, C.COM, data=False,
                  text=P.region_text(ch, f"{region}\n{cand_at(cand, wrap=True)}\n"
                                         f"pull $\\sigma$ {gfits[cand]['sigma']:.2f}"), year=C.YEAR)
        ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE_LEGEND, title=leg_title,
                  title_fontsize=FONT_SIZE_LEGEND, alignment="left")
        fit_header(ax)
        P.save(fig, out / f"nsig_err_{cand}")
        # the pull N_sig / sigma(N_sig) over the toys with the Gaussian fitted to its histogram (the CMS
        # convention: EXO-17-005 Sec. 4.1, the Combine parametric tutorial); its mean is the bias, its
        # sigma the check of the fit error
        if ok.sum() < 10:                                   # no passing fits: no pull or injection figures
            L.append(f"- {cand}: only {int(ok.sum())} toys passed the checks: no pull or injection figures")
            continue
        pulls = n[ok] / err[ok]
        g = gfits[cand]
        p_mean, p_std = float(pulls.mean()), float(pulls.std())
        fig, ax = P.scan_figure()
        bins = np.linspace(-5.0, 5.0, g["nbins"] + 1 if g["ok"] else (41 if len(pulls) < 5000 else 101))
        ax.hist(pulls, bins=bins, histtype="step", lw=2.0, color=col)
        ax.axvline(0.0, color="k", lw=1.0)
        handles = [Line2D([], [], color=col, lw=2.0, label="toys"), Line2D([], [], color="k", lw=1.0, label="injected: 0")]
        if g["ok"]:
            xg = np.linspace(-5.0, 5.0, 401)
            ax.plot(xg, gauss_curve(g, xg), color=FIT_COLOR, lw=2.2)
            ax.axvline(g["mean"], color=FIT_COLOR, lw=1.6, ls="--")
            handles += [Line2D([], [], color=FIT_COLOR, lw=2.2, label="Gaussian fit"),
                        Line2D([], [], color=FIT_COLOR, lw=1.6, ls="--", label=rf"$\mu$ = {g['mean']:+.2f}, $\sigma$ = {g['sigma']:.2f}")]
        ax.set_xlabel(r"$N_{\rm sig}$ / $\sigma(N_{\rm sig})$, no signal injected")
        ax.set_ylabel(f"toys / {bins[1] - bins[0]:.2f}")
        cms_label(ax, C.LUMI, C.COM, data=False,
                  text=P.region_text(ch, f"{region}\n{cand_at(cand, wrap=True)}"), year=C.YEAR)
        ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE_LEGEND, title=leg_title,
                  title_fontsize=FONT_SIZE_LEGEND, alignment="left")
        fit_header(ax)
        P.save(fig, out / f"nsig_pull_{cand}")
        L.append(f"- pull N_sig / sigma over the converged toys: mean {p_mean:+.2f}, std {p_std:.2f}; Gaussian fit mean "
                 f"{g['mean']:+.2f} +- {g['mean_err']:.2f}, sigma {g['sigma']:.2f} +- {g['sigma_err']:.2f} "
                 f"({g['nbins']} bins in [-5, 5], chi2/ndf {g['chi2']:.1f}/{g['ndf']})")
        # injections
        fig, ax = P.scan_figure()
        styles = {0.0: ("-", "nothing injected"), 1.0: ("--", r"1$\times$ limit injected"), 2.0: (":", r"2$\times$ limit injected")}
        passing = lambda k: (lambda a: a[0][a[2]] if a[2].any() else a[0])(arrays(cand, k))     # noqa: E731
        allv = np.concatenate([passing(k) for k in (0.0, 1.0, 2.0) if (cand, k) in own or f"{cand}_{int(m)}_{k:g}" in npz.files])
        bins = np.linspace(float(np.percentile(allv, 0.5)) - rms, float(np.percentile(allv, 99.5)) + rms, 46 if len(allv) < 15000 else 136)
        handles = []
        top_hist = max(float(np.histogram(passing(k), bins=bins)[0].max())
                       for k in styles if (cand, k) in own or f"{cand}_{int(m)}_{k:g}" in npz.files)
        for k, (ls, lab) in styles.items():
            key = f"{cand}_{int(m)}_{k:g}"
            if (cand, k) not in own and key not in npz.files:
                continue
            nk = passing(k)
            n_inj = rows[cand].get(k, {}).get("n_inj", k * 1.96 * a_err)
            ax.hist(nk, bins=bins, histtype="step", lw=2.0, color=col, ls=ls)
            # marker lines in data units (an axes-fraction line follows every raise of the axis)
            ax.plot([n_inj, n_inj], [0.0, 1.15 * top_hist], color="k", lw=1.2, ls=ls)
            ax.plot([float(nk.mean())] * 2, [0.0, 1.15 * top_hist], color=col, lw=1.2, ls=ls)
            handles.append(Line2D([], [], color=col, lw=2.0, ls=ls, label=f"inj. {n_inj:.0f}: mean {nk.mean():+.1f}"))
            L.append(f"- injection {k:g}x ({n_inj:.1f}): mean N_sig {nk.mean():+.2f}, median {np.median(nk):+.2f}, RMS {nk.std():.2f}, "
                     f"bias (mean - injected) / RMS = {(nk.mean() - n_inj) / nk.std():+.2f}")
        handles.append(Line2D([], [], color="k", lw=1.2, label="black: injected"))
        ax.set_xlim(bins[0], bins[-1] + 0.6 * (bins[-1] - bins[0]))      # room on the right for the legend
        ax.set_xlabel(r"fitted $N_{\rm sig}$ [events]")
        ax.set_ylabel(f"toys / {bins[1] - bins[0]:.1f} events")
        cms_label(ax, C.LUMI, C.COM, data=False,
                  text=P.region_text(ch, f"{region}\n{cand_at(cand, wrap=True)}"), year=C.YEAR)   # the legend names the injections
        ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE_LEGEND, title=leg_title,
                  title_fontsize=FONT_SIZE_LEGEND, alignment="left")
        fit_header(ax)
        P.save(fig, out / f"nsig_inject_{cand}")
        L.append("")

    # the candidates side by side
    fig, ax = P.scan_figure()
    rms_ref = max(s[5] for s in stats.values())
    lo_x = min(s[3] for s in stats.values()) - 4 * rms_ref
    hi_x = max(s[3] for s in stats.values()) + 4 * rms_ref
    bins = np.linspace(lo_x, hi_x, 46 if len(stats[cands[0]][0]) < 5000 else 136)
    handles = []
    for cand in cands:
        n, err, ok, med, mean, rms = stats[cand]
        col = EXPL_COLOR.get(cand, pf.COLOR[cand])
        ax.hist(n[ok] if ok.any() else n, bins=bins, histtype="step", lw=2.0, color=col)     # the fits that pass
        ax.axvline(med, color=col, lw=2.4)
        handles.append(Line2D([], [], color=col, lw=2.0, label=f"{cand}: median {med:+.1f}"))
    ax.axvline(0.0, color="k", lw=1.0)
    handles.append(Line2D([], [], color="k", lw=1.0, label="injected: 0"))
    ax.set_xlabel(r"fitted $N_{\rm sig}$ [events]")
    ax.set_ylabel(f"toys / {bins[1] - bins[0]:.1f} events")
    cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(ch, f"{region}\n$m_{{W_R}}$ = {m / 1000:g} TeV, no injection"), year=C.YEAR)
    ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE_LEGEND, title=leg_title,
              title_fontsize=FONT_SIZE_LEGEND, alignment="left")
    fit_header(ax)
    P.save(fig, out / "nsig_dist_compare")
    (out / "numbers.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
