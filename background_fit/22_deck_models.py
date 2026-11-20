#!/usr/bin/env python3
"""The background models of the study deck on one page: the raw MC, the RooKeysPdf, the
RooKeysPdf with an exponential tail per process, and the exponential of a quadratic fitted to the summed MC.

  models  per channel, results*/<ch>/22_models/:
            expo2_fit    the summed MC with the exponential of a quadratic fitted from 1200 GeV, alone;
                         below, (MC - fit) / sigma_MC
            cumulative   N(m > m_min) of the MC (band = its statistical error) and of the three curves, from the
                         fit floor up; below, curve / MC
            slices       events per mass slice for the MC and the three curves; below, curve - MC in events with
                         the MC error band
          and results*/MODELS.md (the numbers of both figures).
  bias    the +-5 sigma window from 1200 GeV with one function on the four models: the Gaussian mu and sigma of
          the zero-injection pulls against the mass, results*/<ch>/22_models/bias4_<function>_{pull_mean,width}
          (toys 6_windows/k5_lo1200<tag>/ for the raw MC, the RooKeysPdf and the RooKeysPdf + expo tail,
          k5_lo1200<tag>_fit/ for the fitted curve).

  cd WR_Plotter && FIT_DATASET=run3 python background_fit/22_deck_models.py models|bias [--tag _100k]
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))
import config as C                                                       # noqa: E402

logger = logging.getLogger("22_deck_models")
VARIANT = "lo_ht"
FLOOR = 1200.0
XMAX = 8000.0                    # the coarse per-slice figure (its last slice ends at 8 TeV)
MC_COLOR = "#717581"
# key -> (results sub-directory, file, legend label, colour, line style)
MODELS = {
    "kde9": ("3_truths_kde9", "truth_kde", "RooKeysPdf", "#7a21dd", "-"),
    # model 3: the RooKeysPdf sum with ONE exponential tail from 4 TeV (the sumtail step); the
    # per-process tails it replaced are PER_PROCESS_TAIL
    "kde_expo9": ("3_truths_sumtail", "truth_kde_sumexpo", "RooKeysPdf + expo tail", "#5790fc", "--"),
    "fitexpo2": ("3_truths_fit", "truth_fitexpo2", "exp. of a quadratic fit", "#e42536", "-"),
}
PER_PROCESS_TAIL = ("3_truths_split9", "truth_kde_expo")      # one exponential tail per process, summed
SLICES = [1200.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 5000.0, 6000.0, 8000.0]
REF = [2000.0, 3000.0, 4000.0, 5000.0, 6000.0]
XLABEL = r"$m_\mathrm{min}$ [GeV]"
YLABEL = r"Events with $m_{\ell\ell jj} > m_\mathrm{min}$"
# the bias figure: key -> (window folder suffix, truth name in the files, legend label, colour, line style, marker)
BIAS_MODELS = [("raw", "", "raw", "Raw MC", MC_COLOR, "-", "o"),
               ("kde9", "", "kde9", "RooKeysPdf", "#7a21dd", "--", "s"),
               ("kde_expo9", "_sumtail", "kde_sumexpo", "RooKeysPdf + expo tail", "#5790fc", "-.", "^"),
               ("fitexpo2", "_fit", "fitexpo2", "exp. of a quadratic fit", "#e42536", ":", "D")]
CAND_LABEL = {"expo": "Expo", "powexp": r"Power $\times$ expo"}
PASS = 0.9
CUM_YMIN = 1e-3                  # the floor of the cumulative figure: the MC holds 2e-3 events above 8 TeV


def rcum(x):
    """N(>= e_i): the sum of the bins from i on."""
    return np.cumsum(np.asarray(x, float)[::-1])[::-1]


def mc_first(ax, P):
    h, l = ax.get_legend_handles_labels()
    idx = sorted(range(len(l)), key=lambda i: 0 if l[i].startswith(("MC,", "Simulation")) else 1 if l[i].startswith("MC stat") else 2)
    ax.legend([h[i] for i in idx], [l[i] for i in idx], loc="upper right", fontsize=P.FONT_SIZE_LEGEND, handlelength=1.6)


def run_models(args):
    import plots as P
    from spectrum import centers, load_components, rebin
    from truth_sets import load_truth
    L = [f"# The background models, {C.ERA} {C.REGION_NAME} ({C.LUMI:.1f} fb-1), LO HT-binned DY, nine processes", "",
         f"Curves from {FLOOR:.0f} GeV (the fit floor of the windows). In brackets: the difference to the MC in units of "
         "the MC statistical error.", ""]
    for ch in args.channels:
        out = C.RESULTS / ch / "22_models"
        out.mkdir(parents=True, exist_ok=True)
        comps10 = load_components(ch, VARIANT, "split9")
        e10, v10, var10 = comps10["sum"]
        c10 = centers(e10)
        curves = {}
        for key, (sub, name, *_r) in MODELS.items():
            p = C.RESULTS / ch / sub / VARIANT / name
            if p.with_suffix(".npz").exists():
                curves[key] = np.asarray(load_truth(p)[1]["sum"], float)
            else:
                logger.warning("%s: no %s", ch, p)
        # --- the exponential of a quadratic alone on the MC -------------------------------------------------
        if "fitexpo2" in curves:
            e, v, var = rebin(e10, v10, var10)
            _, f100, _ = rebin(e10, curves["fitexpo2"], np.zeros_like(v10))
            c = centers(e)
            sig = np.sqrt(np.maximum(var, 0.0))
            sel = (c >= C.THRESHOLD) & (c <= P.XMAX_FULL)
            fit = sel & (c >= FLOOR)
            fig, (ax, axr) = P.spectrum_figure()
            ax.errorbar(c[sel], v[sel], yerr=sig[sel], fmt="o", ms=4, color="k", label="Simulation")
            ax.plot(c[fit], np.maximum(f100[fit], P.YMIN_FULL), color=MODELS["fitexpo2"][3], lw=1.8, label="exp. of a quadratic")
            pull = np.where(sig > 0, (v - f100) / np.where(sig > 0, sig, 1.0), np.nan)
            axr.stairs(pull[fit], np.append(e[:-1][fit], e[1:][fit][-1]), color=MODELS["fitexpo2"][3], lw=1.4)
            for y in (-2, 2):
                axr.axhline(y, color="0.5", lw=0.6, ls="--")
            P.finish(ax, axr, channel=ch, xmax=P.XMAX_FULL, top=float(v[sel].max()), ymin=P.YMIN_FULL,
                     extra=f"{C.DY_SHORT[VARIANT]}\nfit from {FLOOR:.0f} GeV", ratio_label=r"(MC $-$ fit) / $\sigma_{\rm MC}$",
                     ratio_lim=(-5, 5), ratio_ref=0.0)
            axr.set_ylabel(r"(MC $-$ fit) / $\sigma_{\rm MC}$", fontsize=20)
            mc_first(ax, P)
            P.save(fig, out / "expo2_fit")
        # --- N(m > m_min) -----------------------------------------------------------------------------------
        lo_e = e10[:-1]
        N_mc, E_mc = rcum(v10), np.sqrt(rcum(var10))
        sel = (lo_e >= FLOOR) & (lo_e <= P.XMAX_FULL)
        fig, (ax, axr) = P.spectrum_figure()
        ax.step(lo_e[sel], np.maximum(N_mc[sel], CUM_YMIN), where="post", color=MC_COLOR, lw=1.8, label="MC, 9 processes")
        ax.fill_between(lo_e[sel], np.maximum(N_mc[sel] - E_mc[sel], CUM_YMIN), np.maximum(N_mc[sel] + E_mc[sel], CUM_YMIN),
                        step="post", color=MC_COLOR, alpha=0.3, lw=0, label="MC stat. error")
        ok = sel & (N_mc > 0)
        axr.fill_between(lo_e[ok], 1.0 - E_mc[ok] / N_mc[ok], 1.0 + E_mc[ok] / N_mc[ok], step="post", color=MC_COLOR, alpha=0.3, lw=0)
        N_of = {}
        for key, cur in curves.items():
            _, _, lab, col, ls = MODELS[key]
            N_of[key] = rcum(cur)
            ax.plot(lo_e[sel], np.maximum(N_of[key][sel], CUM_YMIN), color=col, lw=1.8, ls=ls, label=lab)
            axr.plot(lo_e[ok], N_of[key][ok] / N_mc[ok], color=col, lw=1.8, ls=ls)
        P.finish(ax, axr, channel=ch, xmin=FLOOR, xmax=P.XMAX_FULL, top=float(N_mc[sel].max()), ymin=CUM_YMIN, ylabel=YLABEL,
                 extra=C.DY_SHORT[VARIANT], ratio_label="Curve / MC", ratio_lim=(0.2, 20.0))
        axr.set_xlabel(XLABEL)
        import matplotlib.ticker as mticker
        axr.set_yscale("log")
        axr.set_ylim(0.2, 20.0)
        axr.yaxis.set_major_locator(mticker.FixedLocator([0.2, 0.5, 1, 2, 5, 10, 20]))
        axr.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"{val:g}"))
        axr.yaxis.set_minor_formatter(mticker.NullFormatter())
        mc_first(ax, P)
        P.save(fig, out / "cumulative")
        # --- events per slice ---------------------------------------------------------------------------------
        edges = np.array(SLICES)
        ssum = lambda x: np.array([float(np.asarray(x)[(c10 >= a) & (c10 < b)].sum()) for a, b in zip(edges[:-1], edges[1:])])  # noqa: E731
        mc, err = ssum(v10), np.sqrt(ssum(var10))
        cs_, w = 0.5 * (edges[:-1] + edges[1:]), edges[1:] - edges[:-1]
        fig, (ax, axr) = P.spectrum_figure()
        ax.errorbar(cs_, mc, xerr=w / 2, yerr=err, fmt="o", ms=5, color=MC_COLOR, lw=1.4, label="MC, 9 processes")
        top, sums = 0.0, {}
        for key, cur in curves.items():
            _, _, lab, col, ls = MODELS[key]
            sums[key] = ssum(cur)
            ax.step(edges, np.append(sums[key], sums[key][-1]), where="post", color=col, lw=1.8, ls=ls, label=lab)
            d = sums[key] - mc
            axr.stairs(d, edges, color=col, lw=1.8, ls=ls)
            top = max(top, float(np.abs(d).max()))
        axr.fill_between(edges, np.append(-err, -err[-1]), np.append(err, err[-1]), step="post", color=MC_COLOR, alpha=0.3, lw=0)
        ax.fill_between([], [], [], color=MC_COLOR, alpha=0.3, lw=0, label="MC stat. error (lower panel)")
        lim = 1.3 * max(top, float(err.max()))
        P.finish(ax, axr, channel=ch, xmin=FLOOR, xmax=XMAX, top=float(mc.max()), ymin=1e-2, ylabel="Events in slice",
                 extra=C.DY_SHORT[VARIANT], ratio_label=r"Curve $-$ MC [events]", ratio_lim=(-lim, lim), ratio_ref=0.0)
        axr.set_ylabel(r"Curve $-$ MC [events]", fontsize=20)
        mc_first(ax, P)
        P.save(fig, out / "slices")
        # --- events per 100 GeV slice out to 12 TeV; below, the running sum of curve - MC from the threshold up: the
        #     layout of 18_cumulative's slices_cumulative_w100 (the RooKeysPdf frame) with every model
        fe = np.arange(C.THRESHOLD, 12000.0 + 50.0, 100.0)
        fsum = lambda x: np.array([float(np.asarray(x)[int(np.searchsorted(e10, a_)):int(np.searchsorted(e10, b_))].sum())   # noqa: E731
                                   for a_, b_ in zip(fe[:-1], fe[1:])])
        fmc, ferr = fsum(v10), np.sqrt(np.maximum(fsum(var10), 0.0))
        fc, fw = 0.5 * (fe[:-1] + fe[1:]), fe[1:] - fe[:-1]
        fig, (ax, axr) = P.spectrum_figure()
        ax.errorbar(fc, fmc, xerr=fw / 2, yerr=ferr, fmt="o", ms=5, color=MC_COLOR, lw=1.4, label="MC, 9 processes")
        top, runs = 0.0, {}
        for key, cur in curves.items():
            _, _, lab, col, ls = MODELS[key]
            cs = fsum(cur)
            ax.step(fe, np.append(cs, cs[-1]), where="post", color=col, lw=1.8, ls=ls, label=lab)
            runs[key] = np.cumsum(cs - fmc)
            axr.step(fe, np.append(runs[key], runs[key][-1]), where="post", color=col, lw=2.2, ls=ls)
            top = max(top, float(np.abs(runs[key]).max()))
        lim = 1.3 * top
        P.finish(ax, axr, channel=ch, xmax=float(fe[-1]), top=float(fmc.max()), ymin=1e-3, ylabel="Events in slice",
                 extra=C.DY_SHORT[VARIANT], ratio_label=r"$\sum$ (curve $-$ MC)", ratio_lim=(-lim, lim), ratio_ref=0.0)
        axr.set_ylabel(r"$\sum$ (curve $-$ MC)", fontsize=20)
        mc_first(ax, P)
        P.save(fig, out / "slices_cumulative")
        # --- the numbers ----------------------------------------------------------------------------------------
        heads = [MODELS[k][2] for k in curves]
        run_rows = [(MODELS[k][2], float(runs[k][int(np.argmax(np.abs(runs[k])))]), float(fe[1:][int(np.argmax(np.abs(runs[k])))]),
                     float(runs[k][-1])) for k in curves]
        L += [f"## {ch}", "", "Events per slice:", "", "| slice [GeV] | MC | " + " | ".join(heads) + " |", "|---|---|" + "---|" * len(heads)]
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            L.append(f"| {a:.0f}-{b:.0f} | {mc[i]:.3g} +- {err[i]:.2g} | "
                     + " | ".join(f"{sums[k][i]:.3g} ({(sums[k][i] - mc[i]) / err[i]:+.1f})" for k in curves) + " |")
        L += ["", "Running sum of curve - MC from the threshold, 100 GeV slices:", "",
              "| curve | largest excursion [events] | reached at [GeV] | at 12 TeV [events] |", "|---|---|---|---|"]
        L += [f"| {lab} | {ext:+.1f} | {at:.0f} | {end:+.1f} |" for lab, ext, at, end in run_rows]
        L += ["", "Events above m_min:", "", "| m_min [GeV] | MC | " + " | ".join(heads) + " |", "|---|---|" + "---|" * len(heads)]
        for m in [FLOOR] + REF:
            i = int(np.searchsorted(lo_e, m))
            L.append(f"| {m:.0f} | {N_mc[i]:.3g} +- {E_mc[i]:.2g} | "
                     + " | ".join(f"{N_of[k][i]:.3g} ({N_of[k][i] / N_mc[i]:.2f} x, {(N_of[k][i] - N_mc[i]) / E_mc[i]:+.1f})" for k in curves) + " |")
        # the DY process around its switch from the RooKeysPdf to the exponential tail (the step in the summed curve)
        tp = C.RESULTS / ch / PER_PROCESS_TAIL[0] / VARIANT / PER_PROCESS_TAIL[1]
        if tp.with_suffix(".npz").exists():
            _, tparts, tdiags = load_truth(tp)
            m_T = tdiags.get("DYJets", {}).get("m_T")
            if m_T and "DYJets" in tparts:
                _, dv, dvar = comps10["DYJets"]
                L += ["", "DY at its switch to the exponential tail:", "",      # the per-process tails (the earlier model 3)
                      "| switch mass [GeV] | range [GeV] | DY MC | DY curve | curve / MC |", "|---|---|---|---|---|"]
                for a_, b_ in ((5000.0, m_T), (m_T, m_T + 1000.0)):
                    m = (c10 >= a_) & (c10 < b_)
                    mc_, er_, cu_ = float(dv[m].sum()), float(np.sqrt(dvar[m].sum())), float(np.asarray(tparts["DYJets"])[m].sum())
                    L.append(f"| {m_T:.0f} | {a_:.0f}-{b_:.0f} | {mc_:.3f} +- {er_:.3f} | {cu_:.3f} | {cu_ / mc_:.2f} |")
        L.append("")
        logger.info("%s: wrote %s", ch, out)
    (C.RESULTS / "MODELS.md").write_text("\n".join(L) + "\n")
    logger.info("wrote %s", C.RESULTS / "MODELS.md")


SUMTAIL_DIR, SUMTAIL_NAME = "3_truths_sumtail", "truth_kde_sumexpo"


def run_sumtail(args):
    """The sum of the nine RooKeysPdfs with ONE exponential tail: the RooKeysPdf sum below the switch
    mass, an exponential fitted to the summed MC from 1 TeV below the switch to the last MC event above it, blended
    over the 200 GeV below the switch. Saved as a toy background (3_truths_sumtail/lo_ht/truth_kde_sumexpo), drawn in
    the truth-figure style with its one switch line, and scored against the per-process tails (SUMTAIL.md)."""
    import importlib
    from util import sl
    from spectrum import centers, load_components
    from truth_sets import check_component, load_truth, save_truth
    T3 = importlib.import_module("3_truths")
    edges = np.arange(3000.0, 8001.0, 500.0)
    L = [f"# The RooKeysPdf sum with one exponential tail, {C.ERA} {C.REGION_NAME} ({C.LUMI:.1f} fb-1)", "",
         "Score = the sum of squared pulls (curve - MC) / sigma_MC over the ten 500 GeV slices from 3 to 8 TeV, the largest "
         "|pull| in brackets; then curve / MC above 3 / 4 / 5 / 6 TeV.", ""]
    for ch in args.channels:
        comps = load_components(ch, VARIANT, "split9")
        e, v, var = comps["sum"]
        c = centers(e)
        _, kparts, _ = load_truth(C.RESULTS / ch / MODELS["kde9"][0] / VARIANT / MODELS["kde9"][1])
        kde = np.asarray(kparts["sum"], float)
        per = np.asarray(load_truth(C.RESULTS / ch / PER_PROCESS_TAIL[0] / VARIANT / PER_PROCESS_TAIL[1])[1]["sum"], float)
        e50, v50, var50 = sl.rebin(e, v, var, 5)
        m_last = sl.last_populated(e, v)
        sl_ = lambda x: np.array([float(np.asarray(x)[(c >= a) & (c < b)].sum()) for a, b in zip(edges[:-1], edges[1:])])   # noqa: E731
        mc, err = sl_(v), np.sqrt(sl_(var))

        def build(m_T):
            fit = sl.fit_component("expo", e50, v50, var50, max(m_T - 1000.0, C.THRESHOLD), m_last, m_T, "poisson")
            tail = sl.eval_fit(fit, c, 50.0, sl.NATIVE_BW)
            t = np.clip((c - (m_T - 200.0)) / 200.0, 0.0, 1.0)
            return (1.0 - t) * kde + t * tail, fit

        def line(label, arr):
            pulls = (sl_(arr) - mc) / err
            return (f"| {label} | {float((pulls ** 2).sum()):.0f} ({float(np.abs(pulls).max()):.1f}) | "
                    + " / ".join(f"{arr[c >= thr].sum() / v[c >= thr].sum():.2f}" for thr in (3000.0, 4000.0, 5000.0, 6000.0)) + " |")

        rule = sl.trust_mass(e50, v50, var50, C.TRUST_NEFF, C.TRUST_WIDTH, step=50.0)
        L += [f"## {ch}", "", f"The N_eff rule applied to the summed MC would switch at {rule:.0f} GeV.", "",
              "| background | score | curve / MC above 3 / 4 / 5 / 6 TeV |", "|---|---|---|",
              line("RooKeysPdf alone", kde), line("per-process tails", per)]
        for m_T in np.arange(3000.0, 5751.0, 250.0):
            L.append(line(f"one tail, switch {m_T:.0f}", build(m_T)[0]))
        out, fit = build(args.switch)
        L += ["", f"Built: switch {args.switch:.0f} GeV, exponential slope {fit['params'][1]:+.2f} per TeV.", ""]
        parts = {k: np.asarray(kparts[k], float) for k in kparts if k != "sum"}
        parts["sum"] = np.maximum(out, 0.0)
        diags = {k: {} for k in parts if k != "sum"}
        diags["sum"] = {"m_T": float(args.switch), "tail_family": "expo", "tail_params": [float(x) for x in fit["params"]],
                        "fit_range": [max(args.switch - 1000.0, C.THRESHOLD), float(m_last)], **check_component(e, v, var, parts["sum"], None)}
        save_truth(C.RESULTS / ch / SUMTAIL_DIR / VARIANT / SUMTAIL_NAME, e, parts, diags)
        (C.RESULTS / ch / SUMTAIL_DIR / "plots").mkdir(parents=True, exist_ok=True)
        # in the figure the per-process RooKeysPdf curves stop at the switch: above it the model is the one exponential
        shown = {k: (np.where(c < args.switch, arr, np.nan) if k != "sum" else arr) for k, arr in parts.items()}
        T3.plot_truth_sum(ch, VARIANT, "kde_sumexpo", comps, shown, diags, C.RESULTS / ch / SUMTAIL_DIR / "plots" / "truth_lo_ht_kde_sumexpo",
                          extra_line="")
        logger.info("%s: one tail from %.0f GeV, slope %+.2f per TeV", ch, args.switch, fit["params"][1])
    (C.RESULTS / "SUMTAIL.md").write_text("\n".join(L) + "\n")
    logger.info("wrote %s", C.RESULTS / "SUMTAIL.md")


def load_series(ch, folder, truth, cand):
    p = C.RESULTS / ch / "6_windows" / folder / VARIANT / f"toys_{truth}.csv"
    if not p.exists():
        return None
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            if r["candidate"] == cand and r.get("status") == "ok" and float(r["inject"]) == 0.0 \
                    and r.get("pull_gauss_mean") not in ("", "None", None):
                rows.append((float(r["mass"]), float(r["pull_gauss_mean"]), float(r["pull_gauss_sigma"]),
                             float(r["n_passed"]) / max(float(r["n_conv"]), 1.0), int(float(r["n_toys"]))))
    rows.sort()
    return np.array(rows, float) if rows else None


def run_bias(args):
    import plots as P
    from matplotlib.lines import Line2D
    L = [f"# The +-5 sigma window from {FLOOR:.0f} GeV on the four background models ({C.ERA})", "",
         "Gaussian mu and sigma of the zero-injection pulls; a cell counts while >= 90 % of its fits pass the checks.", ""]
    for ch in args.channels:
        out = C.RESULTS / ch / "22_models"
        out.mkdir(parents=True, exist_ok=True)
        for cand in args.candidates:
            data = {}
            for key, suffix, truth, *_r in BIAS_MODELS:
                for tag in ([args.tag + suffix] if suffix else [args.tag] + list(args.fallback_tags)):
                    s = load_series(ch, f"k5_lo{FLOOR:.0f}{tag}", truth, cand)
                    if s is not None:
                        s = s[s[:, 0] <= args.max_mass]              # the range every model was run on
                        data[key] = (s, tag)
                        break
            L += [f"## {ch}, {cand}", "", "| background | toys per cell | last mass passing 90 % | max abs(mu) up to it (mass) | "
                  "masses beyond 0.5 | beyond 0.3 | sigma min-max |", "|---|---|---|---|---|---|---|"]
            for stat, col_i, ylabel, ylim in (("pull_mean", 1, r"Gaussian $\mu$ of the pulls", (-2.0, 3.0)),
                                              ("width", 2, r"Gaussian $\sigma$ of the pulls", (0.0, 2.5))):
                fig, ax = P.scan_figure()
                handles = []
                for key, _suffix, _truth, lab, col, ls, mk in BIAS_MODELS:
                    if key not in data:
                        continue
                    s, tag = data[key]
                    x, y, good = s[:, 0], s[:, col_i], s[:, 3] >= PASS
                    ax.plot(x, y, lw=1.5, color=col, ls=ls)
                    ax.plot(x[good], y[good], mk, ms=6, color=col)
                    ax.plot(x[~good], y[~good], mk, ms=6, mfc="white", mec=col)
                    handles.append(Line2D([], [], color=col, lw=1.5, ls=ls, marker=mk, ms=6, label=lab))
                    if stat == "pull_mean":
                        above = x > FLOOR                                  # the floor cell has half a window
                        gi = np.nonzero(good & above)[0]
                        last = x[gi[-1]] if len(gi) else None
                        fl = good & above & (x <= (last or 0))
                        i = int(np.argmax(np.where(fl, np.abs(y), -1.0))) if fl.any() else None
                        L.append(f"| {lab} | {int(s[0, 4])} | {'-' if last is None else f'{last:.0f}'} | "
                                 f"{'-' if i is None else f'{y[i]:+.2f} ({x[i]:.0f})'} | "
                                 f"{', '.join(f'{m:.0f}' for m, v, g in zip(x, y, fl) if g and abs(v) > C.BIAS_PASS) or '-'} | "
                                 f"{', '.join(f'{m:.0f}' for m, v, g in zip(x, y, fl) if g and abs(v) > C.BIAS_TARGET) or '-'} | "
                                 f"{np.nanmin(s[fl, 2]):.2f}-{np.nanmax(s[fl, 2]):.2f} |" if fl.any() else f"| {lab} | - | - | - | - | - | - |")
                if stat == "pull_mean":
                    P.bias_lines(ax)
                else:
                    ax.axhline(1.0, color="k", lw=0.8)
                handles.append(Line2D([], [], color="k", lw=0, marker="o", mfc="white", ms=6, label="open: < 90 % pass"))
                P.finish_scan(ax, channel=ch, extra=f"{CAND_LABEL.get(cand, cand)}\n$\\pm 5\\sigma$ window\nfrom {FLOOR:.0f} GeV",
                              ylabel=ylabel, ylim=ylim, handles=handles, ncol=1)
                P.save(fig, out / f"bias4_{cand}_{stat}{args.tag}")
            L.append("")
        logger.info("%s: wrote %s", ch, out)
    (C.RESULTS / f"MODELS_BIAS{args.tag}.md").write_text("\n".join(L) + "\n")
    logger.info("wrote %s", C.RESULTS / f"MODELS_BIAS{args.tag}.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["models", "bias", "sumtail"])
    ap.add_argument("--switch", type=float, default=4000.0, help="sumtail: the mass at which the one exponential tail takes over")
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--candidates", nargs="+", default=["powexp", "expo"])
    ap.add_argument("--tag", default="_100k", help="suffix of the 6_windows folders")
    ap.add_argument("--fallback-tags", nargs="*", default=[], help="other folder suffixes to take a model's toys from when the tag has none")
    ap.add_argument("--max-mass", type=float, default=4800.0, help="bias: the highest mass drawn (the fitted-curve grids stop at 4800 GeV)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    from wrplotter.cli_utils import setup_logging
    setup_logging(args.verbose)
    {"models": run_models, "bias": run_bias, "sumtail": run_sumtail}[args.step](args)


if __name__ == "__main__":
    main()
