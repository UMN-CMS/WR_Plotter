#!/usr/bin/env python3
"""Step 5: toys at data luminosity from every truth, full-range S+B fit with every candidate.

For each channel, DY variant, truth (results/<channel>/3_truths/<variant>/truth_<name>.npz)
and signal mass m_WR:
  - background-only Poisson fit of each candidate to the truth (the seed),
  - the Asimov S+B fit (N_sig_asimov, sigma_fit; the expected 95 % CL upper limit is taken as
    1.96 sigma_fit and sets the injection scale),
  - NTOYS Poisson toys per injection (0, 1x, 2x the expected limit): the S+B fit of each
    candidate on each toy; N_sig and its error are kept.
Summary per (truth, candidate, mass, injection): median and mean N_sig, RMS, the Gaussian fitted
to the pulls (N_sig - injected) / sigma_fit of the toys whose fit passed the checks (its mean is
the CMS bias, its sigma the pull width; lib/pulls.py), the fraction passed; the spurious
signal S_spur = median N_sig / RMS N_sig at zero injection (the ATLAS convention) and the
bias (mean N_sig - injected) / RMS at every injection. Pass lines from config: 0.5 (CMS EXO),
0.3 target.

Outputs results/<channel>/4_toys/<variant>/: toys_<truth>.csv (summary rows), nsig_<truth>.npz
(the per-toy N_sig arrays, key <candidate>_<mass>_<inj>), summary.md (all truths), plots/:
  bias_vs_mass_<truth>       S_spur per candidate vs m_WR with the 0.3 / 0.5 bands
  pull_mean_vs_mass_<truth>  the bias as CMS quotes it: the mean of the Gaussian fitted to the pulls vs m_WR
  width_vs_mass_<truth>      pull width (sigma of the Gaussian fit) per candidate vs m_WR
  linearity_<truth>          (mean N_sig - injected) / RMS vs m_WR at 1x and 2x
  pulls_<truth>_m<mass>      pull histograms with their Gaussian fits at zero injection, three masses
  --restat recomputes the pull statistics of every stored row from the per-toy arrays (2026-10-26:
  the Gaussian fit replaced the 1.48 MAD) and rebuilds the CSVs, summary and plots.

  cd WR_Plotter && python background_fit/4_toys.py
"""
from __future__ import annotations

import argparse
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

logger = logging.getLogger("4_toys")


# ---------------------------------------------------------------------------------------------
# worker: one (truth, mass) cell, every candidate, Asimov + toys
# ---------------------------------------------------------------------------------------------

def run_mass(job):
    """job = dict(channel, edges, truth100 (np array), variances100, mass, candidates, ntoys,
    inject, seed). Returns (rows, {key: nsig array})."""
    import published_functions as pf                                     # noqa: F401
    import ROOT
    from sbfit import FIT_HI, SBModel, bkg_only, pivot, signal_pars, signal_shape
    from spectrum import centers
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    edges, truth, var = job["edges"], job["truth"], job["variances"]
    c = centers(edges)
    m = job["mass"]
    lo, hi = job.get("lo", C.FIT_LO), job.get("hi", FIT_HI)      # a window when 6_windows.py sends one
    mu, sigma = signal_pars(job["channel"], m)
    shape = signal_shape(edges, mu, sigma)
    core = (c >= mu - C.SIG_CORE_K * sigma) & (c <= mu + C.SIG_CORE_K * sigma)
    b_core = float(truth[core].sum())
    from sbfit import nsig_limits
    nsig_bounds = nsig_limits(b_core)                 # None: N_sig is free (config.NSIG_LIMITED for the old guard)
    rng = ROOT.TRandom3(job["seed"])
    rows, arrays = [], {}
    parents = {}
    for name in sorted(job["candidates"], key=lambda n: pf.FUNCS[n][0]):
        fb = bkg_only(name, edges, truth, var, lo, hi, parents, m0=pivot(name, mu))
        parents[name] = fb
        base = {"channel": job["channel"], "variant": job["variant"], "truth": job["truth_name"],
                "candidate": name, "mass": m, "mu": mu, "sigma": sigma, "b_core": b_core, "lo": lo, "hi": hi}
        if fb is None:
            rows.append({**base, "inject": 0.0, "status": "no background fit"})
            continue
        model = SBModel(name, mu, sigma, lo, hi)
        asim = model.fit(edges, truth, fb["params"], 0.0, nsig_bounds)
        if asim is None or asim["nsig_err"] <= 0:
            rows.append({**base, "inject": 0.0, "status": "no Asimov fit"})
            continue
        ul = 1.96 * asim["nsig_err"]
        for k in job["inject"]:
            n_inj = k * ul
            expect = truth + n_inj * shape
            nsigs, errs, passed = [], [], []
            for _ in range(job["ntoys"]):
                toy = np.array([rng.Poisson(max(x, 0.0)) for x in expect], float)
                r = model.fit(edges, toy, asim["bkg"], n_inj, nsig_bounds)
                if r is None:
                    continue
                nsigs.append(r["nsig"])
                errs.append(r["nsig_err"])
                passed.append(r["passed"] and r["nsig_err"] > 0)
            a_n, a_e, a_p = np.array(nsigs), np.array(errs), np.array(passed, bool)
            row = {**base, "inject": k, "n_inj": n_inj, "ul_expected": ul,
                   "asimov_nsig": asim["nsig"], "asimov_err": asim["nsig_err"],
                   "asimov_pull": asim["nsig"] / asim["nsig_err"], "asimov_passed": asim["passed"],
                   "n_toys": job["ntoys"], "n_conv": int(len(a_n)), "n_passed": int(a_p.sum()),
                   "status": "ok"}
            if len(a_n):
                med = float(np.median(a_n))
                rms = float(a_n.std())
                row.update({"nsig_median": med, "nsig_mean": float(a_n.mean()), "nsig_rms": rms,
                            "bias_median": (med - n_inj) / rms if rms > 0 else float("nan"),
                            "bias_mean": (float(a_n.mean()) - n_inj) / rms if rms > 0 else float("nan"),
                            "sspur_asimov_units": (med - n_inj) / asim["nsig_err"]})
                if a_p.sum() >= 10:
                    from pulls import pull_stats
                    row.update(pull_stats((a_n[a_p] - n_inj) / a_e[a_p]))
            rows.append(row)
            arrays[f"{name}_{int(m)}_{k:g}"] = a_n
            arrays[f"{name}_{int(m)}_{k:g}_err"] = a_e
            arrays[f"{name}_{int(m)}_{k:g}_ok"] = a_p
    return rows, arrays


# ---------------------------------------------------------------------------------------------

PASS_FRAC = 0.9
TRUTH_BKG = {"raw": "Raw MC Bkg", "kde": "RooKeysPdf Bkg", "kde9": "RooKeysPdf Bkg"}
TRUTH_STYLE = {"raw": ("#717581", "-", "o"), "kde9": ("#7a21dd", "--", "s"), "kde": ("#7a21dd", "--", "s")}


def pass_fractions(rows, cand):
    """[(mass, passed / converged)] of the candidate's zero-injection cells, in mass order."""
    return sorted((r["mass"], r["n_passed"] / max(r["n_conv"], 1)) for r in rows
                  if r["candidate"] == cand and r["inject"] == 0.0 and r.get("status") == "ok")


def first_sparse(rows, cand, frac=PASS_FRAC):
    """Where the candidate's sparse regime begins: the first mass above the last one at which at
    least `frac` of its zero-injection fits pass the checks (None when the top mass still passes).
    A low mass failing on its own, e.g. a window clipped at the fit floor, does not set it."""
    pts = pass_fractions(rows, cand)
    good = [m_ for m_, f_ in pts if f_ >= frac]
    if not good:
        return pts[0][0] if pts else None
    above = [m_ for m_, _ in pts if m_ > good[-1]]
    return above[0] if above else None


def _series(rows, cand, inj, key):
    """(masses, values) of one candidate and injection, in mass order (appended masses sit at the
    end of the files, so the rows are not sorted)."""
    pts = sorted((r["mass"], r.get(key)) for r in rows if r["candidate"] == cand and r["inject"] == inj
                 and r.get("status") == "ok" and r.get(key) is not None)
    return [p[0] for p in pts], [p[1] for p in pts]


CAND_COLOR = {"expo": "#5790fc", "powexp": "#f89c20"}         # the overlays: colour = candidate (the explainer's fit colours),
TRUTH_LM = {"raw": ("-", "o"), "kde9": ("--", "s"), "kde": ("--", "s")}   # line style and marker = truth


def plot_truth_compare(channel, variant, cand, rows_by_truth, out, tag=""):
    """One candidate (or a list of them) on several truths in one figure: the Gaussian mu of the
    pulls and its sigma against m_WR. One candidate: each truth with its own 90 % line (the raw MC
    against the RooKeysPdf). Several candidates: colour = candidate, line style
    and marker = truth, filled markers where >= 90 % of that cell's fits pass and open ones where
    not, no lines. Files <stat>_vs_mass_<cand>[_<cand2>]_<truthA>_vs_<truthB>."""
    import matplotlib.pyplot as plt
    import mplhep as hep
    import numpy as np
    import plots as P
    from matplotlib.lines import Line2D
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    hep.style.use("CMS")
    cands = [cand] if isinstance(cand, str) else list(cand)
    multi = len(cands) > 1
    extra = (f"{C.DY_SHORT[variant]}{tag}\nspurious signal" if multi
             else f"{C.DY_SHORT[variant]}\n{C.MENU_LABEL.get(cands[0], cands[0])}{tag}\nspurious signal")
    name = "_".join(cands) + "_" + "_vs_".join(rows_by_truth)
    for stat, key, ylabel, ylim in [("pull_mean", "pull_gauss_mean", r"Gaussian $\mu$ of the pulls", (-2.0, 4.0)),
                                    ("width", "pull_width", r"Gaussian $\sigma$ of the pulls", (0.0, 3.5))]:
        fig, ax = plt.subplots(figsize=(10, 8))
        for truth, rows in rows_by_truth.items():
            col0, ls0, mk0 = TRUTH_STYLE.get(truth, ("k", "-", "o"))
            for cnd in cands:
                if multi:
                    col = CAND_COLOR.get(cnd, "k")
                    ls, mk = TRUTH_LM.get(truth, ("-", "o"))
                    lab = f"{cnd}, {TRUTH_BKG.get(truth, truth)}"
                else:
                    col, ls, mk, lab = col0, ls0, mk0, TRUTH_BKG.get(truth, truth)
                x, y = _series(rows, cnd, 0.0, key)
                if multi:
                    x, y = np.array(x), np.array(y, float)
                    frac = dict(pass_fractions(rows, cnd))
                    good = np.array([frac.get(m_, 0.0) >= PASS_FRAC for m_ in x])
                    ax.plot(x, y, lw=1.5, color=col, ls=ls, label=lab)
                    ax.plot(x[good], y[good], mk, ms=6, color=col)
                    ax.plot(x[~good], y[~good], mk, ms=6, mfc="white", mec=col)
                else:
                    ax.plot(x, y, marker=mk, ms=5, lw=1.5, color=col, ls=ls, label=lab)
                    thr = first_sparse(rows, cnd)
                    if thr:
                        ax.axvline(thr, color=col, ls=":", lw=1.2, label=rf"$< 90\,\%$ pass from {thr / 1000:g} TeV")
        if stat == "pull_mean":
            for yv, c_ in [(C.BIAS_TARGET, "0.5"), (C.BIAS_PASS, "0.2")]:
                ax.axhline(yv, color=c_, lw=0.9, ls="--")
                ax.axhline(-yv, color=c_, lw=0.9, ls="--")
            ax.axhline(0, color="k", lw=0.8)
        else:
            ax.axhline(1.0, color="k", lw=0.8)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$m_{W_R}$ [GeV]")
        ax.set_ylabel(ylabel)
        cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, extra), year=C.YEAR)
        if multi:
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Line2D([], [], marker="o", mfc="white", mec="k", color="none", ms=6, label=r"open: $< 90\,\%$ of fits pass"))
            ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE_LEGEND)
        else:
            ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
        fit_header(ax, ignore_above=(stat == "width"))
        P.save(fig, out / f"{stat}_vs_mass_{name}")


def plot_truth_summary(channel, variant, truth_name, rows, arrays, out, tag="", only=None):
    """The summary figures of one truth; `only` = a subset of the candidates, written beside the
    full set with the candidates' names as a suffix (e.g. _expo for the talk slides)."""
    import matplotlib.pyplot as plt
    import mplhep as hep
    import published_functions as pf
    import plots as P
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    hep.style.use("CMS")
    cands = []
    for r in rows:
        if r["candidate"] not in cands and (not only or r["candidate"] in only):
            cands.append(r["candidate"])
    suffix = "" if not only else "_" + "_".join(only)
    # the label block names the background model as the explainer figures do
    extra = f"{C.DY_SHORT[variant]}\n{TRUTH_BKG.get(truth_name, 'truth: ' + truth_name)}{tag}\nspurious signal"

    def series(cand, inj, key):
        return _series(rows, cand, inj, key)

    # the dotted line of a single-candidate figure: the first mass where fewer than 90 % of that
    # candidate's fits pass the checks, measured on these very fits; a figure
    # with several candidates draws no line
    sparse_at = first_sparse(rows, cands[0]) if len(cands) == 1 else None

    def regime_line(ax):
        if sparse_at:
            ax.axvline(sparse_at, color="0.4", ls=":", lw=1.2, label=rf"$< 90\,\%$ of fits pass from {sparse_at / 1000:g} TeV")

    # 1. spurious signal vs mass
    fig, ax = plt.subplots(figsize=(10, 8))
    for cand in cands:
        x, y = series(cand, 0.0, "bias_median")
        ax.plot(x, y, marker="o", ms=5, lw=1.5, color=pf.COLOR[cand], ls=pf.LS[cand], label=C.MENU_LABEL.get(cand, cand))
    for yv, col in [(C.BIAS_TARGET, "0.5"), (C.BIAS_PASS, "0.2")]:
        ax.axhline(yv, color=col, lw=0.9, ls="--")
        ax.axhline(-yv, color=col, lw=0.9, ls="--")
    ax.axhline(0, color="k", lw=0.8)
    regime_line(ax)
    ax.set_ylim(-2.0, 4.0)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"median $N_{\rm sig}$ / RMS $N_{\rm sig}$")
    cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, extra), year=C.YEAR)
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    fit_header(ax)
    P.save(fig, out / f"bias_vs_mass_{truth_name}{suffix}")
    # 1b. the bias as CMS quotes it: the mean of the Gaussian fitted to the pulls vs mass
    fig, ax = plt.subplots(figsize=(10, 8))
    for cand in cands:
        x, y = series(cand, 0.0, "pull_gauss_mean")
        ax.plot(x, y, marker="o", ms=5, lw=1.5, color=pf.COLOR[cand], ls=pf.LS[cand], label=C.MENU_LABEL.get(cand, cand))
    for yv, col in [(C.BIAS_TARGET, "0.5"), (C.BIAS_PASS, "0.2")]:
        ax.axhline(yv, color=col, lw=0.9, ls="--")
        ax.axhline(-yv, color=col, lw=0.9, ls="--")
    ax.axhline(0, color="k", lw=0.8)
    regime_line(ax)
    ax.set_ylim(-2.0, 4.0)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"Gaussian $\mu$ of the pulls")
    cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, extra), year=C.YEAR)
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    fit_header(ax)
    P.save(fig, out / f"pull_mean_vs_mass_{truth_name}{suffix}")
    # 2. pull width vs mass
    fig, ax = plt.subplots(figsize=(10, 8))
    for cand in cands:
        x, y = series(cand, 0.0, "pull_width")
        ax.plot(x, y, marker="o", ms=5, lw=1.5, color=pf.COLOR[cand], ls=pf.LS[cand], label=C.MENU_LABEL.get(cand, cand))
    ax.axhline(1.0, color="k", lw=0.8)
    regime_line(ax)
    ax.set_ylim(0.0, 3.5)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"Gaussian $\sigma$ of the pulls")
    cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, extra), year=C.YEAR)
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    fit_header(ax, ignore_above=True)          # a degenerate fit's width runs off the frame on purpose
    P.save(fig, out / f"width_vs_mass_{truth_name}{suffix}")
    # 3. linearity: (mean N_sig - injected) / RMS at 1x and 2x
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(10, 8))
    for cand in cands:
        for inj, mk in [(1.0, "o"), (2.0, "s")]:
            x, y = series(cand, inj, "bias_mean")
            ax.plot(x, y, marker=mk, ms=5, lw=1.2, color=pf.COLOR[cand], ls=pf.LS[cand])
    handles = [Line2D([], [], color=pf.COLOR[c], ls=pf.LS[c], lw=1.5, label=C.MENU_LABEL.get(c, c)) for c in cands]
    handles += [Line2D([], [], color="k", ls="none", marker=mk, ms=6, label=f"{inj:g}x limit injected")
                for inj, mk in [(1.0, "o"), (2.0, "s")]]
    for yv, col in [(C.BIAS_TARGET, "0.5"), (C.BIAS_PASS, "0.2")]:
        ax.axhline(yv, color=col, lw=0.9, ls="--")
        ax.axhline(-yv, color=col, lw=0.9, ls="--")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylim(-2.0, 4.0)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"(mean $N_{\rm sig}$ $-$ injected) / RMS $N_{\rm sig}$")
    cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, extra), year=C.YEAR)
    ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE_LEGEND)
    fit_header(ax)
    P.save(fig, out / f"linearity_{truth_name}{suffix}")
    # 4. pull histograms at three masses, zero injection
    for m in [1600, 2600, 3400]:
        fig, ax = plt.subplots(figsize=(10, 8))
        any_ = False
        for cand in cands:
            key = f"{cand}_{m}_0"
            if key not in arrays:
                continue
            n, e, ok = arrays[key], arrays[key + "_err"], arrays[key + "_ok"]
            if ok.sum() < 10:
                continue
            pulls = n[ok] / e[ok]
            any_ = True
            from pulls import gauss_curve, gauss_fit
            g = gauss_fit(pulls, nbins=40)
            ax.hist(pulls, bins=40, range=(-5, 5), histtype="step", lw=1.8, color=pf.COLOR[cand], ls=pf.LS[cand],
                    label=f"{cand} ({g['mean']:+.2f}, {g['sigma']:.2f})")
            if g["ok"]:                                     # the fitted Gaussian over its histogram
                xg = np.linspace(-5, 5, 401)
                ax.plot(xg, gauss_curve(g, xg), color=pf.COLOR[cand], lw=1.2, ls="-", alpha=0.8)
        if not any_:
            plt.close(fig)
            continue
        ax.set_xlabel(r"$N_{\rm sig}$ / $\sigma(N_{\rm sig})$, no signal injected")
        ax.set_ylabel("toys")
        cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, extra + f"\n$m_{{W_R}}$ = {m} GeV"), year=C.YEAR)
        ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND, title=r"function (Gaussian $\mu$, $\sigma$)",
                  title_fontsize=FONT_SIZE_LEGEND)
        fit_header(ax)
        P.save(fig, out / f"pulls_{truth_name}_m{m}{suffix}")


def _type_row(r):
    """A CSV row back to numbers (None for a field the cell never wrote, e.g. the pull width when
    fewer than 10 toys passed the checks)."""
    rr = dict(r)
    for k, v in rr.items():
        if k in ("channel", "variant", "truth", "candidate", "status"):
            continue
        if v is None or v == "":
            rr[k] = None
            continue
        try:
            rr[k] = float(v)
        except (TypeError, ValueError):
            pass
    for k in ("n_toys", "n_conv", "n_passed"):
        if k in rr and isinstance(rr[k], float):
            rr[k] = int(rr[k])
    return rr


PULL_KEYS = ("pull_mean", "pull_rms", "pull_median", "pull_width", "pull_gauss_mean", "pull_gauss_mean_err",
             "pull_gauss_sigma", "pull_gauss_sigma_err")


def restat(out):
    """Recompute the pull statistics of every stored row from the per-toy arrays of its npz and
    rewrite the per-truth CSVs (the Gaussian fit of lib/pulls.py; rows whose arrays are missing
    keep what they have)."""
    from pulls import pull_stats
    from util import read_csv, write_csv
    for p in sorted(Path(out).glob("*/toys_*.csv")):
        truth_name = p.stem[len("toys_"):]
        npz_path = p.parent / f"nsig_{truth_name}.npz"
        if not npz_path.exists():
            logger.warning("%s: no npz, rows kept", p)
            continue
        z = np.load(npz_path)
        rows = [_type_row(r) for r in read_csv(p)]
        n_done = 0
        for r in rows:
            if r.get("status") != "ok" or r.get("n_inj") is None:
                continue
            key = f"{r['candidate']}_{int(r['mass'])}_{r['inject']:g}"
            if key not in z.files:
                continue
            n, e, ok = z[key], z[key + "_err"], z[key + "_ok"].astype(bool)
            for k in PULL_KEYS:
                r[k] = None
            if ok.sum() >= 10:
                r.update(pull_stats((n[ok] - r["n_inj"]) / e[ok]))
            n_done += 1
        write_csv(p, rows)
        logger.info("%s: pull statistics of %d rows recomputed", p, n_done)


def compare_truths(all_rows, args, out, tag=""):
    """--compare-truths: the overlay figures per variant and candidate of --plot-candidates."""
    if not args.compare_truths:
        return
    for variant in sorted({r["variant"] for r in all_rows}):
        rows_by_truth = {tn: [r for r in all_rows if r["variant"] == variant and r["truth"] == tn] for tn in args.compare_truths}
        if any(not v for v in rows_by_truth.values()):
            logger.warning("%s: not every truth of %s has rows, overlay skipped", variant, args.compare_truths)
            continue
        cands = args.plot_candidates or ["expo"]
        for cand in cands:
            plot_truth_compare(all_rows[0]["channel"], variant, cand, rows_by_truth, Path(out) / "plots" / variant, tag=tag)
        if len(cands) > 1:
            plot_truth_compare(all_rows[0]["channel"], variant, cands, rows_by_truth, Path(out) / "plots" / variant, tag=tag)


def collect_rows(out):
    """All per-truth CSV rows under results/<channel>/<out>/<variant>/toys_<truth>.csv, typed."""
    from util import read_csv
    rows = []
    for p in sorted(Path(out).glob("*/toys_*.csv")):
        rows += [_type_row(r) for r in read_csv(p)]
    return rows


def merge_previous(odir, truth_name, cands, rows, arrays):
    """--append: keep the rows and N_sig arrays of an earlier run of this truth except the
    (candidate, mass) cells the new run redid, so a new candidate or new masses join the files
    instead of replacing them (masses added 2026-10-26: the grid grew downward to 1000 GeV)."""
    from util import read_csv
    csv_path, npz_path = Path(odir) / f"toys_{truth_name}.csv", Path(odir) / f"nsig_{truth_name}.npz"
    redone = {(r["candidate"], int(r["mass"])) for r in rows}
    old_rows = ([r for r in (_type_row(x) for x in read_csv(csv_path)) if (r["candidate"], int(r["mass"])) not in redone]
                if csv_path.exists() else [])
    old_arrays = {}
    if npz_path.exists():
        z = np.load(npz_path)
        for k in z.files:
            cand, mass = k.split("_")[0], int(k.split("_")[1])
            if (cand, mass) not in redone:
                old_arrays[k] = z[k]
    return old_rows + rows, {**old_arrays, **arrays}


def write_summary(channel, all_rows, path, title=None, range_text=None):
    from util import fmt
    L = [title or f"# Step 5, {channel} {C.REGION_NAME}: toys from every truth, full-range S+B fit with every candidate", "",
         (range_text or f"Fit range [{C.FIT_LO:.0f}, 8000] GeV") + f", {C.BIN_WIDTH:.0f} GeV bins, Poisson likelihood; Gaussian signal with the "
         "window-map mu(m_WR), sigma(m_WR) of the channel; N_sig free. S_spur = median N_sig / RMS N_sig at zero "
         "injection; bias(k) = (mean N_sig - injected) / RMS at k x the expected limit (1.96 sigma_fit of the Asimov "
         f"fit); pull statistics over the toys that passed the checks: width = sigma of the Gaussian fitted to "
         f"the pulls, expected 1 (no numeric tolerance is published). Pass line |bias| < {C.BIAS_PASS}, target "
         f"{C.BIAS_TARGET}. Cell format: S_spur | bias(1x) / bias(2x) | "
         "width | passed fraction.", ""]
    variants = []
    for r in all_rows:
        if r["variant"] not in variants:
            variants.append(r["variant"])
    for variant in variants:
        truths = []
        for r in all_rows:
            if r["variant"] == variant and r["truth"] not in truths:
                truths.append(r["truth"])
        for truth in truths:
            rs = [r for r in all_rows if r["variant"] == variant and r["truth"] == truth and r.get("status") == "ok"]
            if not rs:
                continue
            cands = []
            for r in rs:
                if r["candidate"] not in cands:
                    cands.append(r["candidate"])
            masses = sorted({r["mass"] for r in rs})
            L += [f"## {variant}, truth {truth}", "",
                  "| m_WR | B(+-2 sigma) | sigma_fit expo / powexp | " + " | ".join(cands) + " |",
                  "|---|---|---|" + "---|" * len(cands)]
            for m in masses:
                cells = []
                for cand in cands:
                    r0 = next((r for r in rs if r["candidate"] == cand and r["mass"] == m and r["inject"] == 0.0), None)
                    r1 = next((r for r in rs if r["candidate"] == cand and r["mass"] == m and r["inject"] == 1.0), None)
                    r2 = next((r for r in rs if r["candidate"] == cand and r["mass"] == m and r["inject"] == 2.0), None)
                    if r0 is None or r0.get("nsig_rms") is None:
                        cells.append("x")
                        continue
                    frac = r0["n_passed"] / max(r0["n_conv"], 1)
                    cells.append(f"{fmt(r0.get('bias_median'))} \\| {fmt(r1.get('bias_mean') if r1 else None)} / "
                                 f"{fmt(r2.get('bias_mean') if r2 else None)} \\| {fmt(r0.get('pull_width'))} \\| {frac:.2f}")
                r_e = next((r for r in rs if r["candidate"] == "expo" and r["mass"] == m and r["inject"] == 0.0), None)
                r_p = next((r for r in rs if r["candidate"] == "powexp" and r["mass"] == m and r["inject"] == 0.0), None)
                sig = f"{fmt(r_e.get('asimov_err') if r_e else None, 1)} / {fmt(r_p.get('asimov_err') if r_p else None, 1)}"
                b = next((r["b_core"] for r in rs if r["mass"] == m), float("nan"))
                L.append(f"| {m:.0f} | {b:.1f} | {sig} | " + " | ".join(cells) + " |")
            L += ["", "| candidate | max abs S_spur (mass) | masses with abs S_spur > 0.3 | > 0.5 | max abs bias at 1x / 2x | "
                  "min passed fraction | width range |", "|---|---|---|---|---|---|---|"]
            for cand in cands:
                r0s = [r for r in rs if r["candidate"] == cand and r["inject"] == 0.0 and r.get("bias_median") is not None]
                if not r0s:
                    continue
                worst = max(r0s, key=lambda r: abs(r["bias_median"]))
                n03 = [f"{r['mass']:.0f}" for r in r0s if abs(r["bias_median"]) > C.BIAS_TARGET]
                n05 = [f"{r['mass']:.0f}" for r in r0s if abs(r["bias_median"]) > C.BIAS_PASS]
                b1 = max((abs(r["bias_mean"]) for r in rs if r["candidate"] == cand and r["inject"] == 1.0 and r.get("bias_mean") is not None), default=float("nan"))
                b2 = max((abs(r["bias_mean"]) for r in rs if r["candidate"] == cand and r["inject"] == 2.0 and r.get("bias_mean") is not None), default=float("nan"))
                fr = min(r["n_passed"] / max(r["n_conv"], 1) for r in r0s)
                ws = [r["pull_width"] for r in r0s if isinstance(r.get("pull_width"), float)]
                L.append(f"| {cand} | {abs(worst['bias_median']):.2f} ({worst['mass']:.0f}) | {', '.join(n03) or '-'} | "
                         f"{', '.join(n05) or '-'} | {fmt(b1)} / {fmt(b2)} | {fr:.2f} | "
                         f"{fmt(min(ws)) if ws else '-'} - {fmt(max(ws)) if ws else '-'} |")
            L.append("")
    Path(path).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--variants", nargs="+", default=list(C.DY_VARIANTS))
    ap.add_argument("--truths", nargs="+", default=C.TRUTH_NAMES)
    ap.add_argument("--candidates", nargs="+", default=C.CANDIDATES)
    ap.add_argument("--masses", nargs="+", type=float, default=C.MASSES)
    ap.add_argument("--ntoys", type=int, default=C.NTOYS)
    ap.add_argument("--inject", nargs="+", type=float, default=C.INJECT)
    ap.add_argument("--workers", type=int, default=max(1, min(8, mp.cpu_count() - 1)))
    ap.add_argument("--lo", type=float, default=C.FIT_LO, help="fit start in GeV (6_windows.py overrides it per mass)")
    ap.add_argument("--out", default="4_toys", help="results/<channel>/<out>/ (another name for another fit start)")
    ap.add_argument("--truth-dir", default="3_truths", help="results/<channel>/<truth-dir>/ to take the truths from")
    ap.add_argument("--append", action="store_true",
                    help="keep the rows and N_sig arrays of the other candidates already in the per-truth files")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--summary-only", action="store_true", help="rebuild toys_all.csv and summary.md from the per-truth CSVs")
    ap.add_argument("--plots-only", action="store_true", help="rebuild the plots from the per-truth CSV and npz files")
    ap.add_argument("--plot-candidates", nargs="+", default=None, help="summary figures of these candidates only, saved with a suffix")
    ap.add_argument("--compare-truths", nargs="+", default=None, help="plots-only: overlay these truths per candidate of --plot-candidates (plot_truth_compare)")
    ap.add_argument("--restat", action="store_true",
                    help="recompute the pull statistics of the stored rows from the npz arrays, then rebuild "
                         "the CSVs, summary and plots")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    from wrplotter.cli_utils import setup_logging
    setup_logging(args.verbose)
    from spectrum import rebin
    from truth_sets import load_truth
    from util import write_csv
    ctx = mp.get_context("spawn")
    for channel in args.channels:
        out = C.RESULTS / channel / args.out
        (out / "plots").mkdir(parents=True, exist_ok=True)
        title = None if args.lo == C.FIT_LO else (f"# Step 5, {channel} {C.REGION_NAME}: toys from every truth, S+B fit from "
                                                   f"{args.lo:.0f} GeV with every candidate")
        range_text = None if args.lo == C.FIT_LO else f"Fit range [{args.lo:.0f}, 8000] GeV"
        all_rows = []
        if args.restat:
            restat(out)
            args.plots_only = True
        if args.summary_only or args.plots_only:
            all_rows = collect_rows(out)
            if args.plots_only:
                for p in sorted(out.glob("*/toys_*.csv")):
                    variant, truth_name = p.parent.name, p.stem[len("toys_"):]
                    rows = [r for r in all_rows if r["variant"] == variant and r["truth"] == truth_name]
                    z = np.load(p.parent / f"nsig_{truth_name}.npz")
                    arrays = {k: z[k] for k in z.files}
                    plot_truth_summary(channel, variant, truth_name, rows, arrays, out / "plots" / variant, only=args.plot_candidates)
                compare_truths(all_rows, args, out)
            write_csv(out / "toys_all.csv", all_rows)
            write_summary(channel, all_rows, out / "summary.md", title=title, range_text=range_text)
            continue
        for variant in args.variants:
            vdir = C.RESULTS / channel / args.truth_dir / variant
            vrows = []
            for truth_name in args.truths:
                tpath = vdir / f"truth_{truth_name}"
                if not tpath.with_suffix(".npz").exists():
                    logger.warning("%s %s %s: no truth file, skipped", channel, variant, truth_name)
                    continue
                edges10, parts, _ = load_truth(tpath)
                raw10 = None
                # the raw MC variances give the (unused by the Poisson fit) histogram errors
                e100, t100, var100 = rebin(edges10, parts["sum"], np.maximum(parts["sum"], 0.0))
                jobs = [{"channel": channel, "variant": variant, "truth_name": truth_name, "edges": e100,
                         "truth": t100, "variances": var100, "mass": m, "lo": args.lo, "candidates": args.candidates,
                         "ntoys": args.ntoys, "inject": args.inject,
                         "seed": C.SEED + 1000 * int(m)
                         + zlib.crc32(f"{channel}/{variant}/{truth_name}".encode()) % 997}
                        for m in args.masses]
                with ctx.Pool(args.workers) as pool:
                    results = pool.map(run_mass, jobs)
                rows, arrays = [], {}
                for r, a in results:
                    rows += r
                    arrays.update(a)
                if args.append:
                    rows, arrays = merge_previous(out / variant, truth_name, args.candidates, rows, arrays)
                vrows += rows
                (out / variant).mkdir(parents=True, exist_ok=True)
                write_csv(out / variant / f"toys_{truth_name}.csv", rows)
                np.savez_compressed(out / variant / f"nsig_{truth_name}.npz", **arrays)
                if not args.no_plots:
                    plot_truth_summary(channel, variant, truth_name, rows, arrays, out / "plots" / variant, only=args.plot_candidates)
                ok = [r for r in rows if r.get("status") == "ok" and r["inject"] == 0.0 and r.get("bias_median") is not None]
                for cand in args.candidates:
                    cs = [r for r in ok if r["candidate"] == cand]
                    if cs:
                        worst = max(cs, key=lambda r: abs(r["bias_median"]))
                        logger.info("%s %s %-14s %-9s max|S_spur| %.2f at %4.0f; >0.3 at %s", channel, variant, truth_name,
                                    cand, abs(worst["bias_median"]), worst["mass"],
                                    ",".join(f"{r['mass']:.0f}" for r in cs if abs(r["bias_median"]) > C.BIAS_TARGET) or "-")
            all_rows = collect_rows(out)                 # everything on disk for this channel, all phases
        write_csv(out / "toys_all.csv", all_rows)
        write_summary(channel, all_rows, out / "summary.md", title=title, range_text=range_text)
        logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
