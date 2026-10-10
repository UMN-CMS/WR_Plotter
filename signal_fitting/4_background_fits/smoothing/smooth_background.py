#!/usr/bin/env python3
"""Stage 4 / smoothing -- build the smoothed per-component background.

Runs the four smoothers of smooth_lib (kernel, hybrid, param, kde) on each
background component of the run2 2018 ee-resolved SR (native 10 GeV bins) and
writes, per method,

  run2/ee_resolved/smoothed_<method>.root   TH1D per component (content =
        smoothed, error = sqrt(raw sumw2)), <comp>_raw, bkg_smooth, bkg_raw
        and bkg_native (= bkg_smooth clipped >= 0, no errors: the drop-in
        twin of 8_combine_limits/optimization/inputs/ee_resolved.root)
  run2/ee_resolved/smoothed_<method>.json   hyperparameters + per-component
        diagnostics (trust masses, fit parameters and verdicts, rescale
        factors, clipped bins, totals)

and the plots (all at the 50 GeV card binning):

  plots/spectrum_<comp>.*   raw MC +- stat band with the four smoothers
                            overlaid, ratio panel smooth / raw
  plots/spectrum_sum.*      the same for the summed background
  plots/neff.*              running N_eff per component (the trust-mass rule)
                            and the kernel bandwidth h(m)
  plots/seam_<comp>.*       the hybrid seam: raw, tail fit, blend band

--scan writes the hyperparameter-variant grid to scan/<method>__<variant>.*
(no plots); compare_methods.py turns it into the smoothing systematic.

  PYTHONPATH=/opt/homebrew/lib/root ../../../.venv-fitting/bin/python smooth_background.py -v
  (LPC: source LCG_106 first)
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import smooth_lib as sl                                                  # noqa: E402
from smooth_lib import (COMP_LABEL, METHODS, METHOD_COLOR, METHOD_LABEL,   # noqa: E402
                        centers_of, rebin)
from wrplotter.cli_utils import setup_logging                            # noqa: E402
from wrplotter.config import load_lumi                                   # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root                # noqa: E402
from wrplotter.plotting_helpers import custom_log_formatter              # noqa: E402
from plot_mc_backgrounds import draw_stat_band                           # noqa: E402

logger = logging.getLogger("smooth_background")

# house colors (data/sample_groups.yaml)
COMP_COLOR = sl.COMP_COLOR
XMIN, XMAX = 800.0, 6000.0

# hyperparameter variants for --scan: method -> list of (name, kwargs override)
SCAN = {
    "kernel": [(f"nt{nt}_hmin{hm:g}_d{d}",
                {"n_target": float(nt), "h_min": hm, "degree": d})
               for nt, hm, d in itertools.product((10, 20, 40),
                                                  (0.03, 0.05, 0.1), (1, 2))],
    "hybrid": [(f"nt{nt}_{fam}_w{tw:.0f}",
                {"trust_neff": float(nt), "family": fam, "trust_width": tw})
               for nt, fam, tw in itertools.product((10, 20, 40),
                                                    ("expo", "powexp", "expo2"),
                                                    (500.0, 1000.0))]
              # diagnostic fit statistics (known-biased on mixed-weight MC:
              # chi2 = Neyman, neff = single-weight-per-bin assumption) --
              # written but excluded from the systematic by compare_methods
              + [(f"diag_{st}", {"stat": st}) for st in ("neff", "chi2")],
    "param": [(f"{fam}_{rn}",
               {"families": (fam,), "lo": lo, "hi": hi})
              for fam, (rn, lo, hi) in itertools.product(
                  ("expo", "powexp", "expo2"),
                  (("full", 800.0, None), ("r1000-3500", 1000.0, 3500.0)))]
             + [(f"diag_{st}", {"stat": st}) for st in ("neff", "chi2")],
    "kde": [(f"rho{rho:g}_{mir}_rs{rs}",
             {"rho": rho, "mirror": mir, "resample": rs})
            for rho, mir, rs in itertools.product((0.5, 1.0, 2.0),
                                                  ("MirrorLeft", "NoMirror"),
                                                  (1, 5))],
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--group-rest", action="store_true",
                   help="merge Nonprompt+Other into one 'rest' component")
    p.add_argument("--era", default=sl.ERA,
                   help="analysis era of the background histograms "
                        "(default RunIISummer20UL18; e.g. RunIII2024Summer24)")
    p.add_argument("--bkg-dir", default=sl.BKG_DIR,
                   help="rootfiles subdir of the background histograms "
                        "(default: the legacy converted set)")
    p.add_argument("--entries", default="bins", choices=["bins", "events"],
                   help="kde entries: one per populated 10 GeV bin (default) or "
                        "one per MC event from the events_<region> trees "
                        "(--save-events runs); events implies --methods kde")
    p.add_argument("--components", default=None, choices=list(sl.COMPONENT_SETS),
                   help="component set (default legacy4; split8 = the "
                        "per-process WrCoffea rerun)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default run2/ee_resolved (legacy) or "
                        "run2/ee_resolved_<components> otherwise")
    p.add_argument("--bw", type=float, default=50.0, help="plot bin width (card)")
    p.add_argument("--scan", action="store_true",
                   help="write the hyperparameter-variant grid to scan/")
    p.add_argument("--no-plots", action="store_true")
    # kernel
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--h-min", type=float, default=0.05)
    p.add_argument("--h-max", type=float, default=0.4)
    p.add_argument("--n-target", type=float, default=20.0)
    # hybrid
    p.add_argument("--trust-neff", type=float, default=20.0)
    p.add_argument("--trust-width", type=float, default=500.0)
    p.add_argument("--l-below", type=float, default=1000.0)
    p.add_argument("--l-above", type=float, default=500.0)
    p.add_argument("--l-blend", type=float, default=200.0)
    p.add_argument("--tail-family", default="expo",
                   choices=["expo", "powexp", "expo2"])
    p.add_argument("--tail-stat", default="poisson",
                   choices=["poisson", "wpoisson", "neff", "chi2"])
    p.add_argument("--conserve-total", action="store_true")
    # param
    p.add_argument("--param-families", nargs="+", default=["expo", "powexp", "expo2"])
    p.add_argument("--param-lo", type=float, default=800.0)
    p.add_argument("--param-hi", type=float, default=None)
    p.add_argument("--param-pivot", type=float, default=2000.0)
    p.add_argument("--param-stat", default="poisson",
                   choices=["poisson", "wpoisson", "neff", "chi2"])
    # kde
    p.add_argument("--kde-var", default="m", choices=["logm", "m"])
    p.add_argument("--kde-mirror", default="MirrorLeft", choices=list(sl._MIRROR))
    p.add_argument("--kde-rho", type=float, default=1.0)
    p.add_argument("--kde-resample", type=int, default=1)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def method_kwargs(args, method):
    if method == "kernel":
        return {"degree": args.degree, "h_min": args.h_min, "h_max": args.h_max,
                "n_target": args.n_target}
    if method == "hybrid":
        return {"trust_neff": args.trust_neff, "trust_width": args.trust_width,
                "l_below": args.l_below, "l_above": args.l_above,
                "l_blend": args.l_blend, "family": args.tail_family,
                "stat": args.tail_stat, "conserve_total": args.conserve_total}
    if method == "param":
        return {"families": tuple(args.param_families), "lo": args.param_lo,
                "hi": args.param_hi, "pivot": args.param_pivot,
                "stat": args.param_stat}
    if method == "kde":
        return {"var_kind": args.kde_var, "mirror": args.kde_mirror,
                "rho": args.kde_rho, "resample": args.kde_resample}
    raise ValueError(method)


def run_method(method, kwargs, comps):
    """{comp: smoothed}, {comp: diag}"""
    fn = sl.SMOOTHERS[method]
    smooth, diags = {}, {}
    for name, (e, v, var) in comps.items():
        s, d = fn(e, v, var, **kwargs)
        smooth[name], diags[name] = s, d
        c = centers_of(e)
        msg = (f"  {method:6s} {name:9s} raw {v[c >= XMIN].sum():7.2f} -> "
               f"smooth {s.sum():7.2f}  clipped {d['n_clipped']:3d}")
        if method == "hybrid":
            f = d["fit"]
            msg += (f"  m_T={d['m_T']:.0f} {f['name']}({f['stat']}) "
                    f"b={f['params'][1]:.2f} passed={f['passed']} "
                    f"seam={d['seam_pull']:+.2f}s tail={d['tail_pull']:+.2f}s")
        elif method == "param":
            f = d["fit"]
            msg += (f"  best={d['best']}({f['stat']}) chi2P/ndf="
                    f"{f['chi2_pearson'] / f['ndf_pearson']:.2f} "
                    f"passed={f['passed']} fit/raw={d['total_ratio']:.3f}")
        elif method == "kernel":
            msg += f"  rescale={d['rescale']:.4f} n_at_hmax={d['n_at_hmax']}"
        elif method == "kde":
            msg += f"  entries={d['n_entries']} dropped_neg={d['n_negative_dropped']}"
        logger.info(msg)
    return smooth, diags


def write_outputs(out_root, out_json, method, kwargs, edges, smooth, comps, diags):
    sl.write_smoothed_root(out_root, edges, smooth, comps)
    c = centers_of(edges)
    payload = {
        "method": method, "kwargs": kwargs, "era": sl.ERA, "bkg_dir": sl.BKG_DIR,
        "lumi": float(load_lumi(sl.ERA)["lumi"]),
        "region": sl.REGION, "variable": sl.VARIABLE, "threshold": sl.THRESHOLD,
        "components": {n: sl.jsonable(d) for n, d in diags.items()},
        "totals": {n: {"raw": float(comps[n][1][c >= XMIN].sum()),
                       "raw_err": float(np.sqrt(comps[n][2][c >= XMIN].sum())),
                       "smooth": float(smooth[n].sum())} for n in comps},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(sl.jsonable(payload), fh, indent=1)
    logger.info("  wrote %s / %s", out_root.name, out_json.name)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def _cms(ax, lumi, com, fontsize=16):
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=fontsize)


def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("  wrote %s", out.with_suffix(".png"))


def plot_spectrum(name, raw, smooth_by_method, *, bw, lumi, com, out):
    """Raw MC (50 GeV) +- stat band with the smoothers; ratio smooth/raw."""
    e, v, var = raw
    f = int(round(bw / sl.NATIVE_BW))
    e5, v5, var5 = rebin(e, v, var, f)
    c5 = centers_of(e5)
    hep.style.use("CMS")
    fig, (ax, axr) = plt.subplots(2, 1, sharex=True, figsize=(10, 10),
                                  gridspec_kw={"height_ratios": [3, 1],
                                               "hspace": 0.06})
    ax.stairs(v5, e5, color="black", linewidth=1.6, label="MC (raw)", zorder=4)
    draw_stat_band(ax, e5, v5, var5, label="MC stat.")
    for method, s in smooth_by_method.items():
        _, s5, _ = rebin(e, s, np.zeros_like(s), f)
        pos = s5 > 0
        ax.plot(c5[pos], s5[pos], color=METHOD_COLOR[method], linewidth=1.8,
                label=METHOD_LABEL[method], zorder=5)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(v5 > 0, s5 / v5, np.nan)
        axr.plot(c5, ratio, color=METHOD_COLOR[method], linewidth=1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(v5 > 0, np.sqrt(var5) / v5, np.nan)
    axr.stairs(1 + rel, e5, baseline=1 - rel, color="gray", alpha=0.3, fill=True,
               label="raw MC stat.")
    axr.axhline(1.0, color="black", linewidth=0.8)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    vis = v5[(c5 >= XMIN) & (c5 <= XMAX) & (v5 > 0)]
    ax.set_ylim(max(vis.min() / 5, vis.max() * 1e-6), vis.max() * 40)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylabel(f"Events / {bw:.0f} GeV")
    ax.text(0.04, 0.95, f"{COMP_LABEL[name]}\nee resolved SR\n{sl.ERA}",
            transform=ax.transAxes, fontsize=14, va="top")
    ax.legend(loc="upper right", fontsize=12)
    axr.set_ylim(0.0, 2.5)
    axr.set_ylabel("smooth / raw", fontsize=16)
    axr.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    axr.legend(loc="upper left", fontsize=11)
    _cms(ax, lumi, com)
    _save(fig, out)


def plot_neff(comps, diags_by_method, args, *, lumi, com, out):
    """Running N_eff (trust-mass rule) and the kernel bandwidth per component."""
    hep.style.use("CMS")
    fig, (ax, axh) = plt.subplots(2, 1, sharex=True, figsize=(10, 10),
                                  gridspec_kw={"height_ratios": [2, 1],
                                               "hspace": 0.06})
    for name, (e, v, var) in comps.items():
        ms, ne = sl.running_neff(e, v, var, args.trust_width)
        ax.plot(ms + 0.5 * args.trust_width, np.where(ne > 0, ne, np.nan),
                color=COMP_COLOR[name], linewidth=1.8, label=COMP_LABEL[name])
        hd = diags_by_method.get("hybrid", {}).get(name)
        if hd:
            ax.axvline(hd["m_T"], color=COMP_COLOR[name], linestyle=":",
                       linewidth=1.2)
        kd = diags_by_method.get("kernel", {}).get(name)
        if kd:
            axh.plot(kd["h_of_m"]["m"], kd["h_of_m"]["h"],
                     color=COMP_COLOR[name], linewidth=1.6)
    ax.axhline(args.trust_neff, color="gray", linestyle="--", linewidth=1.0,
               label=f"trust N_eff = {args.trust_neff:g}")
    ax.set_yscale("log")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylabel(rf"$N_{{\rm eff}}$ in [m, m+{args.trust_width:.0f} GeV]")
    ax.text(0.04, 0.95, f"ee resolved SR\n{sl.ERA}\n"
            f"dotted: hybrid trust mass $m_T$", transform=ax.transAxes,
            fontsize=13, va="top")
    ax.legend(loc="upper right", fontsize=12)
    axh.axhline(args.h_min, color="gray", linestyle="--", linewidth=0.8)
    axh.axhline(args.h_max, color="gray", linestyle="--", linewidth=0.8)
    axh.set_ylim(0, args.h_max * 1.15)
    axh.set_ylabel("kernel $h$ [ln m]", fontsize=16)
    axh.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    _cms(ax, lumi, com)
    _save(fig, out)


def plot_seam(name, raw, s_hybrid, d, *, bw, lumi, com, out):
    """Hybrid seam zoom: raw points, tail fit (solid in its range), blend band."""
    e, v, var = raw
    f = int(round(bw / sl.NATIVE_BW))
    e5, v5, var5 = rebin(e, v, var, f)
    c5 = centers_of(e5)
    _, s5, _ = rebin(e, s_hybrid, np.zeros_like(s_hybrid), f)
    fit = d["fit"]
    lo_x = max(XMIN, d["m_T"] - 1500.0)
    hi_x = min(XMAX, max(d["m_T"] + 1200.0, d["m_last"] + 200.0))
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 7.5))
    sel = (c5 >= lo_x) & (c5 <= hi_x)
    ax.errorbar(c5[sel], v5[sel], yerr=np.sqrt(var5[sel]), fmt="o", color="black",
                markersize=4, linewidth=1.0, label="MC (raw)")
    mgrid = np.linspace(lo_x, hi_x, 400)
    fgrid = sl.eval_fit(fit, mgrid, bw, bw)
    inr = (mgrid >= fit["range"][0]) & (mgrid <= fit["range"][1])
    ax.plot(mgrid[inr], fgrid[inr], color=METHOD_COLOR["hybrid"], linewidth=2.0,
            label=f"tail fit {fit['name']} ({fit['stat']}), b = {fit['params'][1]:.2f}")
    ax.plot(mgrid[~inr], fgrid[~inr], color=METHOD_COLOR["hybrid"], linewidth=1.2,
            linestyle="--")
    ax.stairs(np.where(s5 > 0, s5, np.nan), e5, color=METHOD_COLOR["hybrid"],
              linewidth=1.2, linestyle=":", label="hybrid result")
    ax.axvspan(d["m_T"] - d["l_blend"], d["m_T"], color="gold", alpha=0.25,
               label="blend band")
    ax.axvline(d["m_T"], color="gray", linewidth=1.0)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    vis = v5[sel & (v5 > 0)]
    ax.set_ylim(max(vis.min() / 10, vis.max() * 1e-5), vis.max() * 30)
    ax.set_xlim(lo_x, hi_x)
    ax.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    ax.set_ylabel(f"Events / {bw:.0f} GeV")
    ax.text(0.04, 0.95,
            f"{COMP_LABEL[name]}\n$m_T$ = {d['m_T']:.0f} GeV "
            f"(N_eff < {d['trust_neff']:g} in {d['trust_width']:.0f} GeV)\n"
            f"seam pull {d['seam_pull']:+.2f}$\\sigma$, "
            f"tail raw/fit = {d['tail_raw']:.2f}/{d['tail_fit']:.2f} "
            f"({d['tail_pull']:+.2f}$\\sigma$)",
            transform=ax.transAxes, fontsize=13, va="top")
    ax.legend(loc="upper right", fontsize=12)
    _cms(ax, lumi, com)
    _save(fig, out)


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    sl.ERA = args.era
    info = load_lumi(sl.ERA)
    lumi, com = info["lumi"], info.get("com", 13.0)
    run_sub = {"RunII": "run2", "Run3": "run3"}[str(info["run"])]
    sl.BKG_DIR = args.bkg_dir
    if args.entries == "events":
        args.methods = ["kde"]
    if args.output_dir is None:
        tag = "" if args.components in (None, "legacy4") else f"_{args.components}"
        if args.entries == "events":
            tag += "_events"
        args.output_dir = HERE / run_sub / f"ee_resolved{tag}"
    dirs, _ = input_dirs_for_era(sl.ERA, repo_root(), sl.BKG_DIR)
    comps = sl.load_components(dirs, group_rest=args.group_rest,
                               component_set=args.components)
    edges = next(iter(comps.values()))[0]
    logger.info("components: %s  (%d native bins, %.0f GeV) from %s", list(comps),
                len(edges) - 1, edges[1] - edges[0], sl.BKG_DIR)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    events = None
    if args.entries == "events":
        cset = args.components or "legacy4"
        events = sl.load_components_events(dirs, cset)
        for name, (m, w) in events.items():
            logger.info("  events %-9s %d MC events, sum of weights %.2f (hist %.2f)",
                        name, len(m), w[(m >= XMIN) & (m < edges[-1])].sum(),
                        comps[name][1][centers_of(edges) >= XMIN].sum())

    smooth_by_method, diags_by_method = {}, {}
    for method in args.methods:
        kw = method_kwargs(args, method)
        if events is not None:
            kw = {k: v for k, v in kw.items() if k != "resample"}
            kw["entries"] = "events"
            logger.info("== kde (one entry per MC event) %s", kw)
            smooth, diags = {}, {}
            for name, (m, w) in events.items():
                kw_call = {k: v for k, v in kw.items() if k != "entries"}
                smooth[name], diags[name] = sl.smooth_kde_events(edges, m, w, **kw_call)
                logger.info("  kde    %-9s events %d (neg %d) -> smooth %.2f  clipped %d",
                            name, diags[name]["n_entries"], diags[name]["n_negative_dropped"],
                            smooth[name].sum(), diags[name]["n_clipped"])
        else:
            logger.info("== %s %s", method, kw)
            smooth, diags = run_method(method, kw, comps)
        smooth_by_method[method], diags_by_method[method] = smooth, diags
        write_outputs(out / f"smoothed_{method}.root", out / f"smoothed_{method}.json",
                      method, kw, edges, smooth, comps, diags)

    if not args.no_plots:
        pdir = out / "plots"
        for name in comps:
            plot_spectrum(name, comps[name],
                          {m: smooth_by_method[m][name] for m in args.methods},
                          bw=args.bw, lumi=lumi, com=com,
                          out=pdir / f"spectrum_{name}")
        raw_sum = sl.sum_components(comps)
        plot_spectrum("sum", raw_sum,
                      {m: sum(smooth_by_method[m].values()) for m in args.methods},
                      bw=args.bw, lumi=lumi, com=com, out=pdir / "spectrum_sum")
        plot_neff(comps, diags_by_method, args, lumi=lumi, com=com, out=pdir / "neff")
        if "hybrid" in args.methods:
            for name in comps:
                plot_seam(name, comps[name], smooth_by_method["hybrid"][name],
                          diags_by_method["hybrid"][name], bw=args.bw, lumi=lumi,
                          com=com, out=pdir / f"seam_{name}")

    if args.scan:
        sdir = out / "scan"
        for method in args.methods:
            base = method_kwargs(args, method)
            for vname, override in SCAN[method]:
                kw = {**base, **override}
                logger.info("== scan %s %s", method, vname)
                try:
                    smooth, diags = run_method(method, kw, comps)
                except Exception as exc:              # a variant may not fit
                    logger.warning("  %s %s failed: %s", method, vname, exc)
                    continue
                stem = f"{method}__{vname}"
                write_outputs(sdir / f"{stem}.root", sdir / f"{stem}.json",
                              method, kw, edges, smooth, comps, diags)


if __name__ == "__main__":
    main()
