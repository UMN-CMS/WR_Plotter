#!/usr/bin/env python3
"""Step 8: the simultaneous SR + e-mu flavour-CR fit on the step 5 toys (tried, not pursued).

Model (lib/simfit.py): SR = TF x f_tt + f_rest + N_sig x S, CR = f_tt, with f_tt a registry form
fitted to the whole e-mu CR and shared with the SR as the non-DY part (TF = SR non-DY / CR,
fixed to the truth ratio or free), f_rest a second form for the DY. Configurations
(candidate names fcr_<tt form>_<DY form>_<fix|free>):

  fcr_powexp_powexp_fix    both power x exponential, TF fixed      (the data-driven hybrid)
  fcr_powexp_powexp_free   the same with TF free                   (the CR fixes shape only)
  fcr_expo_powexp_fix      exponential tt shape                    (is the tt shape simpler?)
  fcr_powexp_expo_fix      exponential DY                          (is the DY shape simpler?)
  fcr_expo_expo_fix        both exponentials, TF fixed             (the double-exponential version)
  fcr_atlasemu_powexp_fix  running power law tt shape              (the alternative tt family)

Truths: the SR truths of step 4 (per DY variant and smoother; the DY part and the non-DY part
are kept apart) and the raw e-mu CR MC (tt+tW N_eff 92k above 800 GeV) as the CR truth; toys
Poisson in both regions, signal injected in the SR only. Statistics, rows, files and plots as
in 4_toys.py, so 5_decide.py --step 8_flavor_cr judges them.

--data-fit: fit the registry forms to the real e-mu CR DATA (unblinded by construction) from
the fit start: Poisson deviance, data F-test ladder, fitted tails, a fit-and-pull figure.
Output results/flavor_cr/.

  cd WR_Plotter && python background_fit/8_flavor_cr.py --workers 8
  ... background_fit/8_flavor_cr.py --data-fit
"""
from __future__ import annotations

import argparse
import importlib
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
import published_functions as pf                                         # noqa: E402

T = importlib.import_module("4_toys")
logger = logging.getLogger("8_flavor_cr")
PRIMARY = ["lo_ht", "nlo_mll"]
CONFIGS = [("powexp", "powexp", "fix"), ("powexp", "powexp", "free"), ("expo", "powexp", "fix"),
           ("powexp", "expo", "fix"), ("expo", "expo", "fix"), ("atlasemu", "powexp", "fix")]
PALETTE = ["#e42536", "#f89c20", "#5790fc", "#964a8b", "#9c9ca1", "#7a21dd"]
DATA_MENU = ["expo", "powlaw", "powexp", "atlasemu", "wprime7", "dilep8", "expo2"]


def cfg_name(tt, dy, tf):
    return f"fcr_{tt}_{dy}_{tf}"


def register_labels():
    """Colours, line styles and labels of the configurations for the step 5 plot helpers."""
    for i, (tt, dy, tf) in enumerate(CONFIGS):
        n = cfg_name(tt, dy, tf)
        pf.COLOR[n] = PALETTE[i % len(PALETTE)]
        pf.LS[n] = "-" if tf == "fix" else "--"
        C.MENU_LABEL[n] = f"CR {tt} + DY {dy}, TF {'fixed' if tf == 'fix' else 'free'}"


register_labels()


# ---------------------------------------------------------------------------------------------
# worker: one (truth, mass) cell, every configuration, Asimov + toys
# ---------------------------------------------------------------------------------------------

def run_mass_sim(job):
    import ROOT
    from sbfit import bkg_only, signal_pars, signal_shape
    from simfit import HI, SimModel, concat_edges, concat_values
    from spectrum import centers
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    register_labels()
    e, lo, m = job["edges"], job["lo"], job["mass"]
    sr, sr_dy, sr_nondy, cr = job["sr"], job["sr_dy"], job["sr_nondy"], job["cr"]
    c = centers(e)
    sl, e_cat = concat_edges(e, lo)
    mu, sigma = signal_pars(job["channel"], m)
    shape = signal_shape(e, mu, sigma)
    core = (c >= mu - C.SIG_CORE_K * sigma) & (c <= mu + C.SIG_CORE_K * sigma)
    b_core = float(sr[core].sum())
    nsig_bounds = (-0.8 * b_core, max(5.0 * b_core + 50.0, 200.0))
    tf0 = float(sr_nondy[sl].sum() / cr[sl].sum())
    rng = ROOT.TRandom3(job["seed"])
    rows, arrays = [], {}
    seeds = {}

    def seed_of(kind, name):
        key = (kind, name)
        if key not in seeds:
            v = cr if kind == "tt" else sr_dy
            fb = bkg_only(name, e, v, np.maximum(v, 0.0), lo, HI, {})
            seeds[key] = None if fb is None else list(fb["params"])
        return seeds[key]

    asim_cat = concat_values(sr, cr, sl)
    for tt, dy, tfm in job["configs"]:
        name = cfg_name(tt, dy, tfm)
        base = {"channel": job["channel"], "variant": job["variant"], "truth": job["truth_name"],
                "candidate": name, "mass": m, "mu": mu, "sigma": sigma, "b_core": b_core, "lo": lo, "hi": HI,
                "tf_true": tf0}
        tts, dys = seed_of("tt", tt), seed_of("dy", dy)
        if tts is None or dys is None:
            rows.append({**base, "inject": 0.0, "status": "no background fit"})
            continue
        model = SimModel(tt, dy, tfm, mu, sigma, lo)
        asim = model.fit(e_cat, asim_cat, tts, dys, tf0, 0.0, nsig_bounds)
        if asim is None or asim["nsig_err"] <= 0:
            rows.append({**base, "inject": 0.0, "status": "no Asimov fit"})
            continue
        ul = 1.96 * asim["nsig_err"]
        for k in job["inject"]:
            n_inj = k * ul
            expect_sr = (sr + n_inj * shape)[sl]
            expect_cr = cr[sl]
            nsigs, errs, passed, tfs = [], [], [], []
            for _ in range(job["ntoys"]):
                toy = np.array([rng.Poisson(max(x, 0.0)) for x in expect_sr] + [rng.Poisson(max(x, 0.0)) for x in expect_cr], float)
                r = model.fit(e_cat, toy, asim["tt"], asim["dyp"], asim["tf"], n_inj, nsig_bounds)
                if r is None:
                    continue
                nsigs.append(r["nsig"])
                errs.append(r["nsig_err"])
                passed.append(r["passed"] and r["nsig_err"] > 0)
                tfs.append(r["tf"])
            a_n, a_e, a_p, a_t = np.array(nsigs), np.array(errs), np.array(passed, bool), np.array(tfs)
            row = {**base, "inject": k, "n_inj": n_inj, "ul_expected": ul,
                   "asimov_nsig": asim["nsig"], "asimov_err": asim["nsig_err"],
                   "asimov_pull": asim["nsig"] / asim["nsig_err"], "asimov_passed": asim["passed"],
                   "asimov_tf": asim["tf"], "asimov_tf_err": asim["tf_err"],
                   "n_toys": job["ntoys"], "n_conv": int(len(a_n)), "n_passed": int(a_p.sum()), "status": "ok"}
            if len(a_n):
                med, rms = float(np.median(a_n)), float(a_n.std())
                row.update({"nsig_median": med, "nsig_mean": float(a_n.mean()), "nsig_rms": rms,
                            "bias_median": (med - n_inj) / rms if rms > 0 else float("nan"),
                            "bias_mean": (float(a_n.mean()) - n_inj) / rms if rms > 0 else float("nan"),
                            "sspur_asimov_units": (med - n_inj) / asim["nsig_err"],
                            "tf_mean": float(a_t.mean()), "tf_rms": float(a_t.std())})
                if a_p.sum() >= 10:
                    from pulls import pull_stats
                    row.update(pull_stats((a_n[a_p] - n_inj) / a_e[a_p]))
            rows.append(row)
            arrays[f"{name}_{int(m)}_{k:g}"] = a_n
            arrays[f"{name}_{int(m)}_{k:g}_err"] = a_e
            arrays[f"{name}_{int(m)}_{k:g}_ok"] = a_p
    return rows, arrays


# ---------------------------------------------------------------------------------------------
# the real e-mu CR data
# ---------------------------------------------------------------------------------------------

def data_fit(lo_list, out):
    """Every registry form of DATA_MENU fitted to the e-mu CR data from each start: Poisson
    deviance against the saturated model, the data F-test ladder, fitted tails; a figure per start."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import mplhep as hep
    import plots as P
    from fits import fit_poisson, seeds_for, wilks
    from simfit import HI, load_cr, load_cr_data
    from spectrum import centers, rebin
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, custom_log_formatter, fit_header
    out.mkdir(parents=True, exist_ok=True)
    e10, d10, _ = load_cr_data()
    e, d, _ = rebin(e10, d10, d10)
    comps = load_cr()
    _, mc, mcvar = rebin(*comps["sum"])
    c = centers(e)
    L = ["# The e-mu flavour CR data (2024, Muon stream) fitted with the registry forms", "",
         f"Resolved flavour CR, m(lljj) in {C.BIN_WIDTH:.0f} GeV bins, Poisson likelihood over [start, {HI:.0f}] GeV, "
         "every parameter free. dev = deviance against the saturated model (2 sum[mu - n + n ln(n/mu)]), with the "
         "chi2 p-value for ndf = bins - parameters (indicative: the bins above ~3 TeV hold 0-2 events); "
         "2 dNLL = the data F-test statistic of the nested pair (Wilks p); tails = fitted / observed events "
         "above 2 and 3 TeV. MC = the 2024 simulation (LO HT DY) for comparison.", ""]
    for lo in lo_list:
        sel = (c > lo) & (c < HI)
        nb = int(sel.sum())
        n20, n30 = float(d[c > 2000].sum()), float(d[c > 3000].sum())
        L += [f"## Fit start {lo:.0f} GeV ({nb} bins, {d[sel].sum():.0f} events; MC {mc[sel].sum():.1f}; data above 2 / 3 TeV: {n20:.0f} / {n30:.0f}, MC {mc[c > 2000].sum():.1f} / {mc[c > 3000].sum():.1f})", "",
              "| form | np | dev / ndf (p) | fitted / data > 2 TeV | > 3 TeV | parameters |", "|---|---|---|---|---|---|"]
        fits = {}
        for name in sorted(DATA_MENU, key=lambda n: pf.FUNCS[n][0]):
            seeds = seeds_for(name, c, d, np.maximum(d, 1.0), C.PIVOT, fits)
            fr = fit_poisson(name, e, d, np.maximum(d, 1.0), lo, HI, C.PIVOT, seeds)
            fits[name] = fr
            if fr is None:
                L.append(f"| {name} | {pf.FUNCS[name][0]} | failed | | | |")
                continue
            f = fr["tf1"]
            model = np.array([f.Eval(x) for x in c])
            dev = fr["dev"]
            ndf = nb - fr["npar"]
            p = pf.chi2_pvalue(dev, ndf)
            t20 = float(model[(c > 2000) & (c < HI)].sum()); t30 = float(model[(c > 3000) & (c < HI)].sum())
            flag = "" if fr["passed"] else " !"
            L.append(f"| {name}{flag} | {fr['npar']} | {dev:.1f} / {ndf} ({p:.3f}) | {t20:.1f} / {n20:.0f} | {t30:.1f} / {n30:.0f} | "
                     + ", ".join(f"{v:.3g}" for v in fr["params"]) + " |")
        L += ["", "| parent -> child | 2 dNLL | Wilks p | add? |", "|---|---|---|---|"]
        for par, child in [("expo", "powexp"), ("expo", "expo2"), ("powexp", "dilep8"), ("powlaw", "atlasemu"), ("powlaw", "wprime7")]:
            if fits.get(par) and fits.get(child):
                dn = max(fits[par]["dev"] - fits[child]["dev"], 0.0)
                _, pw = wilks(fits[par]["dev"], fits[child]["dev"], pf.FUNCS[child][0] - pf.FUNCS[par][0])
                L.append(f"| {par} -> {child} | {dn:.2f} | {pw:.3f} | {'yes' if pw < C.F_ALPHA else 'no'} |")
        L.append("")
        # figure: data, MC, fits, pulls
        hep.style.use("CMS")
        fig, (ax, axr) = plt.subplots(2, 1, sharex=True, figsize=(10, 10),
                                      gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1})
        xs = c[sel]
        ax.errorbar(xs, d[sel], yerr=np.sqrt(np.maximum(d[sel], 1.0)), fmt="o", color="k", ms=4, label="Data, e$\\mu$ CR")
        ax.step(e[:-1][sel], np.maximum(mc[sel], 1e-3), where="post", color="0.5", lw=1.3, label="Simulation (LO HT DY)")
        grid = np.linspace(lo, HI, 400)
        for name in ["expo", "powexp", "atlasemu", "dilep8"]:
            fr = fits.get(name)
            if fr is None:
                continue
            f = fr["tf1"]
            ax.plot(grid, [max(f.Eval(x), 1e-3) for x in grid], color=pf.COLOR[name], ls=pf.LS[name], lw=1.6,
                    label=f"{C.MENU_LABEL.get(name, name)}: dev {fr['dev']:.0f} / {nb - fr['npar']}")
            model = np.array([f.Eval(x) for x in xs])
            axr.plot(xs, (d[sel] - model) / np.sqrt(np.maximum(model, 1e-9)), marker="o", ms=3, lw=0, color=pf.COLOR[name])
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
        ax.set_ylim(1e-2, 50 * max(d[sel].max(), 1.0))
        ax.set_xlim(lo, 6000)
        ax.set_ylabel(f"Events / {C.BIN_WIDTH:.0f} GeV")
        axr.set_ylim(-4, 4)
        axr.axhline(0, color="k", lw=0.8)
        axr.set_ylabel(r"(data $-$ fit) / $\sqrt{\rm fit}$")
        axr.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
        cms_label(ax, C.LUMI, C.COM, data=True, text=f"e$\\mu$\nResolved flavour CR\nfit from {lo:.0f} GeV", year=C.YEAR)
        ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
        fit_header(ax)
        P.save(fig, out / f"cr_data_fit_lo{lo:.0f}")
    (out / "cr_data_fit.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--variants", nargs="+", default=PRIMARY)
    ap.add_argument("--truths", nargs="+", default=C.TRUTH_NAMES)
    ap.add_argument("--configs", nargs="+", default=[cfg_name(*c) for c in CONFIGS])
    ap.add_argument("--masses", nargs="+", type=float, default=C.MASSES)
    ap.add_argument("--ntoys", type=int, default=C.NTOYS)
    ap.add_argument("--inject", nargs="+", type=float, default=C.INJECT)
    ap.add_argument("--lo", type=float, default=1200.0, help="fit start of both regions")
    ap.add_argument("--out", default="8_flavor_cr", help="results/<channel>/<out>/")
    ap.add_argument("--workers", type=int, default=max(1, min(8, mp.cpu_count() - 1)))
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--data-fit", action="store_true", help="fit the forms to the e-mu CR data and stop")
    ap.add_argument("--data-lo", nargs="+", type=float, default=[1000.0, 1200.0])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    from wrplotter.cli_utils import setup_logging
    setup_logging(args.verbose)
    if args.data_fit:
        data_fit(args.data_lo, C.RESULTS / "flavor_cr")
        return
    from simfit import load_cr
    from spectrum import rebin
    from truth_sets import load_truth
    from util import write_csv
    configs = [c for c in CONFIGS if cfg_name(*c) in args.configs]
    cr10 = load_cr()
    e100, cr100, _ = rebin(*cr10["sum"])
    ctx = mp.get_context("spawn")
    title_of = lambda ch: (f"# Step 8, {ch} resolved SR + e-mu CR: toys from every truth, simultaneous S+B fit from "
                           f"{args.lo:.0f} GeV with every configuration")
    range_text = f"Fit range [{args.lo:.0f}, 8000] GeV in both regions"
    for channel in args.channels:
        out = C.RESULTS / channel / args.out
        (out / "plots").mkdir(parents=True, exist_ok=True)
        if args.summary_only:
            all_rows = T.collect_rows(out)
            write_csv(out / "toys_all.csv", all_rows)
            T.write_summary(channel, all_rows, out / "summary.md", title=title_of(channel), range_text=range_text)
            continue
        for variant in args.variants:
            vdir = C.RESULTS / channel / "3_truths" / variant
            for truth_name in args.truths:
                tpath = vdir / f"truth_{truth_name}"
                if not tpath.with_suffix(".npz").exists():
                    logger.warning("%s %s %s: no truth file, skipped", channel, variant, truth_name)
                    continue
                edges10, parts, _ = load_truth(tpath)
                _, sr, _ = rebin(edges10, parts["sum"], np.maximum(parts["sum"], 0.0))
                _, sr_dy, _ = rebin(edges10, parts["DYJets"], np.maximum(parts["DYJets"], 0.0))
                nondy10 = sum(parts[k] for k in ["tt_tW", "Nonprompt", "Other"])
                _, sr_nondy, _ = rebin(edges10, nondy10, np.maximum(nondy10, 0.0))
                jobs = [{"channel": channel, "variant": variant, "truth_name": truth_name, "edges": e100,
                         "sr": sr, "sr_dy": sr_dy, "sr_nondy": sr_nondy, "cr": cr100, "mass": m, "lo": args.lo,
                         "configs": configs, "ntoys": args.ntoys, "inject": args.inject,
                         "seed": C.SEED + 1000 * int(m) + zlib.crc32(f"{channel}/{variant}/{truth_name}".encode()) % 997}
                        for m in args.masses]
                with ctx.Pool(args.workers) as pool:
                    results = pool.map(run_mass_sim, jobs)
                rows, arrays = [], {}
                for r, a in results:
                    rows += r
                    arrays.update(a)
                (out / variant).mkdir(parents=True, exist_ok=True)
                write_csv(out / variant / f"toys_{truth_name}.csv", rows)
                np.savez_compressed(out / variant / f"nsig_{truth_name}.npz", **arrays)
                if not args.no_plots:
                    T.plot_truth_summary(channel, variant, truth_name, rows, arrays, out / "plots" / variant,
                                         tag=f", SR + CR from {args.lo:.0f} GeV")
                ok = [r for r in rows if r.get("status") == "ok" and r["inject"] == 0.0 and r.get("bias_median") is not None]
                for tt, dy, tfm in configs:
                    cs = [r for r in ok if r["candidate"] == cfg_name(tt, dy, tfm)]
                    if cs:
                        worst = max(cs, key=lambda r: abs(r["bias_median"]))
                        logger.info("%s %s %-14s %-26s max|S_spur| %.2f at %4.0f; >0.3 at %s", channel, variant, truth_name,
                                    cfg_name(tt, dy, tfm), abs(worst["bias_median"]), worst["mass"],
                                    ",".join(f"{r['mass']:.0f}" for r in cs if abs(r["bias_median"]) > C.BIAS_TARGET) or "-")
        all_rows = T.collect_rows(out)
        write_csv(out / "toys_all.csv", all_rows)
        T.write_summary(channel, all_rows, out / "summary.md", title=title_of(channel), range_text=range_text)
        logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
