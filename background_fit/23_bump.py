#!/usr/bin/env python3
"""Where a bump of the pull mean comes from (the bump near 2 TeV that every DY sample shows).

The pull mean of the bias test at one mass is, to 0.03, N_sig / sigma_fit of the S+B fit to the background itself (no
fluctuation), so the question needs no toys. Per channel, on the nine-process MC (LO HT-binned DY), the power x
exponential in the +-5 sigma window from 1200 GeV:

  swap       the three process groups (DY; tt+tW; the seven other processes) replaced one at a time, and all together,
             by a smooth curve (the group's own power x exponential fitted from 1200 GeV): the group whose
             replacement removes the bump carries it; "all smooth" is what the function and the window do on their own
  processes  the contribution of each of the nine processes at the channel's cell (a small process: its bins replaced
             by the smooth shape of its group at its own yield), with the spread that process's MC statistics gives
  diboson    the diboson sample campaign by campaign, and the 2024 one by dataset (WW, WZ, ZZ; from single-dataset
             analyzer runs, rootfiles/.../RunIII2024Summer24/20261115_diboson_<dataset>/, when present)
  swap2024   the pull against the mass with the 2024 diboson sample, the 2024 tt+tW sample, and both, made smooth
  spectrum   the 2024 diboson sample per 100 GeV bin of the fit window against its smooth curve, the signal core shaded
  residuals  MC - smooth curve per 100 GeV bin for each group, with the MC statistical error, the signal core shaded
  mcstat     the MC fluctuated within its statistical errors (Gaussian per 100 GeV bin and group), the fit repeated:
             the spread a bump of this kind has from the simulation's statistics alone
  also       the same cells with the window clipped at 1000 and 800 GeV instead of 1200

Writes results*/<ch>/23_bump/bump_{swap,swap2024,processes_m<mass>,diboson_m<mass>,diboson_spectrum_m<mass>,
residuals_m<mass>,mcstat_m<mass>}.{pdf,png} and results*/BUMP.md.

  cd WR_Plotter && FIT_DATASET=run3 python background_fit/23_bump.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))
import config as C                                                       # noqa: E402

logger = logging.getLogger("23_bump")
CAND = "powexp"
K = 5.0
FLOOR = 1200.0
MASSES = [1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]
GROUPS = {"DY": ["DYJets"], "tt+tW": ["tt_tW"],
          "other": ["WJets", "TTSemiLep", "tWSemiLep", "SingleTop", "Diboson", "ttV", "Triboson"]}
GROUP_OF = {k: g for g, ks in GROUPS.items() for k in ks}
GROUP_LABEL = {"DY": "DY+jets", "tt+tW": r"$t\bar{t}$+tW", "other": "other processes"}
GROUP_COLOR = {"DY": "#5790fc", "tt+tW": "#f89c20", "other": "#e42536"}
FOCUS = {"ee": 2000, "mumu": 1800}                 # the cell each channel's figures are drawn for
NREP = 1000
ERA_SHORT = {"Run3Summer22": "2022", "Run3Summer22EE": "2022EE", "Run3Summer23": "2023", "Run3Summer23BPix": "2023BPix",
             "RunIII2024Summer24": "2024 (+2025)"}
LATE = "RunIII2024Summer24"                        # the campaign that also carries the 2025 luminosity
VV = ("WW", "WZ", "ZZ")
VV_COLOR = {"WW": "#964a8b", "WZ": "#c9a0c4", "ZZ": "#5a2d57"}
DIBOSON_COLOR = "#964a8b"


def asimov(ch, e, truth, m, floor=FLOOR):
    """(N_sig, sigma_fit, lo, hi, mu, sigma) of the S+B fit to the spectrum itself; None when a fit fails."""
    from sbfit import FIT_HI, SBModel, bkg_only, pivot, signal_pars
    from spectrum import centers
    mu, sigma = signal_pars(ch, m)
    lo, hi = max(mu - K * sigma, floor), min(mu + K * sigma, FIT_HI)
    c = centers(e)
    core = (c >= mu - C.SIG_CORE_K * sigma) & (c <= mu + C.SIG_CORE_K * sigma)
    b_core = float(truth[core].sum())
    bounds = (-0.8 * b_core, max(5.0 * b_core + 50.0, 200.0))
    fb = bkg_only(CAND, e, truth, np.maximum(truth, 0.0), lo, hi, {}, m0=pivot(CAND, mu))
    if fb is None:
        return None
    r = SBModel(CAND, mu, sigma, lo, hi).fit(e, truth, fb["params"], 0.0, bounds)
    return None if r is None or r["nsig_err"] <= 0 else (r["nsig"], r["nsig_err"], lo, hi, mu, sigma)


def pull(ch, e, truth, m, floor=FLOOR):
    r = asimov(ch, e, truth, m, floor)
    return float("nan") if r is None else r[0] / r[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--nrep", type=int, default=NREP)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    from wrplotter.cli_utils import setup_logging
    setup_logging(args.verbose)
    import uproot
    import plots as P
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from fits import fit_all
    from spectrum import COMPONENT_COLOR, COMPONENT_LABEL, centers, hist_key, last_populated, load_components, rebin
    L = [f"# Where the pull-mean bump comes from, {C.ERA} {C.REGION_NAME} ({C.LUMI:.1f} fb-1)", "",
         f"N_sig / sigma_fit of the S+B fit to the background itself (no fluctuation): the {CAND} in the +-{K:g} sigma "
         f"window from {FLOOR:.0f} GeV, nine-process MC with the LO HT-binned DY. A group is 'smooth' when its bins from "
         f"{FLOOR:.0f} GeV up are replaced by its own {CAND} fit; its contribution is the pull with everything raw minus "
         "the pull with that group smooth. A single process, campaign or dataset is made smooth by replacing its bins "
         "with the smooth shape of its group at its own yield in the fit window.", ""]
    for ch in args.channels:
        out = C.RESULTS / ch / "23_bump"
        out.mkdir(parents=True, exist_ok=True)
        comps = load_components(ch, "lo_ht", "split9")
        g, single = {}, {}
        for name, members in GROUPS.items():
            e, v, var = rebin(comps[members[0]][0], sum(comps[k][1] for k in members), sum(comps[k][2] for k in members))
            g[name] = (v, var)
            for k in members:
                single[k] = rebin(*comps[k])[1:]
        c = centers(e)
        fitm = c >= FLOOR
        smooth = {}
        for name, (v, var) in g.items():
            Lf, _, _ = fit_all(e, v, var, FLOOR, last_populated(e, v), C.PIVOT, menu=["expo", CAND])
            s = np.array(v, float)
            s[fitm] = Lf[CAND]["model"][fitm]
            smooth[name] = s
        raw = sum(v for v, _ in g.values())
        truths = {"all raw": raw}
        for name in g:
            truths[f"{name} smooth"] = raw - g[name][0] + smooth[name]
        truths["all smooth"] = sum(smooth.values())
        scan = {t: [pull(ch, e, arr, m) for m in MASSES] for t, arr in truths.items()}
        m0 = FOCUS[ch]
        i0 = MASSES.index(m0)
        base = asimov(ch, e, raw, m0)
        p0 = scan["all raw"][i0]
        win = (c >= base[2]) & (c <= base[3])
        rng = np.random.default_rng(20261115)

        def smoothed(v, grp, window):
            """A piece's bins from the floor up replaced by its group's smooth shape at the piece's yield in the window."""
            tot = float(smooth[grp][window].sum())
            repl = np.array(v, float)
            repl[fitm] = smooth[grp][fitm] * (float(v[window].sum()) / tot if tot > 0 else 0.0)
            return repl

        def piece(label, v, var, grp, color):
            """A piece (process, campaign, dataset): its contribution to the pull at m0 and the spread its MC statistics give."""
            d = p0 - pull(ch, e, raw - v + smoothed(v, grp, win), m0)
            vals = [pull(ch, e, np.maximum(raw + rng.normal(0.0, np.sqrt(np.maximum(var, 0.0))), 0.0), m0) for _ in range(200)]
            n, s2 = float(v[win].sum()), float(var[win].sum())
            neff = n * n / s2 if s2 > 0 else 0.0
            return {"label": label, "v": v, "var": var, "grp": grp, "d": d, "spread": float(np.nanstd(vals)), "n": n,
                    "err": float(np.sqrt(s2)), "neff": neff, "color": color}

        L += [f"## {ch}", "", "Pull against the mass:", "", "| background | " + " | ".join(str(m) for m in MASSES) + " |",
              "|---|" + "---|" * len(MASSES)]
        for t, vals in scan.items():
            L.append(f"| {t} | " + " | ".join(f"{x:+.2f}" for x in vals) + " |")
        for floor in (1000.0, 800.0):
            L.append(f"| all raw, floor {floor:.0f} | " + " | ".join(f"{pull(ch, e, raw, m, floor):+.2f}" for m in MASSES) + " |")
        L += ["", f"Contributions at {m0} GeV:", "", "| part | pull | events |", "|---|---|---|",
              f"| all raw | {p0:+.2f} | {base[0]:+.1f} |"]
        for name in g:
            d = p0 - scan[f"{name} smooth"][i0]
            L.append(f"| {name} | {d:+.2f} | {d * base[1]:+.1f} |")
        L.append(f"| all smooth | {scan['all smooth'][i0]:+.2f} | {scan['all smooth'][i0] * base[1]:+.1f} |")

        # ---- every process on its own -----------------------------------------------------------------------------
        procs = []
        for k in ["DYJets", "tt_tW"] + GROUPS["other"]:
            v, var = single[k]
            pc = piece(COMPONENT_LABEL[k], v, var, GROUP_OF[k], COMPONENT_COLOR[k])
            if k in ("DYJets", "tt_tW"):                         # a whole group: its own smooth fit, as in the swap
                pc["d"] = p0 - scan[f"{GROUP_OF[k]} smooth"][i0]
            pc["key"] = k
            procs.append(pc)
        L += ["", f"The other processes one by one at {m0} GeV:", "", "| process | events in the window | contribution to the pull |",
              "|---|---|---|"]
        L += [f"| {pc['key']} | {pc['n']:.1f} +- {pc['err']:.1f} | {pc['d']:+.2f} |" for pc in procs if pc["key"] in GROUPS["other"]]
        L += ["", f"Every process at {m0} GeV:", "", "| process | contribution to the pull | spread from its MC statistics |", "|---|---|---|"]
        L += [f"| {pc['key']} | {pc['d']:+.2f} | {pc['spread']:.2f} |" for pc in procs]

        # ---- the diboson and tt+tW samples campaign by campaign, the 2024 diboson sample by dataset ---------------
        head = ["| campaign | weight | events in the window | effective MC events | events per MC event | contribution to the pull | "
                "spread from its MC statistics |", "|---|---|---|---|---|---|---|"]
        row = lambda pc, w: (f"| {pc['key']} | {w:.2f} | {pc['n']:.1f} +- {pc['err']:.1f} | {pc['neff']:.0f} | "            # noqa: E731
                             f"{pc['n'] / pc['neff'] if pc['neff'] else 0:.2f} | {pc['d']:+.2f} | {pc['spread']:.2f} |")
        eras = {}
        for comp, grp in (("Diboson", "other"), ("tt_tW", "tt+tW")):
            L += ["", f"{comp} campaign by campaign at {m0} GeV:", ""] + head
            eras[comp] = []
            for sub, d_, sd_, w in C.ERAS:
                f_ = uproot.open(C.REPO / "rootfiles" / sub / (d_ if comp in C.GROUPS else sd_) / f"WRAnalyzer_{comp}.root")
                if hist_key(ch) not in f_:
                    continue
                h_ = f_[hist_key(ch)]
                _, v, var = rebin(h_.axes[0].edges(), h_.values() * w, h_.variances() * w * w)
                pc = piece(ERA_SHORT.get(sub.split("/")[-1], sub.split("/")[-1]), v, var, grp, DIBOSON_COLOR if comp == "Diboson" else GROUP_COLOR["tt+tW"])
                pc["key"], pc["weight"] = sub.split("/")[-1], w
                eras[comp].append(pc)
                L.append(row(pc, w))
        w24 = next((w for sub, _d, _sd, w in C.ERAS if sub.endswith(LATE)), None)
        vvp = {t: C.REPO / "rootfiles" / "Run3/2024/RunIII2024Summer24" / f"20261115_diboson_{t}" / "WRAnalyzer_Diboson.root" for t in VV}
        datasets = []
        if w24 and all(q.exists() for q in vvp.values()):
            L += ["", f"The 2024 diboson datasets at {m0} GeV:", ""] + [head[0].replace("campaign", "dataset"), head[1]]
            for t, q in vvp.items():
                f_ = uproot.open(q)
                if hist_key(ch) not in f_:
                    continue
                h_ = f_[hist_key(ch)]
                _, v, var = rebin(h_.axes[0].edges(), h_.values() * w24, h_.variances() * w24 * w24)
                pc = piece(f"2024 {t}", v, var, "other", VV_COLOR[t])
                pc["key"] = t
                datasets.append(pc)
                L.append(row(pc, w24))

        # ---- the pull against the mass with the 2024 samples made smooth -------------------------------------------
        late = {comp: next((pc for pc in eras[comp] if pc["key"] == LATE), None) for comp in eras}
        scan24 = {}
        if all(late.values()):
            variants = {"2024 diboson smooth": [("Diboson", "other")], "2024 tt+tW smooth": [("tt_tW", "tt+tW")],
                        "both smooth": [("Diboson", "other"), ("tt_tW", "tt+tW")]}
            for vname, pieces in variants.items():
                vals = []
                for m in MASSES:
                    r = asimov(ch, e, raw, m)
                    wm = (c >= r[2]) & (c <= r[3])
                    t = np.array(raw, float)
                    for comp, grp in pieces:
                        t = t - late[comp]["v"] + smoothed(late[comp]["v"], grp, wm)
                    vals.append(pull(ch, e, t, m))
                scan24[vname] = vals
            L += ["", "Pull against the mass with the 2024 samples made smooth:", "", "| background | " + " | ".join(str(m) for m in MASSES) + " |",
                  "|---|" + "---|" * len(MASSES), "| all raw | " + " | ".join(f"{x:+.2f}" for x in scan["all raw"]) + " |"]
            L += [f"| {vname} | " + " | ".join(f"{x:+.2f}" for x in vals) + " |" for vname, vals in scan24.items()]

        # ---- the MC statistics: fluctuate the MC within its errors, one group at a time and all together -----------
        spread, reps_all = {}, None
        for which in list(g) + ["all"]:
            vals = []
            for _ in range(args.nrep if which == "all" else max(args.nrep // 4, 100)):
                t = np.zeros_like(raw)
                for name, (v, var) in g.items():
                    t = t + (v + rng.normal(0.0, np.sqrt(np.maximum(var, 0.0))) if which in (name, "all") else v)
                x = pull(ch, e, np.maximum(t, 0.0), m0)
                if np.isfinite(x):
                    vals.append(x)
            spread[which] = float(np.std(vals))
            if which == "all":
                reps_all = np.array(vals)
        L += ["", f"Spread from the MC statistics at {m0} GeV:", "", "| fluctuated | spread of the pull |", "|---|---|"]
        L += [f"| {k} | {s:.2f} |" for k, s in spread.items()]
        L += ["", f"The bump is {abs(p0) / spread['all']:.1f} times the spread from the MC statistics.", ""]

        window_text = f"Power $\\times$ expo\n$\\pm 5\\sigma$ window\nfrom {FLOOR:.0f} GeV"
        ylab = r"$N_{\rm sig}$ / $\sigma_{\rm fit}$, no fluctuation"

        # ---- figure: one group at a time smooth ---------------------------------------------------------------------
        fig, ax = P.scan_figure()
        styles = [("all raw", "k", "-", "o", "raw MC"), ("DY smooth", GROUP_COLOR["DY"], "-", "^", "DY+jets smooth"),
                  ("tt+tW smooth", GROUP_COLOR["tt+tW"], "-", "s", r"$t\bar{t}$+tW smooth"),
                  ("other smooth", GROUP_COLOR["other"], "-", "D", "other processes smooth"),
                  ("all smooth", "0.45", "--", "v", "all smooth")]
        for t, col, ls, mk, lab in styles:
            ax.plot(MASSES, scan[t], color=col, ls=ls, lw=1.6, marker=mk, ms=7, label=lab)
        P.bias_lines(ax)
        handles = [Line2D([], [], color=col, ls=ls, lw=1.6, marker=mk, ms=7, label=lab) for _, col, ls, mk, lab in styles]
        P.finish_scan(ax, channel=ch, extra=window_text, ylabel=ylab, ylim=(-1.0, 2.4), handles=handles, ncol=1)
        P.save(fig, out / "bump_swap")

        # ---- figure: the 2024 diboson and tt+tW samples made smooth -------------------------------------------------
        if scan24:
            fig, ax = P.scan_figure()
            styles = [("all raw", scan["all raw"], "k", "-", "o", "raw MC"),
                      ("d", scan24["2024 diboson smooth"], DIBOSON_COLOR, "-", "D", "2024 diboson smooth"),
                      ("t", scan24["2024 tt+tW smooth"], GROUP_COLOR["tt+tW"], "-", "s", r"2024 $t\bar{t}$+tW smooth"),
                      ("b", scan24["both smooth"], "0.45", "--", "v", "both smooth")]
            for _, vals, col, ls, mk, lab in styles:
                ax.plot(MASSES, vals, color=col, ls=ls, lw=1.6, marker=mk, ms=7)
            P.bias_lines(ax)
            handles = [Line2D([], [], color=col, ls=ls, lw=1.6, marker=mk, ms=7, label=lab) for _, _v, col, ls, mk, lab in styles]
            P.finish_scan(ax, channel=ch, extra=window_text, ylabel=ylab, ylim=(-1.0, 2.4), handles=handles, ncol=1)
            P.save(fig, out / "bump_swap2024")

        # ---- figures: bars of the contributions, with the spread each piece's own MC statistics give -----------------
        def bars(pieces, total_label, name, extra):
            fig, ax = P.scan_figure()
            rows = pieces + [{"label": total_label, "d": p0, "spread": spread["all"], "color": "k"}]
            y = np.arange(len(rows))[::-1]
            ax.barh(y, [r["d"] for r in rows], xerr=[r["spread"] for r in rows], color=[r["color"] for r in rows], height=0.62,
                    error_kw={"ecolor": "0.25", "elinewidth": 1.6, "capsize": 4})
            ax.axvline(0.0, color="k", lw=0.8)
            ax.set_yticks(y, [r["label"] for r in rows])
            ax.tick_params(axis="y", which="minor", left=False, right=False)
            ax.set_xlim(-0.9, 0.9)
            ax.set_ylim(-0.6, len(rows) - 0.4)
            handles = [Line2D([], [], color="0.25", lw=1.6, marker="|", ms=10, label="spread from its MC stat.")]
            P.finish_scan(ax, channel=ch, extra=extra, ylabel="", xlabel=f"Contribution to {ylab.split(',')[0]}",
                          handles=handles, ncol=1)
            P.save(fig, out / name)

        # one short line under the region: a taller label block would reach down over the first bars
        bars(procs, "all processes", f"bump_processes_m{m0}", f"$\\pm 5\\sigma$ window, {m0 / 1000:g} TeV")
        if eras.get("Diboson"):
            bars(eras["Diboson"] + datasets, "all processes", f"bump_diboson_m{m0}", f"Diboson MC, {m0 / 1000:g} TeV")

        # ---- figure: the 2024 diboson sample in the fit window against its smooth curve ------------------------------
        mu, sigma = base[4], base[5]
        sel = (c >= base[2]) & (c <= base[3])
        es = np.append(e[:-1][sel], e[1:][sel][-1])
        if late.get("Diboson"):
            d24 = late["Diboson"]
            fig, ax = P.scan_figure()
            ax.axvspan(mu - C.SIG_CORE_K * sigma, mu + C.SIG_CORE_K * sigma, color="0.85", lw=0, zorder=0)
            handles, bottom = [], np.zeros(int(sel.sum()))
            for pc in (datasets[::-1] if datasets else [dict(d24, key="diboson", color=DIBOSON_COLOR)]):
                ax.stairs(bottom + pc["v"][sel], es, baseline=bottom, fill=True, color=pc["color"], zorder=2)
                bottom = bottom + pc["v"][sel]
                handles.insert(0, Patch(facecolor=pc["color"], lw=0, label=pc["key"]))
            err = np.sqrt(np.maximum(d24["var"][sel], 0.0))
            ax.stairs(d24["v"][sel] + err, es, baseline=np.clip(d24["v"][sel] - err, 0.0, None), hatch="////", facecolor="none",
                      lw=0, edgecolor="k", alpha=0.5, zorder=3)
            sm = smoothed(d24["v"], "other", win)[sel]
            ax.plot(c[sel], sm, color="k", lw=2.0, zorder=4)
            ax.set_xlim(base[2], base[3])
            handles += [Patch(hatch="////", facecolor="none", lw=0, edgecolor="k", alpha=0.5, label="MC stat. error"),
                        Line2D([], [], color="k", lw=2.0, label="smooth curve"),
                        Patch(facecolor="0.85", ec="0.6", lw=0.8, label=rf"signal core, {m0 / 1000:g} TeV")]
            top = float((d24["v"][sel] + err).max())
            P.finish_scan(ax, channel=ch, extra="2024 diboson MC", ylabel=f"Events / {C.BIN_WIDTH:.0f} GeV", xlabel=P.MASS_LABEL,
                          ylim=(0.0, 1.9 * top), handles=handles, ncol=1)
            P.save(fig, out / f"bump_diboson_spectrum_m{m0}")

        # ---- figure: MC - smooth per bin and group, the signal core shaded -------------------------------------------
        fig, ax = P.scan_figure()
        ax.axvspan(mu - C.SIG_CORE_K * sigma, mu + C.SIG_CORE_K * sigma, color="0.85", lw=0, zorder=0)
        top = 0.0
        for j, name in enumerate(g):
            v, var = g[name]
            d, err = (v - smooth[name])[sel], np.sqrt(np.maximum(var, 0.0))[sel]
            ax.errorbar(c[sel] + (j - 1) * 22.0, d, yerr=err, fmt="o", ms=6, lw=1.4, color=GROUP_COLOR[name], label=GROUP_LABEL[name])
            top = max(top, float(np.abs(d).max() + err.max()))
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xlim(base[2], base[3])
        handles = [Line2D([], [], color=GROUP_COLOR[n], marker="o", ms=6, lw=1.4, label=GROUP_LABEL[n]) for n in g]
        handles.append(Patch(facecolor="0.85", ec="0.6", lw=0.8, label=rf"signal core, {m0 / 1000:g} TeV"))
        P.finish_scan(ax, channel=ch, extra=f"fit window of {m0 / 1000:g} TeV", ylabel=f"MC $-$ smooth fit [events / {C.BIN_WIDTH:.0f} GeV]",
                      xlabel=P.MASS_LABEL, ylim=(-1.15 * top, 2.6 * top), handles=handles, ncol=1)
        P.save(fig, out / f"bump_residuals_m{m0}")

        # ---- figure: the pull under the MC's own statistics ----------------------------------------------------------
        fig, ax = P.scan_figure()
        lo_x, hi_x = min(-1.0, reps_all.min() - 0.1), max(1.6, reps_all.max() + 0.1)
        ax.hist(reps_all, bins=np.linspace(lo_x, hi_x, 53), histtype="stepfilled", color="#c9d6f5", ec="#5790fc", lw=1.4)
        ax.axvline(p0, ymax=0.6, color="k", lw=2.0)                          # short of the legend and the label block
        ax.axvline(0.0, ymax=0.6, color="0.4", lw=1.4, ls="--")
        handles = [Patch(facecolor="#c9d6f5", ec="#5790fc", lw=1.4, label="MC varied within its stat. error"),
                   Line2D([], [], color="k", lw=2.0, label=f"the MC as it is: {p0:+.2f}"),
                   Line2D([], [], color="0.4", lw=1.4, ls="--", label="no bias"),
                   Line2D([], [], color="none", label=f"spread {spread['all']:.2f}")]
        P.finish_scan(ax, channel=ch, extra=f"Power $\\times$ expo\n$\\pm 5\\sigma$ window\n{m0 / 1000:g} TeV",
                      ylabel="MC replicas", xlabel=ylab, handles=handles, ncol=1)
        P.save(fig, out / f"bump_mcstat_m{m0}")
        logger.info("%s: wrote %s", ch, out)
    (C.RESULTS / "BUMP.md").write_text("\n".join(L) + "\n")
    logger.info("wrote %s", C.RESULTS / "BUMP.md")


if __name__ == "__main__":
    main()
