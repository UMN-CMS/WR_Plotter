#!/usr/bin/env python3
"""MC-ONLY comparison cards: the Run 2 way. Every background is a fixed
simulation template in the window (DYJets, tt_tW, Nonprompt, Other, at the
card binning), the signal the same Gaussian as the data-driven cards, and the
systematics the analyzer provides on every template:

  * per-bin MC statistics            -> autoMCStats (Barlow-Beeston-lite)
  * each weight/shape variation      -> one log-normal per source on every
                                        background (window yield up/down),
                                        correlated across processes
  * PDF / alpha_s (--systs pdf)      -> one log-normal each, from the analyzer's
                                        PDF members summed over the window
                                        BEFORE the Hessian quadrature (exact for
                                        the window yield; wrplotter/pdf.py)
  * the signal                       -> the same lnN groups as the data-driven
                                        card, plus the peak shifts as `shape`
                                        templates (JES, JER, e scale, e smear)
  * luminosity                       -> one nuisance on everything

Plain TH1 inputs (no RooFit workspace), built locally; combine then runs
AsymptoticLimits exactly as for the data-driven cards.

  python make_mc_cards.py --inputs inputs_2024_run2sig --out cards_2024_mconly \\
      --signal-systs inputs_2024_run2sig/signal_systs.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import ROOT

HERE = Path(__file__).resolve().parent
SIGFIT = HERE.parents[1]
sys.path.insert(0, str(SIGFIT.parent))
sys.path.insert(0, str(SIGFIT / "shared"))
from wrplotter.cli_utils import setup_logging                           # noqa: E402
from wrplotter.config import load_systematics                           # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root               # noqa: E402
from syst_band import _sum_hist, variation_keys, PLACEHOLDER            # noqa: E402
from card_sb_fit import rebin_snap                                      # noqa: E402
from wrplotter.pdf import sample_members, total_shifts                  # noqa: E402

ROOT.gROOT.SetBatch(True)
logger = logging.getLogger("make_mc_cards")
REGION = "wr_ee_resolved_sr"
PROCS = ["DYJets", "tt_tW", "Nonprompt", "Other"]
GROUPS = {
    "lumi": ["lumi"], "pileup": ["pileup"],
    "ele_sf": ["electronrecosf", "electronidsf", "electrontrigsf", "looseelectronrecosf"],
    "jes": ["jes"], "jer": ["jer"], "elescale": ["elescale"], "elesmear": ["elesmear"],
    "scale": ["scale"], "ps": ["psisr", "psfsr"],
    "pdf": ["pdf"], "alphas": ["alphas"],     # from the PDF members (pdf_window_rel)
}
SHAPE_GROUPS = ("jes", "jer", "elescale", "elesmear")


def th1(name, edges, vals, errs=None):
    h = ROOT.TH1D(name, "", len(vals), np.ascontiguousarray(edges, np.float64))
    for i, v in enumerate(vals, start=1):
        h.SetBinContent(i, float(v))
        if errs is not None:
            h.SetBinError(i, float(errs[i - 1]))
    return h


def gauss_template(edges, m_c, sigma):
    """Per-bin integral of the Gaussian, normalized to 1 over the window."""
    from scipy.special import erf
    lo, hi = edges[:-1], edges[1:]
    cdf = lambda x: 0.5 * (1.0 + erf((x - m_c) / (sigma * math.sqrt(2.0))))
    s = cdf(hi) - cdf(lo)
    return s / s.sum()


def window(vals, edges10, v):
    e, w = rebin_snap(edges10, vals, v["bw"], v["fit_lo"], v["fit_hi"])
    return e, np.where(np.isfinite(w) & (w > 0), w, 0.0)


def pdf_window_rel(members, edges10, v, rate):
    """{"pdf": rel, "alphas": rel} of the window yield from native-binning PDF members."""
    if not members or rate <= 0:
        return {}
    windowed = {}
    for lhaid, (arr, layout) in members.items():
        rows = np.stack([window(row, edges10, v)[1] for row in arr])
        windowed[lhaid] = (rows.sum(axis=1, keepdims=True), layout)
    shifts = total_shifts(windowed)
    out = {"pdf": float(shifts["PDF"][0]) / rate}
    if "AlphaS" in shifts:
        out["alphas"] = abs(float(shifts["AlphaS"][0])) / rate
    return out


def sig_interp(sig_systs, mass, group, key):
    pts = sorted((int(m), e) for m, e in sig_systs["masses"].items())
    ms = [m for m, _ in pts]
    ys = [e["groups"].get(group, {}).get(key, 0.0) for _, e in pts]
    if mass <= ms[0]:
        return ys[0]
    if mass >= ms[-1]:
        return ys[-1]
    return float(np.interp(mass, ms, ys))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=HERE / "inputs_2024_run2sig")
    p.add_argument("--era", default="RunIII2024Summer24")
    p.add_argument("--bkg-dir", default="20261004_full_systs")
    p.add_argument("--signal-systs", type=Path, default=None)
    p.add_argument("--variant", default="k5_bw50")
    p.add_argument("--out", type=Path, default=HERE / "cards_2024_mconly")
    p.add_argument("--masses", type=int, nargs="*", default=None)
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    tag = "ee_resolved"
    with open(args.inputs / f"{tag}.json") as fh:
        meta = json.load(fh)
    sig_systs = json.load(open(args.signal_systs)) if args.signal_systs else None
    dirs, _ = input_dirs_for_era(args.era, repo_root(), args.bkg_dir)
    sources = [s for s in load_systematics() if s not in PLACEHOLDER]
    for extra in ("electronidsf", "electrontrigsf", "looseelectronrecosf"):
        if extra not in sources:
            sources.append(extra)
    nom_key = f"{REGION}/mass_fourobject_{REGION}"

    # native histograms once: nominal (+sumw2) and every variation per process
    native, native_pdf = {}, {}
    for proc in PROCS:
        files = [Path(d) / f"WRAnalyzer_{proc}.root" for d in dirs]
        native_pdf[proc] = sample_members([f for f in files if f.exists()], REGION, "mass_fourobject",
                                          lambda h: h.values())
        r = _sum_hist(dirs, proc, nom_key)
        edges10, vals, w2 = r
        var = {}
        for src in sources:
            pair = {}
            for d, key in variation_keys(src, REGION, "mass_fourobject"):
                rr = _sum_hist(dirs, proc, key)
                if rr is not None:
                    pair[d] = rr[1]
            if pair:
                var[src] = pair
        native[proc] = (edges10, vals, w2, var)

    out_dir = args.out / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    masses = sorted(int(m) for m in meta["masses"])
    if args.masses:
        masses = [m for m in masses if m in set(args.masses)]
    for mass in masses:
        m = meta["masses"][str(mass)]
        v = m["vars"][args.variant]
        m_c, sigma = m["m_c"], m["sigma"]
        shapes = out_dir / f"shapes_mc_m{mass}.root"
        fout = ROOT.TFile(str(shapes), "RECREATE")
        hists, rates, lnn = [], {}, {}      # lnn[group][proc] = max rel
        total = None
        edges = None
        for proc in PROCS:
            edges10, vals, w2, var = native[proc]
            e, w = window(vals, edges10, v)
            _, w2w = window(w2, edges10, v)
            edges = e
            rates[proc] = float(w.sum())
            total = w.copy() if total is None else total + w
            h = th1(proc, e, w, np.sqrt(w2w))
            h.Write()
            hists.append(h)
            for g, members in GROUPS.items():
                acc = 0.0
                for src in members:
                    if src not in var or rates[proc] <= 0:
                        continue
                    devs = []
                    for d, vv in var[src].items():
                        _, wv = window(vv, edges10, v)
                        devs.append(abs(wv.sum() / rates[proc] - 1.0))
                    acc += max(devs) ** 2 if devs else 0.0
                if acc > 0:
                    lnn.setdefault(g, {})[proc] = math.sqrt(acc)
            for g, x in pdf_window_rel(native_pdf[proc], edges10, v, rates[proc]).items():
                if x > 0:
                    lnn.setdefault(g, {})[proc] = x
        th1("data_obs", edges, total).Write()
        # signal: unit-normalized Gaussian template, rate = lumi*eff
        # TH1 cards: the template integral IS the rate (rate column = -1),
        # so the signal template carries lumi*eff (events per fb of sigma*BR)
        s_nom = gauss_template(edges, m_c, sigma) * v["rate_per_fb"]
        th1("sig", edges, s_nom).Write()
        shape_lines = []
        sig_lnn = {}
        if sig_systs:
            for g in GROUPS:
                rel = sig_interp(sig_systs, mass, g, "rate_rel")
                if g in SHAPE_GROUPS:
                    mu_rel = sig_interp(sig_systs, mass, g, "mu_rel")
                    sg_rel = sig_interp(sig_systs, mass, g, "sigma_rel")
                    if mu_rel > 0 or sg_rel > 0:
                        th1(f"sig_{g}_shapeUp", edges,
                            gauss_template(edges, m_c * (1 + mu_rel), sigma * (1 + sg_rel)) * v["rate_per_fb"]).Write()
                        th1(f"sig_{g}_shapeDown", edges,
                            gauss_template(edges, m_c * (1 - mu_rel), sigma * (1 - sg_rel)) * v["rate_per_fb"]).Write()
                        shape_lines.append(f"{g}_shape shape 1 - - - -")
                if rel > 1e-4:
                    sig_lnn[g] = rel
        fout.Close()

        # processes with no simulated events in the window carry nothing:
        # leave them out of the card (combine refuses empty templates)
        procs_here = [pr for pr in PROCS if rates[pr] > 0.0]
        cols = ["sig"] + procs_here
        lines = [
            f"# MC-only comparison card m_WR={mass}: fixed simulation templates, window [{edges[0]:.0f}, {edges[-1]:.0f}] x {v['bw']:.0f} GeV",
            f"# rate(sig) = lumi*eff = {v['rate_per_fb']:.4f} ev/fb -> r is sigma x BR in fb; backgrounds in events",
            "imax 1", "jmax *", "kmax *", "----------------",
            f"shapes * win {shapes.name} $PROCESS $PROCESS_$SYSTEMATIC",
            "----------------", "bin win", "observation -1", "----------------",
            "bin      " + " ".join(["win"] * len(cols)),
            "process  " + " ".join(cols),
            "process  " + " ".join(str(i) for i in range(len(cols))),
            "rate     " + " ".join(["-1"] * len(cols)),
            f"# template integrals: sig {v['rate_per_fb']:.6g} ev/fb, " + ", ".join(f"{pr} {rates[pr]:.6g}" for pr in procs_here),
            "----------------",
        ]
        for g in GROUPS:
            vals_line = []
            any_ = False
            for c in cols:
                if c == "sig":
                    x = sig_lnn.get(g, 0.0)
                else:
                    x = lnn.get(g, {}).get(c, 0.0)
                if x > 1e-4:
                    vals_line.append(f"{1 + x:.4f}")
                    any_ = True
                else:
                    vals_line.append("-")
            if any_:
                lines.append(f"{g} lnN " + " ".join(vals_line))
        lines = [ln.replace("shape 1 - - - -", "shape 1 " + " ".join(["-"] * len(procs_here))) if ln.endswith("shape 1 - - - -") else ln for ln in lines]
        shape_lines = [ln.replace("shape 1 - - - -", "shape 1 " + " ".join(["-"] * len(procs_here))) for ln in shape_lines]
        lines += shape_lines
        lines.append("win autoMCStats 0 0 1")
        card = out_dir / f"card_mc_m{mass}.txt"
        card.write_text("\n".join(lines) + "\n")
        manifest.append(f"{card}\t{mass}\tmc")
        logger.info("m=%d: window [%.0f, %.0f] bkg %s  lnN %s", mass, edges[0], edges[-1],
                    {pr: round(rates[pr], 1) for pr in PROCS},
                    {g: {pr: round(100 * x, 1) for pr, x in d.items()} for g, d in lnn.items()})
    (out_dir / "manifest_mc.txt").write_text("\n".join(manifest) + "\n")
    logger.info("%d cards -> %s", len(manifest), out_dir)


if __name__ == "__main__":
    main()
