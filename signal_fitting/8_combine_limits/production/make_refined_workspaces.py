#!/usr/bin/env python3
"""Stage 10.8 -- the refined run2 (2018 ee) expected limit, combine-only.

Applies the Stage-10 findings to the run2 limit using NOTHING but combine
cards (no homemade toys):

  * float regime (1400-3200): since Jul 2026 the Stage-10.9 OPTIMIZED card
    (k5 window, 50 GeV bins, expo slope param-constrained to the anchor fit,
    signal mu AND sigma param-constrained at 0.3*sigma0 -- the 'both030'
    variant): ~40% better expected limit than the Stage-9 k3/floating card,
    validated by FitDiagnostics toys (null spurious < 2% of the medians,
    injection recovered to 1-5%). Windows/efficiencies come from the 10.9
    inputs (../optimization/inputs); 1400-1800 adopt the
    winner configuration by extrapolation (the scan covered 2000-3200; their
    k5 windows clamp against the 800 GeV floor, handled in the inputs).
    AsymptoticLimits is valid here (B_window 37-1036 at k5).
  * anchored regime, low edge (1000-1200): the window is left-clamped at the
    800 GeV selection threshold -> no left sideband, background norm/slope
    collinear with the signal (10.1 finding: structurally BROKEN). The card
    fixes the background to an ANCHOR fit of the summed MC over the trusted
    spectrum, transported into the window; only r floats. B_window ~ 900 ->
    AsymptoticLimits fine.
  * anchored regime, sparse tail (>= 3400, B_window < 7): same anchored
    background (kills the free-norm runaway), but AsymptoticLimits
    under-covers at a few events (10.6: 2.3 vs HybridNew 3.25 at 4600) ->
    run_refined.sh runs HybridNew toys for the expected band there.

Anchor fits are done IN THIS SCRIPT (container ROOT, binned Poisson-ML via
TH1::Fit "L") on the run2 background MC -- no dependency on the run3-era
Stage-10.4 outputs. Members:

  central   expo   on [1000, 3500]  (the trusted spectrum)  -> the quoted card
  tail      expo   on [1000, 6000]  (tail-anchored)          |  model-spread
  expo2     expo2  on [1000, 3500]                           |  variants
  powexp    powexp on [1000, 3500]                           |  (asymptotic)

Signal rate = lumi x eff (from the Stage-9 run2 inputs), so r == sigma x
BR(eeqq') in fb -- directly comparable to the Stage-9 baseline and the 2018
reference plot.

Run inside the combine container (see run_refined.sh):
  python3 make_refined_workspaces.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

HERE = Path(__file__).resolve().parent                  # production
STAGE9 = HERE.parent / "baseline"
OPT_INPUTS = HERE.parent / "optimization" / "inputs"

PIVOT = 2000.0   # anchor-fit pivot [GeV]; slopes transport, windows re-pivot
FLOAT_VAR = "k5_bw50"    # the Stage-10.9 winner window/binning
SIG_ALPHA = 0.30         # mu/sigma prior width (x sigma0), the 10.2/10.9 choice
# Shape priors only where the toys validate them: at >= 2800 the two shape
# nuisances collapse FitDiagnostics toy convergence (84/51/32% at 2800/3000/
# 3200 vs 99/88/66% fixed) and drift the survivor median, while COSTING limit
# -- so those float cards keep the fixed shape (10.9 toy re-validation).
SIG_PRIOR_MAX = 2600.0

# member -> (TF1 formula at the anchor pivot, [par names], seeds, {i: limits})
# powexp seeds: near the pivot log f ~ lnA + (p/2 + c)*t, so p/2 + c ~ b_expo
# (~-2.9/TeV); both slope-like params are constrained <= 0 as in bkg_fit_lib.
MEMBERS = {
    "central": ("[0]*exp([1]*((x-{p})/1000.))", ["A", "b"],
                [None, -3.0], {1: (-100.0, 0.0)}),
    "tail":    ("[0]*exp([1]*((x-{p})/1000.))", ["A", "b"],
                [None, -3.0], {1: (-100.0, 0.0)}),
    "expo2":   ("[0]*exp([1]*((x-{p})/1000.)+[2]*pow((x-{p})/1000.,2))",
                ["A", "b", "c"], [None, -3.0, 0.0], {1: (-100.0, 0.0)}),
    "powexp":  ("[0]*pow(x/{p},[1])*exp([2]*((x-{p})/1000.))",
                ["A", "p", "c"], [None, -2.0, -1.9],
                {1: (-100.0, 0.0), 2: (-100.0, 0.0)}),
}
MEMBER_RANGE = {"central": (1000.0, 3500.0), "tail": (1000.0, 6000.0),
                "expo2": (1000.0, 3500.0), "powexp": (1000.0, 3500.0)}

# shape-only RooGenericPdf per member (norm is the separate _norm variable);
# @0 = mass, @1.. = shape params (member 'A' dropped)
SHAPE = {
    "central": ("exp(@1*((@0-{p})/1000.))", ["b"]),
    "tail":    ("exp(@1*((@0-{p})/1000.))", ["b"]),
    "expo2":   ("exp(@1*((@0-{p})/1000.)+@2*pow((@0-{p})/1000.,2))", ["b", "c"]),
    "powexp":  ("pow(@0/{p},@1)*exp(@2*((@0-{p})/1000.))", ["p", "c"]),
}

CARD = """\
# Stage 10.8 refined run2 card  {tag}  m_WR={mass}  variant={variant}
# regime={regime}  window [{slo:.0f}, {shi:.0f}]  m_c={m_c}  sigma={sigma}
# signal rate = lumi*eff = {rate:.4f} ev/fb -> r is sigma x BR in fb
imax 1
jmax 1
kmax *
----------------
shapes sig       win  {ws} w:sig_pdf
shapes bkg       win  {ws} w:bkg_pdf
shapes data_obs  win  {ws} w:data_obs
----------------
bin          win
observation  -1
----------------
bin      win   win
process  sig   bkg
process  0     1
rate     {rate:.6g}  1
----------------
{extra}
"""


def snap_range(h, lo, hi):
    ax = h.GetXaxis()
    i0 = next(i for i in range(1, h.GetNbinsX() + 1) if ax.GetBinCenter(i) >= lo)
    i1 = next(i for i in range(h.GetNbinsX(), 0, -1) if ax.GetBinCenter(i) <= hi)
    return i0, i1


def fit_anchor(h_full, member):
    """Binned Poisson-ML anchor fit of one member over its trusted range."""
    formula, names, seeds, limits = MEMBERS[member]
    lo, hi = MEMBER_RANGE[member]
    f = ROOT.TF1(f"anchor_{member}", formula.format(p=f"{PIVOT:.10g}"), lo, hi)
    for i, s in enumerate(seeds):
        if s is None:                           # amplitude: seed from the MC
            s = max(h_full.GetBinContent(h_full.FindBin(PIVOT)), 0.1)
        f.SetParameter(i, s)
    for i, (plo, phi) in limits.items():
        f.SetParLimits(i, plo, phi)
    r = h_full.Fit(f, "LRSQN")                  # binned Poisson likelihood
    ok = (int(r) == 0)
    pars = [f.GetParameter(i) for i in range(len(seeds))]
    errs = [f.GetParError(i) for i in range(len(seeds))]
    chi2, ndf = f.GetChisquare(), f.GetNDF()
    print(f"  anchor {member:<7} [{lo:.0f},{hi:.0f}]: "
          + " ".join(f"{n}={v:.4g}" for n, v in zip(names, pars))
          + f"  chi2/ndf={chi2/max(ndf,1):.2f}  status={'OK' if ok else int(r)}")
    return {"member": member, "params": pars, "errors": errs, "ok": ok,
            "chi2": chi2, "ndf": ndf, "range": [lo, hi]}


def member_window_norm(anchor, h_full, i0, i1):
    """Sum of the member function over the window bin centres = B_env."""
    formula = MEMBERS[anchor["member"]][0]
    f = ROOT.TF1("tmp_env", formula.format(p=f"{PIVOT:.10g}"), 0.0, 8000.0)
    for i, v in enumerate(anchor["params"]):
        f.SetParameter(i, v)
    ax = h_full.GetXaxis()
    return sum(f.Eval(ax.GetBinCenter(i)) for i in range(i0, i1 + 1))


def build_card(h_full, m, variant, regime, out_dir, tag, *, anchor=None,
               rmax=100.0):
    mass_i = int(float(m["mass"]))
    i0, i1 = snap_range(h_full, m["fit_lo"], m["fit_hi"])
    slo, shi = (h_full.GetXaxis().GetBinLowEdge(i0),
                h_full.GetXaxis().GetBinUpEdge(i1))
    nbins = i1 - i0 + 1
    h_win = ROOT.TH1D("h_win", "", nbins, slo, shi)
    for i in range(nbins):
        h_win.SetBinContent(i + 1, h_full.GetBinContent(i0 + i))
    nobs = h_win.Integral()

    x = ROOT.RooRealVar("mass", "m [GeV]", slo, shi)
    x.setBins(nbins)
    data = ROOT.RooDataHist("data_obs", "", ROOT.RooArgList(x), h_win)

    if variant == "float":
        # the Stage-10.9 optimized float card: slope Gaussian-constrained to
        # the central anchor fit (norm stays locally measured)
        b_hat, b_err = anchor["params"][1], anchor["errors"][1]
        par = ROOT.RooRealVar("b_expo", "b_expo", b_hat, -100.0, 0.0)
        bkg = ROOT.RooGenericPdf(
            "bkg_pdf", "exp(@1*((@0-{0:.10g})/1000.))".format(float(m["m_c"])),
            ROOT.RooArgList(x, par))
        norm = ROOT.RooRealVar("bkg_pdf_norm", "", nobs, 0.0,
                               max(10.0 * nobs, 50.0))
        pars = [par]
        extra = (f"b_expo param {b_hat:.5g} {b_err:.5g}\n"
                 "bkg_pdf_norm flatParam")
        b_env = nobs
    else:                                        # anchored member card
        formula, names = SHAPE[anchor["member"]]
        pars = []
        for nm, v in zip(names, anchor["params"][1:]):
            pv = ROOT.RooRealVar(f"{nm}_anch", f"{nm}_anch", v)
            pv.setConstant(True)
            pars.append(pv)
        bkg = ROOT.RooGenericPdf(
            "bkg_pdf", formula.format(p=f"{PIVOT:.10g}"),
            ROOT.RooArgList(x, *pars))
        b_env = member_window_norm(anchor, h_full, i0, i1)
        norm = ROOT.RooRealVar("bkg_pdf_norm", "", b_env)
        norm.setConstant(True)
        extra = (f"# background anchored: member={anchor['member']} "
                 f"B_env={b_env:.3f} (window MC {nobs:.3f})")

    m_c, s0 = float(m["m_c"]), float(m["sigma"])
    if variant == "float" and float(m["mass"]) <= SIG_PRIOR_MAX:
        # 10.9 'both030': mu and sigma float with 0.3*sigma0 Gaussian priors
        # (nearly free once the slope is constrained: +3.5% vs fixed)
        mu = ROOT.RooRealVar("mu_sig", "", m_c, m_c - 1.5 * s0, m_c + 1.5 * s0)
        sg = ROOT.RooRealVar("sigma_sig", "", s0, 0.5 * s0, 2.5 * s0)
        extra += (f"\nmu_sig param {m_c:.5g} {SIG_ALPHA * s0:.5g}"
                  f"\nsigma_sig param {s0:.5g} {SIG_ALPHA * s0:.5g}")
    else:
        # fixed shape: anchored/HybridNew cards (untested + toy cost at
        # sparse masses) and float cards above SIG_PRIOR_MAX (toy fragility)
        mu = ROOT.RooRealVar("mu_sig", "", m_c)
        sg = ROOT.RooRealVar("sigma_sig", "", s0)
        mu.setConstant(True)
        sg.setConstant(True)
    sig = ROOT.RooGaussian("sig_pdf", "", x, mu, sg)

    w = ROOT.RooWorkspace("w", "w")
    for obj in (data, bkg, norm, sig):
        getattr(w, "import")(obj)

    ws_name = f"ws_{variant}_m{mass_i}.root"
    w.writeToFile(str(out_dir / ws_name))
    card = out_dir / f"card_{variant}_m{mass_i}.txt"
    card.write_text(CARD.format(
        tag=tag, mass=mass_i, variant=variant, regime=regime, ws=ws_name,
        slo=slo, shi=shi, m_c=m["m_c"], sigma=m["sigma"],
        rate=float(m["rate_per_fb"]), extra=extra))
    print(f"  {card.name}: [{slo:.0f},{shi:.0f}] B_MC={nobs:.2f}"
          + (f" B_env={b_env:.2f}" if variant != "float" else "")
          + f" rate={float(m['rate_per_fb']):.2f} rMax={rmax:g}")
    return card, mass_i, b_env


def load_rmax(table, fn="expo", scale=4.0, floor=5.0):
    """{mass: rMax in fb} = 4x the Stage-9 run2 +2sigma edge."""
    out = {}
    if not table.exists():
        return out
    with open(table, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("function") != fn:
                continue
            try:
                out[int(float(r["mWR"]))] = max(
                    scale * float(r["comb_fb_p2s"]), floor)
            except (TypeError, ValueError):
                pass
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--channel", default="ee")
    p.add_argument("--topology", default="resolved")
    p.add_argument("--float-min", type=float, default=1400.0)
    p.add_argument("--float-max", type=float, default=3200.0)
    p.add_argument("--input-dir", type=Path,
                   default=STAGE9 / "run2" / "inputs")
    p.add_argument("--rmax-table", type=Path,
                   default=STAGE9 / "run2" / "combine_limit_table_ee_resolved.csv")
    p.add_argument("--output-dir", type=Path, default=HERE / "cards")
    args = p.parse_args()
    tag = f"{args.channel}_{args.topology}"

    with open(args.input_dir / f"{tag}.json") as fh:
        meta = json.load(fh)["masses"]
    fin = ROOT.TFile(str(args.input_dir / f"{tag}.root"))
    h_full = fin.Get("data_obs_full")

    # Stage-10.9 optimized inputs for the float regime (k5 windows, 50 GeV)
    with open(OPT_INPUTS / f"{tag}.json") as fh:
        opt = json.load(fh)["masses"]
    fin_opt = ROOT.TFile(str(OPT_INPUTS / f"{tag}.root"))
    h50 = fin_opt.Get("bkg_native").Rebin(5, "h50")

    print("anchor fits (binned Poisson-ML on the summed run2 background MC):")
    anchors = {mb: fit_anchor(h_full, mb) for mb in MEMBERS}

    rmax = load_rmax(args.rmax_table)
    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for key, m in sorted(meta.items(), key=lambda kv: float(kv[0])):
        mass = float(key)
        m = dict(m, mass=key)
        rm = rmax.get(int(mass), 100.0)
        if args.float_min <= mass <= args.float_max:
            o = opt[key]
            v = o["vars"][FLOAT_VAR]
            mf = {"mass": key, "m_c": o["m_c"], "sigma": o["sigma"],
                  "fit_lo": v["fit_lo"], "fit_hi": v["fit_hi"],
                  "rate_per_fb": v["rate_per_fb"], "eff": v["eff"],
                  "signal_tag": o["signal_tag"]}
            card, mi, b_env = build_card(h50, mf, "float", "float",
                                         out_dir, tag,
                                         anchor=anchors["central"], rmax=rm)
            manifest.append(f"{card}\t{mi}\tfloat\tfloat\t{rm:g}")
        else:
            regime = "anch_low" if mass < args.float_min else "anch_sparse"
            for mb in MEMBERS:
                variant = "anchored" if mb == "central" else f"anch_{mb}"
                card, mi, b_env = build_card(h_full, m, variant, regime,
                                             out_dir, tag,
                                             anchor=anchors[mb], rmax=rm)
                manifest.append(f"{card}\t{mi}\t{variant}\t{regime}\t{rm:g}")
    (out_dir / "manifest.txt").write_text("\n".join(manifest) + "\n")
    with open(out_dir / "anchors.json", "w") as fh:
        json.dump(anchors, fh, indent=1)
    print(f"{len(manifest)} cards -> {out_dir}/manifest.txt")


if __name__ == "__main__":
    main()
