#!/usr/bin/env python3
"""Stage 10.9, step 2 -- build the variant workspaces (container PyROOT).

Card matrix per mass (2000-3200), all with the fixed Gaussian signal and the
MC-Asimov observation:

  window x binning   from the prepare_opt inputs (k3/k4/k5/a53 x 100/50/20)
  background         float:   expo, slope+norm flatParam        (baseline)
                     bconstr: slope Gaussian-constrained to the anchor fit
                              (`b_expo param b_hat err`), norm flatParam
                     bfixed:  slope constant at the anchor, norm flatParam
                     anch:    slope AND norm constant (knowledge bound)

The anchor (expo, [1000, 3500], pivot 2000, binned Poisson-ML on the 100 GeV
rebin) is refit here and its slope error is what the bconstr card uses.

Scanned combinations (kept to the informative axes rather than the full
cube): every window at bw100/float; every binning at k3/float; the four
background modes at (k3, bw100); and the combo candidates
(k5, bw50, {float,bconstr}) and (a53, bw50, bconstr).

  python3 make_opt_workspaces.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

HERE = Path(__file__).resolve().parent

CARD = """\
# 10.9 opt card  m_WR={mass}  {label}
# window [{slo:.0f}, {shi:.0f}] bw={bw:.0f}  m_c={m_c}  sigma={sigma}
# rate = lumi*eff = {rate:.4f} ev/fb -> r is sigma x BR in fb
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

# (window, binning, bkg-mode[, sig-mode]) combinations to build.
# sig-mode (default "fix"): the signal Gaussian treatment --
#   fix      mu and sigma constant (the official/10.6 recommendation)
#   muAAA    mu floats in m_c +- 1.5 sigma0 with `mu_sig param m_c A*sigma0`
#   sigAAA   sigma floats in [0.5, 2.5] sigma0 with `sigma_sig param s0 A*s0`
#   bothAAA  both of the above                       (AAA = 100*alpha, e.g.
#            sig030 = alpha_sigma 0.30, the Stage-10.2 toy-study winner)
COMBOS = (
    [("k3", 100, "float"), ("k4", 100, "float"), ("k5", 100, "float"),
     ("a53", 100, "float"),
     ("k3", 50, "float"), ("k3", 20, "float"),
     ("k3", 100, "bconstr"), ("k3", 100, "bfixed"), ("k3", 100, "anch"),
     ("k5", 50, "float"), ("k5", 50, "bconstr"), ("a53", 50, "bconstr"),
     # signal-shape retest on the 10.9 winner (and one on the baseline,
     # the run2 analogue of the 10.6 prior-vs-fixed measurement)
     ("k5", 50, "bconstr", "sig030"), ("k5", 50, "bconstr", "sig015"),
     ("k5", 50, "bconstr", "mu030"), ("k5", 50, "bconstr", "mu010"),
     ("k5", 50, "bconstr", "both030"),
     ("k3", 100, "float", "sig030")]
)


def fit_anchor(h100):
    f = ROOT.TF1("anchor", "[0]*exp([1]*((x-2000.)/1000.))", 1000.0, 3500.0)
    f.SetParameter(0, max(h100.GetBinContent(h100.FindBin(2000.0)), 0.1))
    f.SetParameter(1, -3.0)
    f.SetParLimits(1, -100.0, 0.0)
    h100.Fit(f, "LRSQN")
    a, b = f.GetParameter(0), f.GetParameter(1)
    berr = f.GetParError(1)
    print(f"anchor: A={a:.4g}  b={b:.4g} +- {berr:.4g}")
    return a, b, berr


def snap(h, lo, hi):
    ax = h.GetXaxis()
    i0 = next(i for i in range(1, h.GetNbinsX() + 1) if ax.GetBinCenter(i) >= lo)
    i1 = next(i for i in range(h.GetNbinsX(), 0, -1) if ax.GetBinCenter(i) <= hi)
    return i0, i1


def _sig_pars(sigmode, m_c, sigma):
    """(mu RooRealVar, sigma RooRealVar, extra card lines) per sig-mode."""
    lines = []
    if sigmode.startswith(("mu", "both")):
        alpha = int(sigmode[-3:]) / 100.0
        mu = ROOT.RooRealVar("mu_sig", "", m_c, m_c - 1.5 * sigma,
                             m_c + 1.5 * sigma)
        lines.append(f"mu_sig param {m_c:.5g} {alpha * sigma:.5g}")
    else:
        mu = ROOT.RooRealVar("mu_sig", "", m_c)
        mu.setConstant(True)
    if sigmode.startswith(("sig", "both")):
        alpha = int(sigmode[-3:]) / 100.0
        sg = ROOT.RooRealVar("sigma_sig", "", sigma, 0.5 * sigma, 2.5 * sigma)
        lines.append(f"sigma_sig param {sigma:.5g} {alpha * sigma:.5g}")
    else:
        sg = ROOT.RooRealVar("sigma_sig", "", sigma)
        sg.setConstant(True)
    return mu, sg, lines


def build(h_bw, m_c, sigma, v, mode, label, rate, out_dir, mass, anchor,
          sigmode="fix"):
    A, b_hat, b_err = anchor
    i0, i1 = snap(h_bw, v["fit_lo"], v["fit_hi"])
    slo = h_bw.GetXaxis().GetBinLowEdge(i0)
    shi = h_bw.GetXaxis().GetBinUpEdge(i1)
    nbins = i1 - i0 + 1
    h_win = ROOT.TH1D("h_win", "", nbins, slo, shi)
    for i in range(nbins):
        h_win.SetBinContent(i + 1, h_bw.GetBinContent(i0 + i))
    nobs = h_win.Integral()

    x = ROOT.RooRealVar("mass", "m [GeV]", slo, shi)
    x.setBins(nbins)
    data = ROOT.RooDataHist("data_obs", "", ROOT.RooArgList(x), h_win)

    if mode in ("float", "bconstr"):
        par = ROOT.RooRealVar("b_expo", "b_expo", b_hat, -100.0, 0.0)
        extra = ("b_expo flatParam" if mode == "float"
                 else f"b_expo param {b_hat:.5g} {b_err:.5g}")
        extra += "\nbkg_pdf_norm flatParam"
        norm = ROOT.RooRealVar("bkg_pdf_norm", "", nobs, 0.0,
                               max(10.0 * nobs, 50.0))
    elif mode == "bfixed":
        par = ROOT.RooRealVar("b_expo", "b_expo", b_hat)
        par.setConstant(True)
        extra = "bkg_pdf_norm flatParam"
        norm = ROOT.RooRealVar("bkg_pdf_norm", "", nobs, 0.0,
                               max(10.0 * nobs, 50.0))
    else:                                        # anch: both constant
        par = ROOT.RooRealVar("b_expo", "b_expo", b_hat)
        par.setConstant(True)
        ax = h_bw.GetXaxis()
        b_env = sum(A * ROOT.TMath.Exp(b_hat * (ax.GetBinCenter(i) - 2000.0)
                                       / 1000.0) * (v["bw"] / 100.0)
                    for i in range(i0, i1 + 1))
        norm = ROOT.RooRealVar("bkg_pdf_norm", "", b_env)
        norm.setConstant(True)
        extra = f"# background fully anchored (B_env={b_env:.2f})"
    bkg = ROOT.RooGenericPdf("bkg_pdf",
                             f"exp(@1*((@0-{m_c:.10g})/1000.))",
                             ROOT.RooArgList(x, par))

    mu, sg, sig_lines = _sig_pars(sigmode, m_c, sigma)
    if sig_lines:
        extra += "\n" + "\n".join(sig_lines)
    sig = ROOT.RooGaussian("sig_pdf", "", x, mu, sg)

    w = ROOT.RooWorkspace("w", "w")
    for obj in (data, bkg, norm, sig):
        getattr(w, "import")(obj)
    ws = f"ws_{label}_m{mass}.root"
    w.writeToFile(str(out_dir / ws))
    card = out_dir / f"card_{label}_m{mass}.txt"
    card.write_text(CARD.format(mass=mass, label=label, ws=ws, slo=slo,
                                shi=shi, bw=v["bw"], m_c=m_c, sigma=sigma,
                                rate=rate, extra=extra))
    print(f"  {card.name}: [{slo:.0f},{shi:.0f}]x{v['bw']:.0f} B={nobs:.1f} "
          f"rate={rate:.2f}")
    return card


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=HERE / "inputs")
    p.add_argument("--output-dir", type=Path, default=HERE / "cards")
    args = p.parse_args()
    tag = "ee_resolved"

    with open(args.input_dir / f"{tag}.json") as fh:
        meta = json.load(fh)
    fin = ROOT.TFile(str(args.input_dir / f"{tag}.root"))
    h_native = fin.Get("bkg_native")            # 10 GeV bins
    h_by_bw = {bw: h_native.Rebin(int(bw // 10), f"h{bw:.0f}")
               for bw in (100.0, 50.0, 20.0)}
    anchor = fit_anchor(h_by_bw[100.0])

    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for mass_s, m in sorted(meta["masses"].items(), key=lambda kv: int(kv[0])):
        mass = int(mass_s)
        for combo in COMBOS:
            wname, bw, mode = combo[:3]
            sigmode = combo[3] if len(combo) > 3 else "fix"
            v = m["vars"][f"{wname}_bw{bw:.0f}"]
            label = f"{wname}_bw{bw}_{mode}" + (
                "" if sigmode == "fix" else f"_{sigmode}")
            card = build(h_by_bw[float(bw)], m["m_c"], m["sigma"], v, mode,
                         label, v["rate_per_fb"], out_dir, mass, anchor,
                         sigmode=sigmode)
            manifest.append(f"{card}\t{mass}\t{label}")
    (out_dir / "manifest.txt").write_text("\n".join(manifest) + "\n")
    print(f"{len(manifest)} cards -> {out_dir}/manifest.txt")


if __name__ == "__main__":
    main()
