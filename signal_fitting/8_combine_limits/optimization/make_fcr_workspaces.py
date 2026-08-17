#!/usr/bin/env python3
"""Flavor-CR variant, step 2 -- build the CR-anchored workspaces (container).

Card `k5_bw50_fcr`: the 10.9 winner geometry (k5 window, 50 GeV bins, fixed
Gaussian signal, MC-Asimov observation) with the summed floating background
split into

  tt    expo with the slope FIXED to the MC component fit (b_tt from
        prepare_fcr), normalization tied to the flavor CR through the shared
        `mu_tt rateParam` -- the CR-anchored piece
  rest  DY + Nonprompt + Other, floating expo (norm + slope flatParam),
        exactly the treatment the float card gives the whole background

plus a second, one-bin counting channel `fcr` (shapes FAKE): the flavor-CR
yield in the same mass window, processes tt (rate T_cr, scaled by the same
mu_tt) and other (fixed C_other). Stat-only: no lnN on the SR/CR transfer
factor yet.

  python3 make_fcr_workspaces.py               (all masses, k5_bw50)
  python3 make_fcr_workspaces.py --variants k5_bw50 k3_bw100
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
# flavor-CR anchored card  m_WR={mass}  {label}
# window [{slo:.0f}, {shi:.0f}] bw={bw:.0f}  m_c={m_c}  sigma={sigma}
# rate = lumi*eff = {rate:.4f} ev/fb -> r is sigma x BR in fb
# SR: tt (slope fixed {b_tt:.4g}, norm = mu_tt x {t_tt:.6g}) + rest (floating expo)
# FCR: one counting bin, obs {obs_fcr:.6g} = tt {t_cr:.6g} (x mu_tt) + other {c_other:.6g}
imax 2
jmax *
kmax *
----------------
shapes sig       win  {ws} w:sig_pdf
shapes tt        win  {ws} w:tt_pdf
shapes rest      win  {ws} w:rest_pdf
shapes data_obs  win  {ws} w:data_obs
shapes *         fcr  FAKE
----------------
bin          win  fcr
observation  -1   {obs_fcr:.6g}
----------------
bin      win          win          win    fcr          fcr
process  sig          tt           rest   tt           other
process  0            1            2      1            3
rate     {rate:.6g}  {t_tt:.6g}  1      {t_cr:.6g}  {c_other:.6g}
----------------
rest_pdf_norm flatParam
b_rest flatParam
mu_tt rateParam * tt 1 [0.0,5.0]
"""


def snap(h, lo, hi):
    ax = h.GetXaxis()
    i0 = next(i for i in range(1, h.GetNbinsX() + 1) if ax.GetBinCenter(i) >= lo)
    i1 = next(i for i in range(h.GetNbinsX(), 0, -1) if ax.GetBinCenter(i) <= hi)
    return i0, i1


def window_hist(h_bw, lo, hi):
    i0, i1 = snap(h_bw, lo, hi)
    slo = h_bw.GetXaxis().GetBinLowEdge(i0)
    shi = h_bw.GetXaxis().GetBinUpEdge(i1)
    nbins = i1 - i0 + 1
    h_win = ROOT.TH1D("h_win", "", nbins, slo, shi)
    for i in range(nbins):
        h_win.SetBinContent(i + 1, h_bw.GetBinContent(i0 + i))
    return h_win, slo, shi, nbins


def build(h_tot_bw, m_c, sigma, v, f, label, rate, out_dir, mass,
          obs_fcr=None):
    """obs_fcr: the fcr-channel observation. None -> MC Asimov (T_cr +
    C_other); a number -> that count (the unblinded flavor-CR DATA -- the SR
    observation stays MC either way)."""
    h_win, slo, shi, nbins = window_hist(h_tot_bw, v["fit_lo"], v["fit_hi"])

    x = ROOT.RooRealVar("mass", "m [GeV]", slo, shi)
    x.setBins(nbins)
    data = ROOT.RooDataHist("data_obs", "", ROOT.RooArgList(x), h_win)

    b_tt = ROOT.RooRealVar("b_tt", "b_tt", f["b_tt"])
    b_tt.setConstant(True)
    tt_pdf = ROOT.RooGenericPdf("tt_pdf", f"exp(@1*((@0-{m_c:.10g})/1000.))",
                                ROOT.RooArgList(x, b_tt))

    b_rest = ROOT.RooRealVar("b_rest", "b_rest", -3.0, -100.0, 0.0)
    rest_pdf = ROOT.RooGenericPdf("rest_pdf",
                                  f"exp(@1*((@0-{m_c:.10g})/1000.))",
                                  ROOT.RooArgList(x, b_rest))
    rest_norm = ROOT.RooRealVar("rest_pdf_norm", "", f["B_rest"], 0.0,
                                max(10.0 * f["B_rest"], 50.0))

    mu = ROOT.RooRealVar("mu_sig", "", m_c)
    mu.setConstant(True)
    sg = ROOT.RooRealVar("sigma_sig", "", sigma)
    sg.setConstant(True)
    sig = ROOT.RooGaussian("sig_pdf", "", x, mu, sg)

    w = ROOT.RooWorkspace("w", "w")
    for obj in (data, tt_pdf, rest_pdf, rest_norm, sig):
        getattr(w, "import")(obj)
    ws = f"ws_{label}_m{mass}.root"
    w.writeToFile(str(out_dir / ws))

    if obs_fcr is None:
        obs_fcr = f["T_cr"] + f["C_other"]
    card = out_dir / f"card_{label}_m{mass}.txt"
    card.write_text(CARD.format(
        mass=mass, label=label, ws=ws, slo=slo, shi=shi, bw=v["bw"], m_c=m_c,
        sigma=sigma, rate=rate, b_tt=f["b_tt"], t_tt=f["T_tt"],
        t_cr=f["T_cr"], c_other=f["C_other"], obs_fcr=obs_fcr))
    print(f"  {card.name}: [{slo:.0f},{shi:.0f}]x{v['bw']:.0f} "
          f"T_tt={f['T_tt']:.1f} B_rest={f['B_rest']:.1f} "
          f"T_cr={f['T_cr']:.1f} C_other={f['C_other']:.1f} rate={rate:.2f}")
    return card


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=HERE / "inputs")
    p.add_argument("--output-dir", type=Path, default=HERE / "cards")
    p.add_argument("--variants", nargs="+", default=["k5_bw50"],
                   help="window_bw variant keys to build (default: the winner)")
    p.add_argument("--cr-dataset", default="EGamma",
                   choices=["EGamma", "Muon"],
                   help="which primary dataset's count observes the fcr "
                        "channel in the *_fcrd (CR-data) cards")
    args = p.parse_args()
    tag = "ee_resolved"

    with open(args.input_dir / f"{tag}.json") as fh:
        meta = json.load(fh)
    with open(args.input_dir / f"{tag}_fcr.json") as fh:
        fcr = json.load(fh)
    fin = ROOT.TFile(str(args.input_dir / f"{tag}.root"))
    h_native = fin.Get("bkg_native")            # 10 GeV bins, tt+rest summed

    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for mass_s, m in sorted(meta["masses"].items(), key=lambda kv: int(kv[0])):
        mass = int(mass_s)
        for vkey in args.variants:
            v = m["vars"][vkey]
            f = fcr["masses"][mass_s]["vars"][vkey]
            h_bw = h_native.Rebin(int(v["bw"] // 10), f"h{v['bw']:.0f}")
            # fcr: CR observed = MC Asimov; fcrd: CR observed = DATA
            for label, obs in ((f"{vkey}_fcr", None),
                               (f"{vkey}_fcrd",
                                f.get(f"N_data_{args.cr_dataset}"))):
                if label.endswith("fcrd") and obs is None:
                    continue                # inputs predate the data counts
                card = build(h_bw, m["m_c"], m["sigma"], v, f, label,
                             v["rate_per_fb"], out_dir, mass, obs_fcr=obs)
                manifest.append(f"{card}\t{mass}\t{label}")
    (out_dir / "manifest_fcr.txt").write_text("\n".join(manifest) + "\n")
    print(f"{len(manifest)} cards -> {out_dir}/manifest_fcr.txt")


if __name__ == "__main__":
    main()
