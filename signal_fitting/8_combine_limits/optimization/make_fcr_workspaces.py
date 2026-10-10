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
{r_syst_line}"""


CARD_SHAPE = """\
# flavor-CR SHAPE-anchored card  m_WR={mass}  {label}
# window [{slo:.0f}, {shi:.0f}] bw={bw:.0f}  m_c={m_c}  sigma={sigma}
# rate = lumi*eff = {rate:.4f} ev/fb -> r is sigma x BR in fb
# SR:  tt = R x T x expo(b_tt) + rest (floating expo) + r x signal
# FCR: tt = T x expo(b_tt) + other (fixed MC template); 30-bin data histogram
# T (via mu_tt) and b_tt are SHARED between the channels: the simultaneous
# shape fit of the seminar slides.  R = {R:.4f} (global MC transfer factor)
imax 2
jmax *
kmax *
----------------
shapes sig       win  {ws} w:sig_pdf
shapes tt        win  {ws} w:tt_pdf
shapes rest      win  {ws} w:rest_pdf
shapes data_obs  win  {ws} w:data_obs
shapes tt        fcr  {ws} w:tt_cr_pdf
shapes other     fcr  {ws} w:other_cr_pdf
shapes data_obs  fcr  {ws} w:data_obs_cr
----------------
bin          win  fcr
observation  -1   -1
----------------
bin      win          win          win    fcr          fcr
process  sig          tt           rest   tt           other
process  0            1            2      1            3
rate     {rate:.6g}  {rt_cr:.6g}  1      {t_cr:.6g}  {c_other:.6g}
----------------
rest_pdf_norm flatParam
b_rest flatParam
b_tt flatParam
mu_tt rateParam * tt 1 [0.0,5.0]
{r_syst_line}{sig_syst_lines}"""


def signal_syst_lines(sig_systs, mass):
    """Card lines for the signal systematics at this mass, interpolated in
    m_WR between the measured points (relative effects): one lnN per group
    on the sig column, and param constraints on mu_sig / sigma_sig for the
    sources that move the peak."""
    if not sig_systs:
        return "", {"mu_rel": 0.0, "sigma_rel": 0.0}
    pts = sorted((int(m), e) for m, e in sig_systs["masses"].items())
    ms = [m for m, _ in pts]

    def interp(getter):
        ys = [getter(e) for _, e in pts]
        if mass <= ms[0]:
            return ys[0]
        if mass >= ms[-1]:
            return ys[-1]
        import numpy as _np
        return float(_np.interp(mass, ms, ys))

    lines, shape = [], {"mu_rel": 0.0, "sigma_rel": 0.0}
    for g in sorted({k for _, e in pts for k in e["groups"]}):
        rel = interp(lambda e: e["groups"].get(g, {}).get("rate_rel", 0.0))
        if rel > 1e-4:
            lines.append(f"sig_{g} lnN {1.0 + rel:.4f} - - - -")
        if g in ("elescale", "elesmear", "jes", "jer"):
            mu_rel = interp(lambda e: e["groups"].get(g, {}).get("mu_rel", 0.0))
            sg_rel = interp(lambda e: e["groups"].get(g, {}).get("sigma_rel", 0.0))
            shape["mu_rel"] = (shape["mu_rel"] ** 2 + mu_rel ** 2) ** 0.5
            shape["sigma_rel"] = (shape["sigma_rel"] ** 2 + sg_rel ** 2) ** 0.5
    return "\n".join(lines) + ("\n" if lines else ""), shape


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
    # systematics on the SR/CR transfer factor (prepare_fcr --systs): one
    # log-normal on the SR tt column only -- exactly an uncertainty on R
    r_err_rel = float(f.get("R_err_rel", 0.0) or 0.0)
    r_syst_line = (f"R_syst lnN - {1.0 + r_err_rel:.4f} - - -\n"
                   if r_err_rel > 0 else "")
    card = out_dir / f"card_{label}_m{mass}.txt"
    card.write_text(CARD.format(
        mass=mass, label=label, ws=ws, slo=slo, shi=shi, bw=v["bw"], m_c=m_c,
        sigma=sigma, rate=rate, b_tt=f["b_tt"], t_tt=f["T_tt"],
        t_cr=f["T_cr"], c_other=f["C_other"], obs_fcr=obs_fcr,
        r_syst_line=r_syst_line))
    print(f"  {card.name}: [{slo:.0f},{shi:.0f}]x{v['bw']:.0f} "
          f"T_tt={f['T_tt']:.1f} B_rest={f['B_rest']:.1f} "
          f"T_cr={f['T_cr']:.1f} C_other={f['C_other']:.1f} rate={rate:.2f}")
    return card


def build_shape(h_tot_bw, h_crdata_bw, h_crrest_bw, m_c, sigma, v, f, label,
                rate, out_dir, mass, r_global, sig_systs=None):
    """The simultaneous SHAPE fit: SR window (signal + tt expo + free expo)
    and the flavor-CR window DATA histogram (tt expo + fixed non-tt MC
    template), tt size and slope shared through mu_tt and b_tt."""
    h_win, slo, shi, nbins = window_hist(h_tot_bw, v["fit_lo"], v["fit_hi"])
    h_cr, slo2, shi2, nb2 = window_hist(h_crdata_bw, v["fit_lo"], v["fit_hi"])
    h_ot, _, _, _ = window_hist(h_crrest_bw, v["fit_lo"], v["fit_hi"])
    assert (slo, shi, nbins) == (slo2, shi2, nb2)
    h_cr.SetName("h_cr")
    h_ot.SetName("h_ot")

    x = ROOT.RooRealVar("mass", "m [GeV]", slo, shi)
    x.setBins(nbins)
    data = ROOT.RooDataHist("data_obs", "", ROOT.RooArgList(x), h_win)
    data_cr = ROOT.RooDataHist("data_obs_cr", "", ROOT.RooArgList(x), h_cr)
    dh_ot = ROOT.RooDataHist("dh_other_cr", "", ROOT.RooArgList(x), h_ot)
    other_cr = ROOT.RooHistPdf("other_cr_pdf", "", ROOT.RooArgSet(x), dh_ot)

    b_tt = ROOT.RooRealVar("b_tt", "b_tt", f["b_tt"], -100.0, 0.0)
    tt_pdf = ROOT.RooGenericPdf("tt_pdf", f"exp(@1*((@0-{m_c:.10g})/1000.))",
                                ROOT.RooArgList(x, b_tt))
    tt_cr_pdf = ROOT.RooGenericPdf("tt_cr_pdf", f"exp(@1*((@0-{m_c:.10g})/1000.))",
                                   ROOT.RooArgList(x, b_tt))
    b_rest = ROOT.RooRealVar("b_rest", "b_rest", -3.0, -100.0, 0.0)
    rest_pdf = ROOT.RooGenericPdf("rest_pdf",
                                  f"exp(@1*((@0-{m_c:.10g})/1000.))",
                                  ROOT.RooArgList(x, b_rest))
    rest_norm = ROOT.RooRealVar("rest_pdf_norm", "", f["B_rest"], 0.0,
                                max(10.0 * f["B_rest"], 50.0))
    sig_lines, shape = signal_syst_lines(sig_systs, mass)
    if shape["mu_rel"] > 0 or shape["sigma_rel"] > 0:
        # the peak position/width float inside Gaussian constraints (param)
        mu = ROOT.RooRealVar("mu_sig", "", m_c, 0.8 * m_c, 1.2 * m_c)
        sg = ROOT.RooRealVar("sigma_sig", "", sigma, 0.5 * sigma, 2.0 * sigma)
        sig_lines += (f"mu_sig param {m_c:.6g} {shape['mu_rel'] * m_c:.6g}\n"
                      f"sigma_sig param {sigma:.6g} {shape['sigma_rel'] * sigma:.6g}\n")
    else:
        mu = ROOT.RooRealVar("mu_sig", "", m_c)
        mu.setConstant(True)
        sg = ROOT.RooRealVar("sigma_sig", "", sigma)
        sg.setConstant(True)
    sig = ROOT.RooGaussian("sig_pdf", "", x, mu, sg)

    w = ROOT.RooWorkspace("w", "w")
    for obj in (data, data_cr, tt_pdf, tt_cr_pdf, rest_pdf, rest_norm, sig,
                other_cr):
        getattr(w, "import")(obj)
    ws = f"ws_{label}_m{mass}.root"
    w.writeToFile(str(out_dir / ws))

    r_err_rel = float(f.get("R_err_rel", 0.0) or 0.0)
    r_syst_line = (f"R_syst lnN - {1.0 + r_err_rel:.4f} - - -\n"
                   if r_err_rel > 0 else "")
    card = out_dir / f"card_{label}_m{mass}.txt"
    card.write_text(CARD_SHAPE.format(
        mass=mass, label=label, ws=ws, slo=slo, shi=shi, bw=v["bw"], m_c=m_c,
        sigma=sigma, rate=rate, R=r_global, rt_cr=r_global * f["T_cr"],
        t_cr=f["T_cr"], c_other=f["C_other"], r_syst_line=r_syst_line,
        sig_syst_lines=sig_lines))
    print(f"  {card.name}: [{slo:.0f},{shi:.0f}]x{v['bw']:.0f} shape CR "
          f"data={h_cr.Integral():.0f} R*T_cr={r_global * f['T_cr']:.1f} "
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
    p.add_argument("--signal-systs", type=Path, default=None,
                   help="signal_systs.py json: adds lnN per source group on "
                        "the signal and param constraints on mu/sigma "
                        "(shape cards only)")
    p.add_argument("--shape", action="store_true",
                   help="build ONLY the simultaneous shape-fit cards "
                        "(*_fcrs): 30-bin flavor-CR data histogram as the "
                        "second channel, tt size and slope shared")
    args = p.parse_args()
    tag = "ee_resolved"

    with open(args.input_dir / f"{tag}.json") as fh:
        meta = json.load(fh)
    with open(args.input_dir / f"{tag}_fcr.json") as fh:
        fcr = json.load(fh)
    sig_systs = None
    if args.signal_systs:
        with open(args.signal_systs) as fh:
            sig_systs = json.load(fh)
    fin = ROOT.TFile(str(args.input_dir / f"{tag}.root"))
    h_native = fin.Get("bkg_native")            # 10 GeV bins, tt+rest summed
    fcr_file = ROOT.TFile(str(args.input_dir / f"{tag}_fcr.root"))
    h_crdata_native = fcr_file.Get(f"fcr_data_{args.cr_dataset}_native")
    h_crrest_native = fcr_file.Get("fcr_rest_native")

    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for mass_s, m in sorted(meta["masses"].items(), key=lambda kv: int(kv[0])):
        mass = int(mass_s)
        for vkey in args.variants:
            v = m["vars"][vkey]
            f = fcr["masses"][mass_s]["vars"][vkey]
            h_bw = h_native.Rebin(int(v["bw"] // 10), f"h{v['bw']:.0f}")
            if args.shape:
                h_cd = h_crdata_native.Rebin(int(v["bw"] // 10), f"hcd{v['bw']:.0f}")
                h_cr = h_crrest_native.Rebin(int(v["bw"] // 10), f"hcr{v['bw']:.0f}")
                card = build_shape(h_bw, h_cd, h_cr, m["m_c"], m["sigma"], v, f,
                                   f"{vkey}_fcrs", v["rate_per_fb"], out_dir,
                                   mass, fcr["R_global"], sig_systs=sig_systs)
                manifest.append(f"{card}\t{mass}\t{vkey}_fcrs")
                continue
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
