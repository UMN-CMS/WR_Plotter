#!/usr/bin/env python3
"""Stage 10.6, step 1 -- parity workspaces/datacards for CMS combine.

Combine is a binned Poisson-ML machine (empty bins included, nuisances as
constraint terms, profile-likelihood CLs) -- exactly the convention the
Stage-10 fit core adopted.  This step builds the cards that test the
equivalence quantitatively, in three variants:

  fixed     the Stage-9 model verbatim: floating background (norm + slope),
            Gaussian signal with (mu, sigma) CONSTANT.  Baseline.
  prior     same, but sigma_sig FLOATS inside [0.5, 2.5]*sigma0 with the
            datacard Gaussian constraint  `sigma_sig param sigma0 0.3*sigma0`
            -- the combine encoding of the Stage-10.2 winner (alpha_mu = 0,
            alpha_sigma = 0.3).
  anchored  sparse-mass variant: background slope FIXED to the Stage-10.4
            envelope central fit and normalization FIXED to the envelope
            window integral B_env; only r floats.  This is the card-level
            version of the envelope's anchored estimator, to be run with
            HybridNew (toys) against the 10.5 exact counting band --
            AsymptoticLimits is unreliable at a few events, which is the point.

Signal `rate` = 1, so the POI r is the expected signal yield in EVENTS --
directly comparable to the 10.5 band (UL on N_sig).  rMax per card is taken
from the 10.5 table (4x the +2sigma edge).

Runs on plain PyROOT + stdlib ONLY (execute inside the combine container; see
run_parity.sh).  Reads Stage-9 inputs (8_combine_limits/baseline inputs +
.root) and the Stage-10.4 envelope outputs; writes cards/{tag}/... + manifest.

Usage:
  python3 make_workspaces_parity.py --channel ee --topology resolved
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

HERE = Path(__file__).resolve().parent                  # 6_combine_parity
STAGE10 = HERE.parent
STAGE9 = STAGE10.parent / "8_combine_limits" / "baseline"

EXPO_FORMULA = "exp(@1*((@0-{m0})/1000.))"

CARD = """\
# parity card  {tag}  m_WR={mass}  variant={variant}
# window [{slo:.0f}, {shi:.0f}], m_c={m_c}, sigma0={sigma}
# rate = 1 -> r is the signal yield in EVENTS (cf. 10.5 band)
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
rate     1     1
----------------
{extra}
"""


def snap_range(h, lo, hi):
    """First/last bin whose center is inside [lo, hi] (TH1::Fit's selection)."""
    ax = h.GetXaxis()
    i0 = next(i for i in range(1, h.GetNbinsX() + 1) if ax.GetBinCenter(i) >= lo)
    i1 = next(i for i in range(h.GetNbinsX(), 0, -1) if ax.GetBinCenter(i) <= hi)
    return i0, i1


def build_one(h_full, m, variant, out_dir, tag, *, b_anchor=None, b_env=None,
              sigma_prior_frac=0.3, rmax=100.0):
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

    if variant == "anchored":
        par = ROOT.RooRealVar("b_expo", "b_expo", b_anchor)
        par.setConstant(True)
        norm = ROOT.RooRealVar("bkg_pdf_norm", "", b_env)
        norm.setConstant(True)
        extra = "# background fully anchored to the Stage-10.4 envelope"
    else:
        par = ROOT.RooRealVar("b_expo", "b_expo", -3.0, -100.0, 0.0)
        norm = ROOT.RooRealVar("bkg_pdf_norm", "", nobs, 0.0,
                               max(10.0 * nobs, 50.0))
        extra = "b_expo flatParam\nbkg_pdf_norm flatParam"
    bkg = ROOT.RooGenericPdf(
        "bkg_pdf", EXPO_FORMULA.format(m0=f"{m['m_c']:.10g}"),
        ROOT.RooArgList(x, par))

    mu = ROOT.RooRealVar("mu_sig", "", m["m_c"])
    mu.setConstant(True)
    s0 = float(m["sigma"])
    if variant == "prior":
        sg = ROOT.RooRealVar("sigma_sig", "", s0, 0.5 * s0, 2.5 * s0)
        sg.setConstant(False)
        extra += f"\nsigma_sig param {s0:.4g} {sigma_prior_frac * s0:.4g}"
    else:
        sg = ROOT.RooRealVar("sigma_sig", "", s0)
        sg.setConstant(True)
    sig = ROOT.RooGaussian("sig_pdf", "", x, mu, sg)

    w = ROOT.RooWorkspace("w", "w")
    for obj in (data, bkg, norm, sig):
        getattr(w, "import")(obj)

    ws_name = f"ws_{variant}_m{mass_i}.root"
    w.writeToFile(str(out_dir / ws_name))
    card = out_dir / f"card_{variant}_m{mass_i}.txt"
    card.write_text(CARD.format(
        tag=tag, mass=mass_i, variant=variant, ws=ws_name,
        slo=slo, shi=shi, m_c=m["m_c"], sigma=m["sigma"], extra=extra))
    print(f"  {card.name}: bins [{slo:.0f},{shi:.0f}], B_obs={nobs:.1f}"
          + (f", b_anchor={b_anchor:.3f}, B_env={b_env:.2f}"
             if variant == "anchored" else "")
          + f", rMax={rmax:.0f}")
    return card, mass_i


def load_v2_rmax(v2_csv, default=100.0):
    """{mass: rMax} = 4x the v2 +2sigma edge (fallback 4x median, then default)."""
    out = {}
    if not v2_csv.exists():
        return out
    with open(v2_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            hi = r.get("ul_p2s") or r.get("ul_med")
            try:
                out[int(float(r["mWR"]))] = max(4.0 * float(hi), 20.0)
            except (TypeError, ValueError):
                pass
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--channel", default="ee")
    p.add_argument("--topology", default="resolved")
    p.add_argument("--fit-masses", nargs="+", type=int,
                   default=[2000, 2400, 2800, 3200],
                   help="fit-regime benchmarks: fixed + prior cards, "
                        "AsymptoticLimits")
    p.add_argument("--sparse-masses", nargs="+", type=int,
                   default=[4000, 4600],
                   help="counting-regime benchmarks: anchored cards, HybridNew")
    p.add_argument("--sigma-prior-frac", type=float, default=0.3)
    p.add_argument("--input-dir", type=Path, default=STAGE9 / "inputs")
    p.add_argument("--envelope-dir", type=Path, default=STAGE10 / "4_bkg_envelope")
    p.add_argument("--v2-table", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=HERE / "cards")
    args = p.parse_args()
    tag = f"{args.channel}_{args.topology}"

    with open(args.input_dir / f"{tag}.json") as fh:
        meta = json.load(fh)["masses"]
    fin = ROOT.TFile(str(args.input_dir / f"{tag}.root"))
    h_full = fin.Get("data_obs_full")

    with open(args.envelope_dir / f"envelope_members_{tag}.json") as fh:
        members = json.load(fh)
    central = next(m for m in members if m["label"] == "central")
    b_anchor = central["params_pivot2000"][1]     # expo slope: pivot-independent
    b_env = {}
    with open(args.envelope_dir / f"nsp_prediction_{tag}.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            b_env[int(float(r["mWR"]))] = float(r["B_window_env"])
    v2 = args.v2_table or (STAGE10 / "5_expected_limits_v2"
                           / f"expected_limit_v2_table_{tag}.csv")
    rmax = load_v2_rmax(v2)

    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for mass in args.fit_masses:
        m = dict(meta[str(mass)], mass=mass)
        for variant in ("fixed", "prior"):
            rm = rmax.get(mass, 100.0)
            card, mi = build_one(h_full, m, variant, out_dir, tag,
                                 sigma_prior_frac=args.sigma_prior_frac,
                                 rmax=rm)
            manifest.append(f"{card}\t{mi}\t{variant}\tfit\t{rm:.0f}")
    for mass in args.sparse_masses:
        m = dict(meta[str(mass)], mass=mass)
        if mass not in b_env:
            print(f"  m={mass}: no envelope prediction, skipped")
            continue
        rm = rmax.get(mass, 40.0)
        card, mi = build_one(h_full, m, "anchored", out_dir, tag,
                             b_anchor=b_anchor, b_env=b_env[mass], rmax=rm)
        manifest.append(f"{card}\t{mi}\tanchored\tsparse\t{rm:.0f}")
    (out_dir / "manifest.txt").write_text("\n".join(manifest) + "\n")
    print(f"{len(manifest)} cards -> {out_dir}/manifest.txt")


if __name__ == "__main__":
    main()
