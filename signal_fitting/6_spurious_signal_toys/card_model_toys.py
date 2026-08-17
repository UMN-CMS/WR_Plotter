#!/usr/bin/env python3
"""Stage 6 (card model) -- spurious-signal toys of the COMBINE-CARD likelihood.

The classic Stage-6 toys fit the Stage-4 TF1 chi2 model. This variant runs the
same Poisson-toy machinery through the statistical model of the actual combine
cards (shared/card_sb_fit.CardSB), so the homemade N_sp distribution and
combine share binning, window snap, and constraint structure:

  card_float   the k5_bw50 float card: fixed Gaussian signal + single
               floating expo (norm + slope), 50 GeV bins, all-bins Poisson
               likelihood
  card_fcr     the k5_bw50 fcr card: SR background split into tt+tW (expo,
               slope fixed to the MC component fit, norm = mu_tt x T_tt) and
               rest (floating expo), PLUS a one-bin flavor-CR channel
               (obs ~ Poisson(T_cr + C_other), expectation mu_tt T_cr +
               C_other) -- the homemade rateParam. CR bin is fluctuated in
               every toy, exactly like combine's frequentist toys would.

Geometry, split yields, and the tt slope come from the 8_combine_limits
optimization inputs (prepare_opt.py + prepare_fcr.py), i.e. the SAME files
the workspace builders read -- no re-derivation on this side.

The summary CSV uses the Stage-6 schema (function = card_float / card_fcr),
so 7_limit_plots/expected_limit.py maps it to a Brazil band unchanged:

  python expected_limit.py --era RunIISummer20UL18 \\
      --table ../6_spurious_signal_toys/run2/card/card_toy_table_ee_resolved.csv \\
      --functions card_float card_fcr --output-dir run2/card

Outputs (default <script dir>/run2/card):
  nsp_hist/{ch}_{topo}/{model}/m{mWR}.*      per-mass N_sp distribution
  card_toy_table_{ch}_{topo}.csv             summary rows (Stage-6 schema
                                             + mu_tt_mean / mu_tt_rms)

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import array
import csv
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                        # repo root
sys.path.insert(0, str(HERE.parents[0] / "shared"))             # card_sb_fit
sys.path.insert(0, str(HERE))                                   # plot reuse

from wrplotter.cli_utils import setup_logging                           # noqa: E402
from wrplotter.config import load_lumi                                  # noqa: E402

import ROOT                                                             # noqa: E402
from card_sb_fit import CardSB, rebin_snap                              # noqa: E402
from spurious_signal_toys import plot_nsp_hist                          # noqa: E402

logger = logging.getLogger("card_model_toys")

MODELS = ["card_float", "card_fcr"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIISummer20UL18",
                   help="the optimization inputs are run2; era sets lumi/labels")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved",
                   choices=["resolved", "boosted"])
    p.add_argument("--variant", default="k5_bw50",
                   help="window_bw key from the optimization inputs")
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--cr-obs", default="mc", choices=["mc", "EGamma", "Muon"],
                   help="flavor-CR observation for card_fcr. 'mc': Asimov "
                        "(mu_tt = 1 truth). A dataset name: the CR DATA count "
                        "anchors mu_tt = (N_data - C_other)/T_cr, and the toy "
                        "truth is made consistent (SR tt scaled by it, CR bin "
                        "centred on N_data). SR observation stays MC (blind). "
                        "CSV rows are labeled card_fcrd in this mode.")
    p.add_argument("--masses", nargs="+", type=int, default=None,
                   help="subset of the optimization-input masses")
    p.add_argument("--ntoys", type=int, default=1000)
    p.add_argument("--min-toys", type=int, default=100)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--opt-inputs", type=Path,
                   default=HERE.parents[0] / "8_combine_limits" / "optimization"
                   / "inputs")
    p.add_argument("--no-toy-plots", action="store_true")
    p.add_argument("--hist-range", type=float, default=60.0)
    p.add_argument("--hist-bins", type=int, default=30)
    p.add_argument("--hist-adaptive", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: <script dir>/<run2|run3>/card")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    channel, topology = args.channel, args.topology
    tag = f"{channel}_{topology}"
    info = load_lumi(args.era)
    lumi, com = info["lumi"], info.get("com", 13.6)
    run_sub = {"RunII": "run2", "Run3": "run3"}[str(info["run"])]
    if args.output_dir is None:
        args.output_dir = HERE / run_sub / "card"

    with open(args.opt_inputs / f"{tag}.json") as fh:
        meta = json.load(fh)
    with open(args.opt_inputs / f"{tag}_fcr.json") as fh:
        fcr_meta = json.load(fh)
    import uproot
    h = uproot.open(args.opt_inputs / f"{tag}.root")["bkg_native"]
    e10, v10 = h.axes[0].edges(), h.values()
    if args.cr_obs != "mc":
        fs = uproot.open(args.opt_inputs / f"{tag}_fcr.root")
        tt10, rest10 = fs["tt_native"].values(), fs["rest_native"].values()

    masses = sorted(int(m) for m in meta["masses"])
    if args.masses:
        masses = [m for m in masses if m in set(args.masses)]
    logger.info("Card-model toys: %s %s %s -- %d masses, models %s, %d toys",
                channel, topology, args.variant, len(masses), args.models,
                args.ntoys)

    rows = []
    for mass in masses:
        m = meta["masses"][str(mass)]
        v = m["vars"][args.variant]
        f = fcr_meta["masses"][str(mass)]["vars"][args.variant]
        edges, vals = rebin_snap(e10, v10, v["bw"], v["fit_lo"], v["fit_hi"])
        mu_sr = np.where(np.isfinite(vals) & (vals > 0), vals, 0.0)
        mu_cr = f["T_cr"] + f["C_other"]            # mu_tt = 1 truth
        b_window = float(mu_sr.sum())

        for model in args.models:
            mode = "fcr" if model == "card_fcr" else "float"
            midx = MODELS.index(model)
            label = model
            mu_sr_m, mu_cr_m, b_win_m = mu_sr, mu_cr, b_window
            sb = CardSB(edges, m["m_c"], m["sigma"], mode=mode, fcr=f)
            if mode == "fcr" and args.cr_obs != "mc":
                # CR DATA anchor: truth made consistent with the measured
                # mu_tt -- SR tt scaled, CR bin centred on the data count.
                # (SR stays blind: its "data" is still the scaled MC.)
                n_data = f[f"N_data_{args.cr_obs}"]
                mu_tt_data = (n_data - f["C_other"]) / f["T_cr"]
                _, t_v = rebin_snap(e10, tt10, v["bw"],
                                    v["fit_lo"], v["fit_hi"])
                _, r_v = rebin_snap(e10, rest10, v["bw"],
                                    v["fit_lo"], v["fit_hi"])
                vals_d = mu_tt_data * t_v + r_v
                mu_sr_m = np.where(np.isfinite(vals_d) & (vals_d > 0),
                                   vals_d, 0.0)
                mu_cr_m = n_data
                b_win_m = float(mu_sr_m.sum())
                label = "card_fcrd"
                logger.info("  m=%d: CR data (%s) N=%.0f -> mu_tt=%.3f, "
                            "B_window %.1f", mass, args.cr_obs, n_data,
                            mu_tt_data, b_win_m)

            asi = sb.fit(mu_sr_m, data_cr=mu_cr_m if mode == "fcr" else None)
            nsp_asimov = asi["nsig"] if asi["status"] == 0 else float("nan")
            nsp_asimov_err = (asi["nsig_err"] if asi["status"] == 0
                              else float("nan"))

            seed = args.seed * 1_000_003 + mass * 1009 + midx
            rng = ROOT.TRandom3(seed)               # ROOT RNG (not numpy)
            nsps, pulls, mu_tts = [], [], []
            for _ in range(args.ntoys):
                data_toy = np.array([rng.Poisson(x) for x in mu_sr_m],
                                    dtype=float)
                cr_toy = (float(rng.Poisson(mu_cr_m))
                          if mode == "fcr" else None)
                res = sb.fit(data_toy, data_cr=cr_toy)
                if res["status"] != 0:
                    continue
                nf, ne = res["nsig"], res["nsig_err"]
                if not (ne > 0 and math.isfinite(ne) and math.isfinite(nf)):
                    continue
                nsps.append(nf)
                pulls.append(nf / ne)
                if mode == "fcr":
                    mu_tts.append(float(res["params"][0]))
            n_ok = len(nsps)
            base = {"channel": channel, "topology": topology,
                    "function": label, "mWR": mass, "signal_tag": "",
                    "m_N": "", "m_c": round(m["m_c"], 1),
                    "sigma_win": round(m["sigma"], 2),
                    "mu_sig": round(m["m_c"], 1),
                    "sigma_sig": round(m["sigma"], 2),
                    "fit_lo": round(float(edges[0]), 1),
                    "fit_hi": round(float(edges[-1]), 1),
                    "B_window": round(b_win_m, 3), "ntoys": args.ntoys,
                    "n_ok": n_ok,
                    "nsp_asimov": round(nsp_asimov, 4)
                    if math.isfinite(nsp_asimov) else "",
                    "nsp_asimov_err": round(nsp_asimov_err, 4)
                    if math.isfinite(nsp_asimov_err) else ""}
            if n_ok < args.min_toys:
                rows.append({**base, "mean_Nsp": "", "rms_Nsp": "",
                             "pull_mean": "", "pull_mean_err": "",
                             "pull_width": "", "pull_width_err": "",
                             "frac_pull_gt_0p5": "", "q95_abs_Nsp": "",
                             "mu_tt_mean": "", "mu_tt_rms": ""})
                logger.info("  m=%d %-10s -> only %d/%d toys ok, skip",
                            mass, label, n_ok, args.ntoys)
                continue
            nsps_a = array.array("d", nsps)
            pulls_a = array.array("d", pulls)
            mean_nsp = float(ROOT.TMath.Mean(n_ok, nsps_a))
            rms_nsp = float(ROOT.TMath.RMS(n_ok, nsps_a))
            pmean = float(ROOT.TMath.Mean(n_ok, pulls_a))
            pwidth = float(ROOT.TMath.RMS(n_ok, pulls_a))
            frac_gt = sum(1 for x in pulls if abs(x) > 0.5) / n_ok
            absn = array.array("d", sorted(abs(x) for x in nsps))
            qval, qprob = array.array("d", [0.0]), array.array("d", [0.95])
            ROOT.TMath.Quantiles(n_ok, 1, absn, qval, qprob, True)
            q95 = float(qval[0])
            if mu_tts:
                mt_a = array.array("d", mu_tts)
                mt_mean = float(ROOT.TMath.Mean(len(mu_tts), mt_a))
                mt_rms = float(ROOT.TMath.RMS(len(mu_tts), mt_a))
            else:
                mt_mean = mt_rms = None
            rows.append({**base,
                         "mean_Nsp": round(mean_nsp, 4),
                         "rms_Nsp": round(rms_nsp, 4),
                         "pull_mean": round(pmean, 4),
                         "pull_mean_err": round(pwidth / math.sqrt(n_ok), 4),
                         "pull_width": round(pwidth, 4),
                         "pull_width_err":
                         round(pwidth / math.sqrt(2 * (n_ok - 1)), 4),
                         "frac_pull_gt_0p5": round(frac_gt, 4),
                         "q95_abs_Nsp": round(q95, 4),
                         "mu_tt_mean": round(mt_mean, 4)
                         if mt_mean is not None else "",
                         "mu_tt_rms": round(mt_rms, 4)
                         if mt_rms is not None else ""})
            logger.info("  m=%d %-10s -> <Nsp>=%.1f+/-%.1f (Asimov %.1f) "
                        "pull=%.2f+/-%.2f%s [%d toys]", mass, label, mean_nsp,
                        rms_nsp, nsp_asimov, pmean, pwidth,
                        f" mu_tt={mt_mean:.3f}+/-{mt_rms:.3f}"
                        if mt_mean is not None else "", n_ok)
            if not args.no_toy_plots:
                plot_nsp_hist(
                    nsps, nsp_asimov, mass, "expo",   # style key only
                    args.output_dir / "nsp_hist" / tag / label / f"m{mass}",
                    mean=mean_nsp, rms=rms_nsp,
                    hist_range=args.hist_range, hist_bins=args.hist_bins,
                    channel=channel, topology=topology, com=com, lumi=lumi,
                    k=5.0, ntoys=args.ntoys, adaptive=args.hist_adaptive)

    if not rows:
        logger.error("No results produced.")
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"card_toy_table_{tag}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("  wrote %s", csv_path)


if __name__ == "__main__":
    main()
