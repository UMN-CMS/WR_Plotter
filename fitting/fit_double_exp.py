#!/usr/bin/env python3
"""
Simultaneous SR + flavor CR double-exponential fit of the m_lljj background.

The signal region (SR) is modelled as the sum of two exponentials:
    SR:  N1 * exp(c1 * m) + N2 * exp(c2 * m)

The flavor control region (FCR, emu + mue events) constrains the
flavor-symmetric component (tt+tW-like):
    FCR: 2 * N1 * exp(c1 * m)

The factor of 2 arises because the FCR combines emu and mue, while the
SR uses a single same-flavor channel (ee or mumu).  For flavor-symmetric
backgrounds the emu yield equals the ee yield, so emu + mue = 2 * ee.

A RooSimultaneous fit determines c1, c2, N1, N2 simultaneously from both
regions.  The FCR provides an independent constraint on the steep slope c1
and the yield N1, breaking the near-total degeneracy (corr(N1,N2) ~ -1)
observed in single-region double-exponential fits.

Usage examples:
    # Default (both channels, resolved topology):
    python fitting/fit_double_exp.py --era RunIII2024Summer24 --dir 20260223_lo_dy

    # Single channel with custom mass range:
    python fitting/fit_double_exp.py --era RunIII2024Summer24 --dir 20260223_lo_dy \
        --channel ee --mass-range 1000 4000 --rebin 10
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add repo root to path so wrplotter imports work from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import ROOT
except ImportError:
    sys.exit(
        "ERROR: PyROOT is not available. Please set up a ROOT-enabled environment.\n"
        "  Options: cmsenv (CMSSW), conda install root, or source LCG views."
    )

from wrplotter.cli_utils import setup_logging
from wrplotter.config import load_lumi
from wrplotter.paths import input_dirs_for_era, repo_root

from fitting.fit_utils import (
    add_common_args,
    build_region_name,
    build_hist_key,
    load_and_sum_backgrounds,
    load_single_background,
    create_roodatahist,
    run_fit,
    extract_fit_results,
    save_results_json,
    plot_fit,
)
from fitting.fit_single_exp import (
    COMPONENT_FILES,
    plot_background_composition,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flavor CR region helpers
# ---------------------------------------------------------------------------

def build_fcr_region_name(topology: str) -> str:
    """Return the flavor CR region directory name.

    For resolved topology the flavor CR is a single combined emu+mue region.
    Boosted topology has separate emu/mue regions which would need to be summed;
    that is not yet implemented.
    """
    if topology != "resolved":
        raise NotImplementedError(
            f"Flavor CR for topology '{topology}' is not yet implemented. "
            "Boosted flavor CR has separate emu/mue regions that need summing."
        )
    return f"wr_{topology}_flavor_cr"


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_simul_double_exp_model(
    observable: ROOT.RooRealVar,
    sr_data: ROOT.RooDataHist,
    fcr_data: ROOT.RooDataHist,
) -> tuple:
    """Build a simultaneous SR + FCR double-exponential model.

    SR model:  N1 * exp(c1 * m) + N2 * exp(c2 * m)
    FCR model: 2 * N1 * exp(c1 * m)

    The steep slope c1 and yield N1 are shared between SR and FCR.
    The factor of 2 on the FCR yield accounts for emu + mue = 2 * ee.

    Parameters are initialized from the single-exp component fits:
      tt+tW  -> c ~ -0.0032, ~70% of SR events  (steep, flavor-symmetric)
      DYJets -> c ~ -0.0024, ~30% of SR events  (shallow)

    Returns:
        (simPdf, combData, params, model_sr, model_fcr)
        where params = {"c1": ..., "c2": ..., "n1": ..., "n2": ...}
    """
    n_sr = sr_data.sumEntries()

    # --- Free parameters ---
    c1 = ROOT.RooRealVar("c1", "steep slope (flavor-symmetric)", -0.0032, -0.02, 0)
    c2 = ROOT.RooRealVar("c2", "shallow slope (DYJets-like)", -0.0024, -0.02, 0)
    n1 = ROOT.RooRealVar("n1", "SR yield of steep component", 0.7 * n_sr, 0, 50000)
    n2 = ROOT.RooRealVar("n2", "SR yield of shallow component", 0.3 * n_sr, 0, 50000)

    # --- Shared exponential shapes ---
    exp1 = ROOT.RooExponential("exp1", "steep exponential", observable, c1)
    exp2 = ROOT.RooExponential("exp2", "shallow exponential", observable, c2)

    # --- SR model: N1 * exp(c1*m) + N2 * exp(c2*m) ---
    ext_exp1_sr = ROOT.RooExtendPdf("ext_exp1_sr", "extended steep (SR)", exp1, n1)
    ext_exp2_sr = ROOT.RooExtendPdf("ext_exp2_sr", "extended shallow (SR)", exp2, n2)
    model_sr = ROOT.RooAddPdf(
        "model_sr", "double exponential (SR)",
        ROOT.RooArgList(ext_exp1_sr, ext_exp2_sr),
    )

    # --- FCR model: 2 * N1 * exp(c1*m) ---
    # RooFormulaVar computes 2*N1; shares the underlying n1 parameter
    n1_scaled = ROOT.RooFormulaVar("n1_scaled", "2*@0", ROOT.RooArgList(n1))
    model_fcr = ROOT.RooExtendPdf(
        "model_fcr", "scaled steep exp (FCR)", exp1, n1_scaled,
    )

    # --- Simultaneous PDF ---
    sample = ROOT.RooCategory("sample", "sample")
    sample.defineType("SR")
    sample.defineType("FCR")

    simPdf = ROOT.RooSimultaneous("simPdf", "simultaneous SR + FCR", sample)
    simPdf.addPdf(model_sr, "SR")
    simPdf.addPdf(model_fcr, "FCR")

    # --- Combined dataset indexed by category ---
    combData = ROOT.RooDataHist(
        "combData", "combined SR + FCR",
        ROOT.RooArgSet(observable),
        ROOT.RooFit.Index(sample),
        ROOT.RooFit.Import("SR", sr_data),
        ROOT.RooFit.Import("FCR", fcr_data),
    )

    # Prevent Python garbage collection — C++ objects hold cross-references.
    # RooSimultaneous::clone() (called internally by fitTo) needs all
    # component PDFs, the RooCategory, and the RooFormulaVar to be alive.
    for obj in (exp1, exp2, ext_exp1_sr, ext_exp2_sr, n1_scaled,
                model_sr, model_fcr, sample, simPdf, combData):
        ROOT.SetOwnership(obj, False)

    params = {"c1": c1, "c2": c2, "n1": n1, "n2": n2}
    return simPdf, combData, params, model_sr, model_fcr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simultaneous SR + flavor CR double-exponential background fit."
    )
    add_common_args(parser)
    parser.add_argument(
        "--decompose-fcr", action="store_true",
        help="Produce a composition plot and JSON for the flavor CR, "
             "showing per-process fractions (DYJets, tt+tW, Other, Nonprompt).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    # Keep matplotlib/font_manager quiet even in --verbose mode
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    ROOT.gROOT.SetBatch(True)
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

    # --- Resolve paths ---
    info = load_lumi(args.era)
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    input_dir = input_dirs[0]

    base_output_dir = Path(
        args.output_dir
        or str(repo_root() / "fitting" / "outputs" / args.era / "double_exp_simul")
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    channels = [args.channel] if args.channel else ["ee", "mumu"]

    # --- FCR region (shared across channels) ---
    fcr_region = build_fcr_region_name(args.topology)
    fcr_hist_key = build_hist_key(fcr_region)
    logger.info("Flavor CR region: %s", fcr_region)
    logger.info("Flavor CR histogram key: %s", fcr_hist_key)

    # Load and rebin FCR histogram once (it's the same for ee and mumu)
    fcr_hist_raw = load_and_sum_backgrounds(input_dir, fcr_hist_key)
    fcr_rebinned = fcr_hist_raw.Rebin(args.rebin, "h_fcr_rebinned")

    # --- FCR composition (optional) ---
    if args.decompose_fcr:
        logger.info("=" * 60)
        logger.info("Decomposing flavor CR backgrounds...")
        logger.info("=" * 60)

        # Load each background component for the FCR region
        fcr_component_hists = {}
        for label, filename in COMPONENT_FILES.items():
            try:
                h = load_single_background(input_dir, fcr_hist_key, filename)
                fcr_component_hists[label] = h.Rebin(
                    args.rebin, f"h_fcr_{label}_rebinned"
                )
            except RuntimeError:
                logger.warning("Could not load %s for FCR composition", label)

        # Need the observable for the composition plot x-axis range
        mass_lo, mass_hi = args.mass_range
        mass_comp = ROOT.RooRealVar(
            "mass_comp", "m_{lljj}", mass_lo, mass_hi, "GeV"
        )

        # Output directory (use same config tag as the fit)
        bin_width_gev = int(fcr_rebinned.GetBinWidth(1))
        config_tag = f"{int(mass_lo)}-{int(mass_hi)}_{bin_width_gev}GeV"
        comp_output_dir = base_output_dir / config_tag
        comp_output_dir.mkdir(parents=True, exist_ok=True)

        # Composition plot
        plot_background_composition(
            fcr_component_hists, fcr_rebinned, mass_comp,
            comp_output_dir / "composition_fcr.pdf",
            lumi=info["lumi"],
            com=info.get("com", 13.6),
            channel="emu",
            topology=args.topology,
            era=args.era,
            bin_width_gev=fcr_rebinned.GetBinWidth(1),
            region_type="Flavor CR",
        )

        # Composition JSON — per-bin fractions for each component
        comp_json = {"metadata": {
            "region": fcr_region,
            "topology": args.topology,
            "era": args.era,
            "mass_range": [mass_lo, mass_hi],
            "bin_width_gev": fcr_rebinned.GetBinWidth(1),
        }, "bins": []}
        for i in range(1, fcr_rebinned.GetNbinsX() + 1):
            center = fcr_rebinned.GetBinCenter(i)
            total = fcr_rebinned.GetBinContent(i)
            bin_entry = {"bin_center": center, "total": total, "fractions": {}}
            for lbl, ch in fcr_component_hists.items():
                val = ch.GetBinContent(i)
                bin_entry["fractions"][lbl] = {
                    "yield": val,
                    "fraction": val / total if total > 0 else 0.0,
                }
            comp_json["bins"].append(bin_entry)
        save_results_json(comp_json, comp_output_dir / "composition_fcr.json")

        # Summary table
        logger.info("FCR composition summary:")
        total_fcr = fcr_rebinned.Integral()
        for label, hist in fcr_component_hists.items():
            integral = hist.Integral()
            frac = integral / total_fcr if total_fcr > 0 else 0.0
            logger.info("  %-12s  %7.1f events  (%5.1f%%)", label, integral, 100 * frac)
        logger.info("  %-12s  %7.1f events", "Total", total_fcr)
        logger.info("=" * 60)

    # Pre-compute a shared y-range across all channels so plots are comparable
    global_ymin, global_ymax = float("inf"), 0.0
    for ch in channels:
        r = build_region_name(ch, args.topology)
        hk = build_hist_key(r)
        h = load_and_sum_backgrounds(input_dir, hk)
        h = h.Rebin(args.rebin, f"h_yrange_{ch}")
        for i in range(1, h.GetNbinsX() + 1):
            v = h.GetBinContent(i)
            if v > 0:
                global_ymin = min(global_ymin, v)
                global_ymax = max(global_ymax, v)
    global_y_range = (global_ymin * 0.3, global_ymax * 50000) if global_ymax > 0 else None

    # FCR y-range (separate since FCR typically has different yield)
    fcr_ymin, fcr_ymax = float("inf"), 0.0
    for i in range(1, fcr_rebinned.GetNbinsX() + 1):
        v = fcr_rebinned.GetBinContent(i)
        if v > 0:
            fcr_ymin = min(fcr_ymin, v)
            fcr_ymax = max(fcr_ymax, v)
    fcr_y_range = (fcr_ymin * 0.3, fcr_ymax * 50000) if fcr_ymax > 0 else None

    for channel in channels:

        # --- SR histogram ---
        sr_region = build_region_name(channel, args.topology)
        sr_hist_key = build_hist_key(sr_region)
        logger.info("SR region: %s", sr_region)
        logger.info("SR histogram key: %s", sr_hist_key)

        sr_hist_raw = load_and_sum_backgrounds(input_dir, sr_hist_key)
        sr_rebinned = sr_hist_raw.Rebin(args.rebin, "h_sr_rebinned")
        logger.info(
            "SR rebinned: %d bins (factor %d, %.0f GeV/bin)",
            sr_rebinned.GetNbinsX(), args.rebin, sr_rebinned.GetBinWidth(1),
        )

        # --- Define observable with fit range ---
        mass_lo, mass_hi = args.mass_range
        mass = ROOT.RooRealVar("mass", "m_{lljj}", mass_lo, mass_hi, "GeV")

        # Output subdirectory for this mass-range / binning configuration
        bin_width_gev = int(sr_rebinned.GetBinWidth(1))
        config_tag = f"{int(mass_lo)}-{int(mass_hi)}_{bin_width_gev}GeV"
        chan_output_dir = base_output_dir / config_tag
        chan_output_dir.mkdir(parents=True, exist_ok=True)

        # --- Create RooDataHists ---
        sr_data = create_roodatahist(sr_rebinned, mass, name="data_sr")
        fcr_data = create_roodatahist(fcr_rebinned, mass, name="data_fcr")
        logger.info("SR entries in fit range: %.1f", sr_data.sumEntries())
        logger.info("FCR entries in fit range: %.1f", fcr_data.sumEntries())

        # --- Build simultaneous model ---
        simPdf, combData, params, model_sr, model_fcr = build_simul_double_exp_model(
            mass, sr_data, fcr_data,
        )
        logger.info(
            "Simultaneous model: 4 free parameters (c1, c2, n1, n2), "
            "SR = double-exp, FCR = 2*N1*exp(c1*m)"
        )

        # --- Fit ---
        fit_result = run_fit(simPdf, combData, params)

        # --- Extract results ---
        results = {
            "metadata": {
                "era": args.era,
                "channel": channel,
                "topology": args.topology,
                "sr_region": sr_region,
                "fcr_region": fcr_region,
                "variable": "mass_fourobject",
                "model": "double-exp-simultaneous",
                "mass_range": [mass_lo, mass_hi],
                "rebin": args.rebin,
                "n_events_sr": float(sr_data.sumEntries()),
                "n_events_fcr": float(fcr_data.sumEntries()),
                "timestamp": datetime.now().isoformat(),
            },
            **extract_fit_results(fit_result, params),
        }

        # --- Goodness of fit for each region ---
        # SR chi2/ndf
        sr_frame = mass.frame()
        sr_data.plotOn(sr_frame, ROOT.RooFit.Name("sr_data"))
        model_sr.plotOn(sr_frame, ROOT.RooFit.Name("sr_fit"))
        sr_roo_hist = sr_frame.getHist("sr_data")
        n_sr_bins = sr_roo_hist.GetN()
        # SR has 4 effective parameters: c1, c2, n1, n2
        ndf_sr = n_sr_bins - 4
        chi2_ndf_sr = sr_frame.chiSquare("sr_fit", "sr_data", 4)
        results["goodness_of_fit_sr"] = {
            "chi2_per_ndf": chi2_ndf_sr,
            "ndf": ndf_sr,
            "n_bins": n_sr_bins,
            "chi2": chi2_ndf_sr * ndf_sr,
        }

        # FCR chi2/ndf
        fcr_frame = mass.frame()
        fcr_data.plotOn(fcr_frame, ROOT.RooFit.Name("fcr_data"))
        model_fcr.plotOn(fcr_frame, ROOT.RooFit.Name("fcr_fit"))
        fcr_roo_hist = fcr_frame.getHist("fcr_data")
        n_fcr_bins = fcr_roo_hist.GetN()
        # FCR depends on 2 parameters: c1 and n1 (n1_scaled = 2*n1)
        ndf_fcr = n_fcr_bins - 2
        chi2_ndf_fcr = fcr_frame.chiSquare("fcr_fit", "fcr_data", 2)
        results["goodness_of_fit_fcr"] = {
            "chi2_per_ndf": chi2_ndf_fcr,
            "ndf": ndf_fcr,
            "n_bins": n_fcr_bins,
            "chi2": chi2_ndf_fcr * ndf_fcr,
        }

        # --- Save JSON ---
        json_path = chan_output_dir / f"fit_{channel}.json"
        save_results_json(results, json_path)

        # --- SR plot ---
        sr_plot_path = chan_output_dir / f"fit_sr_{channel}.pdf"
        plot_fit(
            mass, sr_data, model_sr, fit_result, params, sr_plot_path,
            lumi=info["lumi"],
            com=info.get("com", 13.6),
            channel=channel,
            topology=args.topology,
            era=args.era,
            model_label="Simul. double exp (SR)",
            bin_width_gev=sr_rebinned.GetBinWidth(1),
            y_range=global_y_range,
            show_components=False,
        )

        # --- FCR plot (single component) ---
        # Build display-only params for the FCR plot: show c1 and 2*N1
        n_fcr_display = ROOT.RooRealVar(
            "n_fcr", "FCR yield (2*N1)",
            2 * params["n1"].getVal(),
        )
        n_fcr_display.setError(2 * params["n1"].getError())
        fcr_display_params = {"c1": params["c1"], "n_fcr": n_fcr_display}

        fcr_plot_path = chan_output_dir / f"fit_fcr_{channel}.pdf"
        plot_fit(
            mass, fcr_data, model_fcr, fit_result, fcr_display_params, fcr_plot_path,
            lumi=info["lumi"],
            com=info.get("com", 13.6),
            channel="emu",
            topology=args.topology,
            era=args.era,
            model_label="Simul. double exp (FCR)",
            bin_width_gev=sr_rebinned.GetBinWidth(1),
            y_range=fcr_y_range,
            region_type="Flavor CR",
        )

        # --- Summary ---
        logger.info("=" * 60)
        logger.info("Simultaneous double-exp fit summary (%s %s):", channel, args.topology)
        logger.info("  Fit status: %d (0 = converged)", results["fit_status"]["status"])
        logger.info("  Cov quality: %d (3 = full accurate)", results["fit_status"]["cov_quality"])
        logger.info("  SR chi2/ndf: %.2f (%d dof)", chi2_ndf_sr, ndf_sr)
        logger.info("  FCR chi2/ndf: %.2f (%d dof)", chi2_ndf_fcr, ndf_fcr)
        for name, p in results["parameters"].items():
            logger.info("  %s = %.6f +/- %.6f", name, p["value"], p["error"])

        # Highlight the key diagnostic: n1-n2 correlation
        corr_matrix = results["correlation_matrix"]
        param_names = sorted(params.keys())
        i_n1 = param_names.index("n1")
        i_n2 = param_names.index("n2")
        corr_n1_n2 = corr_matrix[i_n1][i_n2]
        logger.info("  corr(n1, n2) = %.3f  (was ~-1.0 in single-region fit)", corr_n1_n2)
        logger.info("  Output: %s", chan_output_dir)
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
