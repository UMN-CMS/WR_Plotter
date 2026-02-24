#!/usr/bin/env python3
"""
Fit summed MC background m_lljj distribution with double-exp or pow-exp models.

For single exponential fits, use fit_single_exp.py instead.

Usage examples:
    # Double exponential:
    python fitting/fit_background.py --era RunIII2024Summer24 --dir 20260223_lo_dy \
        --model double-exp

    # Power-law exponential:
    python fitting/fit_background.py --era RunIII2024Summer24 --dir 20260223_lo_dy \
        --model pow-exp

    # Scan double-exp initializations:
    python fitting/fit_background.py --era RunIII2024Summer24 --dir 20260223_lo_dy \
        --model double-exp --scan
"""
from __future__ import annotations

import argparse
import json
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
    create_roodatahist,
    run_fit,
    extract_fit_results,
    save_results_json,
    plot_fit,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_double_exp_model(
    observable: ROOT.RooRealVar,
    n_data: float,
) -> tuple[ROOT.RooAddPdf, dict[str, ROOT.RooRealVar]]:
    """Build N1 * exp(c1*m) + N2 * exp(c2*m) as an extended PDF.

    Two RooExponential shapes, each wrapped in RooExtendPdf with its own
    yield parameter (N1, N2), combined via RooAddPdf.  The extended
    likelihood constrains both shapes and yields simultaneously.

    This is a purely empirical model — the two components do not correspond
    to specific physics processes. c1 is the steeper (more negative) slope
    and c2 is the shallower (less negative) slope.

    Initial values informed by per-component single-exp fits (ee, [1000, 4000]):
      tt+tW  -> c = -0.00324, n_bkg = 609   (steep, ~70% of events)
      DYJets -> c = -0.00258, n_bkg = 248   (shallow, ~30% of events)
    """
    c1 = ROOT.RooRealVar("c1", "steep slope", -0.0032, -0.02, 0)
    c2 = ROOT.RooRealVar("c2", "shallow slope", -0.0026, -0.02, 0)
    n1 = ROOT.RooRealVar("n1", "yield of steep component", 0.7 * n_data, 0, 50000)
    n2 = ROOT.RooRealVar("n2", "yield of shallow component", 0.3 * n_data, 0, 50000)

    exp1 = ROOT.RooExponential("exp1", "steep exponential", observable, c1)
    exp2 = ROOT.RooExponential("exp2", "shallow exponential", observable, c2)
    ext1 = ROOT.RooExtendPdf("ext_exp1", "extended steep exp", exp1, n1)
    ext2 = ROOT.RooExtendPdf("ext_exp2", "extended shallow exp", exp2, n2)
    # Prevent Python GC — C++ RooAddPdf holds refs to all four objects
    for obj in (exp1, exp2, ext1, ext2):
        ROOT.SetOwnership(obj, False)

    pdf = ROOT.RooAddPdf("double_exp", "double exponential",
                         ROOT.RooArgList(ext1, ext2))

    return pdf, {"c1": c1, "c2": c2, "n1": n1, "n2": n2}


def build_pow_exp_model(
    observable: ROOT.RooRealVar,
    n_data: float,
) -> tuple[ROOT.RooExtendPdf, dict[str, ROOT.RooRealVar]]:
    """Build N_bkg * m^a * exp(c * m) as an extended PDF.

    Power-law x exponential: the power-law factor allows the effective slope
    to vary with mass, unlike a pure exponential.  Two parameters (a, c) give
    similar flexibility to a double exponential without the degeneracy issues.
    RooExtendPdf adds a Poisson term for the total yield.

    Initial values:
      a = -1  (mild power-law suppression; range [-10, 10] is broad
               so the fit can decide)
      c = -0.002 (less negative than the single-exp slope of -0.003,
                   because the power-law absorbs part of the falloff)
    """
    a = ROOT.RooRealVar("a", "power-law index", -1.0, -10.0, 10.0)
    c = ROOT.RooRealVar("c", "exponential slope", -0.002, -0.01, 0.0)
    n_bkg = ROOT.RooRealVar("n_bkg", "background yield", n_data, 0, 50000)

    obs_name = observable.GetName()
    formula = f"pow({obs_name}, a) * exp(c * {obs_name})"
    shape = ROOT.RooGenericPdf(
        "pow_exp", "power-law times exponential",
        formula,
        ROOT.RooArgList(observable, a, c),
    )
    ROOT.SetOwnership(shape, False)  # prevent Python GC; C++ RooExtendPdf holds a ref

    pdf = ROOT.RooExtendPdf("ext_pow_exp", "extended power-law exp", shape, n_bkg)

    return pdf, {"a": a, "c": c, "n_bkg": n_bkg}


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def run_staged_double_exp_fit(
    pdf: ROOT.RooAddPdf,
    data: ROOT.RooDataHist,
    params: dict[str, ROOT.RooRealVar],
) -> ROOT.RooFitResult:
    """Staged fitting strategy for the double exponential.

    Multi-parameter exponential fits are notoriously sensitive to initialization.
    This staged approach helps Minuit find the global minimum:
      1. Fix the shallow component, fit only the steep slope and yields
      2. Release all parameters and do the full fit
      3. If that fails, retry with the most thorough strategy
    """
    logger.info("Stage 1: fitting steep component only (c2 and n2 fixed)...")
    params["c2"].setConstant(True)
    params["n2"].setConstant(True)
    pdf.fitTo(
        data,
        ROOT.RooFit.Strategy(0),
        ROOT.RooFit.PrintLevel(-1),
    )
    logger.info("  c1 after stage 1: %.6f", params["c1"].getVal())

    logger.info("Stage 2: releasing all parameters for full fit...")
    params["c2"].setConstant(False)
    params["n2"].setConstant(False)
    result = run_fit(pdf, data, params, strategy=1)

    return result


def scan_double_exp_initializations(
    observable: ROOT.RooRealVar,
    data: ROOT.RooDataHist,
    n_data: float,
) -> list[dict]:
    """Try many initialization values for the double-exp model.

    Scans a grid of (c1_init, c2_init, n1_frac_init) values, fits each one,
    and returns a list of result dicts sorted by NLL (best first).
    This helps determine whether the data supports a genuine two-component
    solution or if all starting points converge to the same minimum.
    """
    # Grid of initialization values to try
    c1_inits = [-0.008, -0.005, -0.004, -0.003, -0.002]
    c2_inits = [-0.0012, -0.0008, -0.0005, -0.0002, -0.00005]
    # Fraction of n_data assigned to the steep component
    n1_frac_inits = [0.2, 0.5, 0.8, 0.95]

    # Also vary the parameter ranges to avoid boundary effects
    range_configs = [
        # (c1_range, c2_range, label)
        ((-0.02, -0.0015), (-0.0015, -0.00001), "default"),
        ((-0.02, -0.001), (-0.001, -0.00001), "split at -0.001"),
        ((-0.02, -0.002), (-0.002, -0.00001), "split at -0.002"),
        ((-0.05, -0.0005), (-0.0005, -0.000001), "wide c1, narrow c2"),
    ]

    results = []
    total = len(c1_inits) * len(c2_inits) * len(n1_frac_inits) * len(range_configs)
    logger.info("Scanning %d initialization combinations...", total)

    # Suppress RooFit output during scan
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.FATAL)

    for i_rc, (c1_range, c2_range, range_label) in enumerate(range_configs):
        for c1_init in c1_inits:
            # Skip if init is outside range
            if not (c1_range[0] <= c1_init <= c1_range[1]):
                continue
            for c2_init in c2_inits:
                if not (c2_range[0] <= c2_init <= c2_range[1]):
                    continue
                for n1_frac in n1_frac_inits:
                    # Build fresh model for each attempt
                    c1 = ROOT.RooRealVar("c1", "steep slope", c1_init, *c1_range)
                    c2 = ROOT.RooRealVar("c2", "shallow slope", c2_init, *c2_range)
                    n1 = ROOT.RooRealVar("n1", "yield steep", n_data * n1_frac, 0, 50000)
                    n2 = ROOT.RooRealVar("n2", "yield shallow", n_data * (1 - n1_frac), 0, 50000)

                    exp1 = ROOT.RooExponential("scan_exp1", "steep", observable, c1)
                    exp2 = ROOT.RooExponential("scan_exp2", "shallow", observable, c2)
                    ext1 = ROOT.RooExtendPdf("scan_ext1", "ext steep", exp1, n1)
                    ext2 = ROOT.RooExtendPdf("scan_ext2", "ext shallow", exp2, n2)
                    for obj in (exp1, exp2, ext1, ext2):
                        ROOT.SetOwnership(obj, False)
                    pdf = ROOT.RooAddPdf("scan_pdf", "scan pdf",
                                         ROOT.RooArgList(ext1, ext2))

                    try:
                        result = pdf.fitTo(
                            data,
                            ROOT.RooFit.Save(True),
                            ROOT.RooFit.Strategy(1),
                            ROOT.RooFit.PrintLevel(-1),
                        )
                    except Exception:
                        continue

                    if result is None:
                        continue

                    # Check if either yield is near zero (degenerate solution)
                    total_yield = n1.getVal() + n2.getVal()
                    degenerate = (
                        total_yield < 1
                        or n1.getVal() / total_yield < 0.02
                        or n2.getVal() / total_yield < 0.02
                    )

                    results.append({
                        "c1_init": c1_init,
                        "c2_init": c2_init,
                        "n1_frac_init": n1_frac,
                        "range_label": range_label,
                        "c1_fit": c1.getVal(),
                        "c2_fit": c2.getVal(),
                        "n1_fit": n1.getVal(),
                        "n2_fit": n2.getVal(),
                        "c1_err": c1.getError(),
                        "c2_err": c2.getError(),
                        "n1_err": n1.getError(),
                        "n2_err": n2.getError(),
                        "nll": result.minNll(),
                        "status": result.status(),
                        "cov_qual": result.covQual(),
                        "edm": result.edm(),
                        "degenerate": degenerate,
                    })

    # Restore RooFit output
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

    # Sort by NLL (best first)
    results.sort(key=lambda r: r["nll"])
    return results


def print_scan_summary(results: list[dict]) -> None:
    """Print a formatted table of scan results."""
    if not results:
        logger.info("No successful fits in scan.")
        return

    # Find unique minima (group by NLL within tolerance)
    minima = []
    nll_tol = 0.5  # fits within 0.5 NLL units are considered the same minimum
    for r in results:
        found = False
        for m in minima:
            if abs(r["nll"] - m["nll"]) < nll_tol:
                m["count"] += 1
                found = True
                break
        if not found:
            minima.append({**r, "count": 1})

    print("\n" + "=" * 110)
    print("DOUBLE-EXP INITIALIZATION SCAN RESULTS")
    print("=" * 110)
    print(f"\nTotal fits attempted: {len(results)}")
    print(f"Converged (status=0): {sum(1 for r in results if r['status'] == 0)}")
    print(f"Unique minima found: {len(minima)}")
    print(f"Non-degenerate: "
          f"{sum(1 for r in results if not r['degenerate'] and r['status'] == 0)}")

    print("\n--- Unique minima (sorted by NLL) ---")
    print(f"{'NLL':>12s}  {'c1':>10s}  {'c2':>10s}  {'N1':>8s}  {'N2':>8s}  "
          f"{'status':>6s}  {'covQ':>4s}  {'#fits':>5s}  {'degen?':>6s}")
    print("-" * 90)
    for m in minima:
        print(f"{m['nll']:12.2f}  {m['c1_fit']:10.6f}  {m['c2_fit']:10.6f}  "
              f"{m['n1_fit']:8.1f}  {m['n2_fit']:8.1f}  "
              f"{m['status']:6d}  {m['cov_qual']:4d}  "
              f"{m['count']:5d}  {'YES' if m['degenerate'] else 'no':>6s}")

    # Show the best non-degenerate fit if it exists
    non_degen = [r for r in results if not r["degenerate"] and r["status"] == 0]
    if non_degen:
        best = non_degen[0]
        print(f"\n--- Best non-degenerate fit ---")
        print(f"  c1 = {best['c1_fit']:.6f} +/- {best['c1_err']:.6f}  "
              f"(init: {best['c1_init']:.4f}, range: {best['range_label']})")
        print(f"  c2 = {best['c2_fit']:.6f} +/- {best['c2_err']:.6f}  "
              f"(init: {best['c2_init']:.5f})")
        print(f"  N1 = {best['n1_fit']:.1f} +/- {best['n1_err']:.1f}")
        print(f"  N2 = {best['n2_fit']:.1f} +/- {best['n2_err']:.1f}")
        print(f"  NLL = {best['nll']:.2f}, status = {best['status']}, "
              f"covQual = {best['cov_qual']}")
    else:
        print("\n  ** No non-degenerate fits found — all converge to one component **")

    print("=" * 110 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the m_lljj background distribution with double-exp or pow-exp models."
    )
    add_common_args(parser)

    parser.add_argument(
        "--model",
        choices=["double-exp", "pow-exp"],
        default="double-exp",
        help="Fit model (default: double-exp)",
    )
    parser.add_argument(
        "--scan", action="store_true",
        help="(double-exp only) Scan over many initialization values and report "
             "all minima found. Saves JSON + plots only for the best fit.",
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
    # Suppress RooFit banner
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

    # --- Resolve paths ---
    info = load_lumi(args.era)
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    input_dir = input_dirs[0]

    model_slug = args.model.replace("-", "_")
    base_output_dir = Path(
        args.output_dir
        or str(repo_root() / "fitting" / "outputs" / args.era / model_slug)
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    channels = [args.channel] if args.channel else ["ee", "mumu"]

    # Pre-compute a shared y-range across all channels so plots are comparable.
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
    global_y_range = (global_ymin * 0.3, global_ymax * 5000) if global_ymax > 0 else None

    for channel in channels:

        # --- Build histogram path ---
        region = build_region_name(channel, args.topology)
        hist_key = build_hist_key(region)
        logger.info("Region: %s", region)
        logger.info("Histogram key: %s", hist_key)
        logger.info("Input directory: %s", input_dir)

        # --- Load and sum background histograms ---
        summed_hist = load_and_sum_backgrounds(input_dir, hist_key)

        # --- Rebin ---
        rebinned = summed_hist.Rebin(args.rebin, "h_bkg_rebinned")
        logger.info(
            "Rebinned: %d bins (factor %d, %.0f GeV/bin)",
            rebinned.GetNbinsX(), args.rebin, rebinned.GetBinWidth(1),
        )

        # --- Define observable with fit range ---
        mass_lo, mass_hi = args.mass_range
        mass = ROOT.RooRealVar("mass", "m_{lljj}", mass_lo, mass_hi, "GeV")

        # Subdirectory for this mass-range / binning configuration
        bin_width_gev = int(rebinned.GetBinWidth(1))
        config_tag = f"{int(mass_lo)}-{int(mass_hi)}_{bin_width_gev}GeV"
        chan_output_dir = base_output_dir / config_tag
        chan_output_dir.mkdir(parents=True, exist_ok=True)

        # --- Create RooDataHist ---
        data_hist = create_roodatahist(rebinned, mass)
        logger.info("RooDataHist entries: %.1f (in fit range)", data_hist.sumEntries())

        # --- Build model ---
        n_data = data_hist.sumEntries()
        model_name = args.model
        if model_name == "double-exp":
            pdf, params = build_double_exp_model(mass, n_data)
        elif model_name == "pow-exp":
            pdf, params = build_pow_exp_model(mass, n_data)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        logger.info("Model: %s (%d parameters)", model_name, len(params))

        # --- Initialization scan (double-exp only) ---
        if args.scan:
            if model_name != "double-exp":
                logger.error("--scan is only supported with --model double-exp")
                sys.exit(1)
            scan_results = scan_double_exp_initializations(mass, data_hist, n_data)
            print_scan_summary(scan_results)

            # Save scan results to JSON
            scan_path = chan_output_dir / f"scan_{channel}.json"
            with open(scan_path, "w") as f:
                json.dump(scan_results, f, indent=2)
            logger.info("Saved scan results: %s", scan_path)

            # If a non-degenerate fit was found, re-initialize the model with
            # those values so the rest of the pipeline (plot, JSON) uses them.
            non_degen = [r for r in scan_results
                         if not r["degenerate"] and r["status"] == 0]
            if non_degen:
                best = non_degen[0]
                logger.info("Re-running best non-degenerate fit for plots...")
                params["c1"].setVal(best["c1_fit"])
                params["c2"].setVal(best["c2_fit"])
                params["n1"].setVal(best["n1_fit"])
                params["n2"].setVal(best["n2_fit"])

        # --- Fit ---
        fit_result = run_fit(pdf, data_hist, params)

        # --- Extract and save results ---
        results = {
            "metadata": {
                "era": args.era,
                "channel": channel,
                "topology": args.topology,
                "region": region,
                "variable": "mass_fourobject",
                "model": model_name,
                "mass_range": [mass_lo, mass_hi],
                "rebin": args.rebin,
                "n_events_total": float(summed_hist.Integral()),
                "n_events_fit_range": float(data_hist.sumEntries()),
                "timestamp": datetime.now().isoformat(),
            },
            **extract_fit_results(fit_result, params),
        }

        # Compute goodness of fit from the plot frame.
        tmp_frame = mass.frame()
        data_hist.plotOn(tmp_frame, ROOT.RooFit.Name("tmp_data"))
        pdf.plotOn(tmp_frame, ROOT.RooFit.Name("tmp_fit"))
        roo_hist = tmp_frame.getHist("tmp_data")
        n_data_bins = roo_hist.GetN()
        ndf = n_data_bins - len(params)
        chi2_ndf = tmp_frame.chiSquare("tmp_fit", "tmp_data", len(params))
        results["goodness_of_fit"] = {
            "chi2_per_ndf": chi2_ndf,
            "ndf": ndf,
            "n_bins": n_data_bins,
            "chi2": chi2_ndf * ndf,
        }

        json_path = chan_output_dir / f"fit_{channel}.json"
        save_results_json(results, json_path)

        # --- Plot ---
        plot_path = chan_output_dir / f"fit_{channel}.pdf"
        plot_fit(
            mass, data_hist, pdf, fit_result, params, plot_path,
            lumi=info["lumi"],
            com=info.get("com", 13.6),
            channel=channel,
            topology=args.topology,
            era=args.era,
            model_label={
                "double-exp": "Double exponential",
                "pow-exp": "Power-law exp",
            }[model_name],
            bin_width_gev=rebinned.GetBinWidth(1),
            y_range=global_y_range,
        )

        # --- Summary ---
        logger.info("=" * 60)
        logger.info("Fit summary for %s in %s %s SR:", model_name, channel, args.topology)
        logger.info("  Status: %d (0 = converged)", results["fit_status"]["status"])
        logger.info("  chi2/ndf: %.2f (%d dof)", chi2_ndf, ndf)
        for name, p in results["parameters"].items():
            logger.info("  %s = %.6f +/- %.6f", name, p["value"], p["error"])
        logger.info("  Output: %s", chan_output_dir)
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
