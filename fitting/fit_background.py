#!/usr/bin/env python3
"""
Stage 1: Fit summed MC background m_lljj distribution with analytic functions.

Usage examples:
    # v0 — single exponential (default):
    python fitting/fit_background.py --era RunIII2024Summer24 --dir 20260223_lo_dy

    # v1 — double exponential:
    python fitting/fit_background.py --era RunIII2024Summer24 --dir 20260223_lo_dy --model double-exp

    # mu-mu channel, custom mass range and rebinning:
    python fitting/fit_background.py --era RunIII2024Summer24 --dir 20260223_lo_dy \\
        --channel mumu --mass-range 600 5000 --rebin 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
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

from wrplotter.cli_utils import add_era_args, setup_logging
from wrplotter.config import load_lumi
from wrplotter.paths import input_dirs_for_era, repo_root
from wrplotter.plotting_helpers import custom_log_formatter

logger = logging.getLogger(__name__)

# Background ROOT files to sum (all MC backgrounds)
BACKGROUND_FILES = [
    "WRAnalyzer_DYJets.root",
    "WRAnalyzer_tt_tW.root",
    "WRAnalyzer_Other.root",
    "WRAnalyzer_Nonprompt.root",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the m_lljj background distribution with analytic functions."
    )
    add_era_args(parser, required=True)

    parser.add_argument(
        "--channel", choices=["ee", "mumu"], default=None,
        help="Lepton channel. If omitted, runs both ee and mumu.",
    )
    parser.add_argument(
        "--topology", choices=["resolved", "boosted"], default="resolved",
        help="Event topology (default: resolved)",
    )
    parser.add_argument(
        "--model",
        choices=["single-exp", "double-exp", "pow-exp"],
        default="single-exp",
        help="Fit model (default: single-exp)",
    )
    parser.add_argument(
        "--mass-range", nargs=2, type=float, default=[800.0, 6000.0],
        metavar=("LOW", "HIGH"),
        help="Fit range in GeV (default: 800 6000)",
    )
    parser.add_argument(
        "--rebin", type=int, default=20,
        help="Rebin factor — merge N original 10-GeV bins (default: 20 → 200 GeV/bin)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for plots and JSON (default: fitting/outputs/<era>)",
    )
    parser.add_argument(
        "--scan", action="store_true",
        help="(double-exp only) Scan over many initialization values and report "
             "all minima found. Saves JSON + plots only for the best fit.",
    )
    parser.add_argument(
        "--component-fits", action="store_true",
        help="Fit DYJets and tt+tW individually with single exponentials, "
             "and plot background composition vs mass.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------

def build_region_name(channel: str, topology: str) -> str:
    """Map (channel, topology) to the ROOT histogram region directory name."""
    return f"wr_{channel}_{topology}_sr"


def build_hist_key(region: str, variable: str = "mass_fourobject") -> str:
    """Construct the full TH1 key inside the ROOT file.

    The naming convention is: {region}/{variable}_{region}
    e.g. wr_ee_resolved_sr/mass_fourobject_wr_ee_resolved_sr
    """
    return f"{region}/{variable}_{region}"


def load_and_sum_backgrounds(
    input_dir: Path,
    hist_key: str,
    background_files: list[str] | None = None,
) -> ROOT.TH1D:
    """Open each background ROOT file, retrieve the histogram, and sum them.

    Uses PyROOT (TFile/TH1) directly since RooFit needs native ROOT objects.
    SetDirectory(0) detaches histograms from their files so they survive closing.
    """
    bg_files = background_files or BACKGROUND_FILES
    combined = None

    for fname in bg_files:
        fpath = input_dir / fname
        tf = ROOT.TFile.Open(str(fpath), "READ")
        if not tf or tf.IsZombie():
            logger.warning("Cannot open %s, skipping", fpath)
            continue

        h = tf.Get(hist_key)
        if not h:
            logger.warning("Key '%s' not found in %s, skipping", hist_key, fpath)
            tf.Close()
            continue

        # Detach from file so the histogram persists after Close()
        h.SetDirectory(0)
        tf.Close()

        if combined is None:
            combined = h.Clone("h_bkg_total")
        else:
            combined.Add(h)

        logger.debug("  %s: %.1f events", fname, h.Integral())

    if combined is None:
        raise RuntimeError(
            f"No background histograms found for key '{hist_key}' in {input_dir}"
        )

    logger.info("Total background integral: %.1f events", combined.Integral())
    return combined


def load_single_background(
    input_dir: Path,
    hist_key: str,
    filename: str,
) -> ROOT.TH1D:
    """Open a single background ROOT file and retrieve its histogram."""
    fpath = input_dir / filename
    tf = ROOT.TFile.Open(str(fpath), "READ")
    if not tf or tf.IsZombie():
        raise RuntimeError(f"Cannot open {fpath}")

    h = tf.Get(hist_key)
    if not h:
        tf.Close()
        raise RuntimeError(f"Key '{hist_key}' not found in {fpath}")

    h.SetDirectory(0)
    tf.Close()
    logger.info("  %s: %.1f events", filename, h.Integral())
    return h


# ---------------------------------------------------------------------------
# RooFit data
# ---------------------------------------------------------------------------

def create_roodatahist(
    hist: ROOT.TH1D,
    observable: ROOT.RooRealVar,
    name: str = "data_bkg",
) -> ROOT.RooDataHist:
    """Convert a TH1D into a RooDataHist for use in RooFit.

    RooDataHist imports the bin contents and errors from the TH1.
    Only bins within the observable's range are included in the fit.
    """
    return ROOT.RooDataHist(name, name, ROOT.RooArgList(observable), hist)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_single_exp_model(
    observable: ROOT.RooRealVar,
    n_data: float,
) -> tuple[ROOT.RooExtendPdf, dict[str, ROOT.RooRealVar]]:
    """v0: Build N_bkg * exp(c * m) as an extended PDF.

    RooExponential provides the shape (normalized over the observable range),
    and RooExtendPdf adds a Poisson term for the total event count so the
    likelihood constrains both the shape and the normalization.

    Initial value -0.003 is estimated from the data: the background falls
    from ~420/100GeV at 800 GeV to ~12/100GeV at 2000 GeV,
    giving ln(12/420) / 1200 ≈ -0.003.
    """
    c = ROOT.RooRealVar("c", "exponential slope", -0.003, -0.01, -0.0001)
    n_bkg = ROOT.RooRealVar("n_bkg", "background yield", n_data, 0, 50000)

    shape = ROOT.RooExponential("single_exp", "single exponential", observable, c)
    ROOT.SetOwnership(shape, False)  # prevent Python GC; C++ RooExtendPdf holds a ref

    pdf = ROOT.RooExtendPdf("ext_single_exp", "extended single exponential", shape, n_bkg)

    return pdf, {"c": c, "n_bkg": n_bkg}


def build_double_exp_model(
    observable: ROOT.RooRealVar,
    n_data: float,
) -> tuple[ROOT.RooAddPdf, dict[str, ROOT.RooRealVar]]:
    """v1: Build N1 * exp(c1*m) + N2 * exp(c2*m) as an extended PDF.

    Two RooExponential shapes, each wrapped in RooExtendPdf with its own
    yield parameter (N1, N2), combined via RooAddPdf.  The extended
    likelihood constrains both shapes and yields simultaneously.

    This is a purely empirical model — the two components do not correspond
    to specific physics processes. c1 is the steeper (more negative) slope
    and c2 is the shallower (less negative) slope. The parameter ranges are
    deliberately non-overlapping to prevent Minuit from swapping the two
    components (a common degeneracy issue).

    Initial values informed by per-component single-exp fits (ee, [1000, 4000]):
      tt+tW  → c = -0.00324, n_bkg = 609   (steep, ~70% of events)
      DYJets → c = -0.00258, n_bkg = 248   (shallow, ~30% of events)
    Ranges are wide and overlapping — the initializations guide Minuit to the
    correct minimum without needing non-overlapping range constraints.
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

    Power-law × exponential: the power-law factor allows the effective slope
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
# Fitting
# ---------------------------------------------------------------------------

def run_fit(
    pdf: ROOT.RooAbsPdf,
    data: ROOT.RooDataHist,
    params: dict[str, ROOT.RooRealVar],
    strategy: int = 1,
) -> ROOT.RooFitResult:
    """Perform a binned maximum-likelihood fit.

    Key RooFit options:
    - Save(True): return the RooFitResult object with status and correlations
    - Strategy(1): default Minuit2 strategy (balance of speed and accuracy)
    - PrintLevel(1): moderate console output from Minuit
    """
    result = pdf.fitTo(
        data,
        ROOT.RooFit.Save(True),
        ROOT.RooFit.Strategy(strategy),
        ROOT.RooFit.PrintLevel(1),
    )

    logger.info("Fit status: %d (0 = converged)", result.status())
    logger.info("EDM: %.2e", result.edm())
    logger.info("Covariance quality: %d (3 = full accurate)", result.covQual())

    # If fit failed, retry with more thorough strategy
    if result.status() != 0 and strategy < 2:
        logger.warning(
            "Fit did not converge (status=%d). Retrying with strategy=2...",
            result.status(),
        )
        result = pdf.fitTo(
            data,
            ROOT.RooFit.Save(True),
            ROOT.RooFit.Strategy(2),
            ROOT.RooFit.PrintLevel(1),
        )
        logger.info("Retry status: %d", result.status())

    return result


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
        print(f"  c1 = {best['c1_fit']:.6f} ± {best['c1_err']:.6f}  "
              f"(init: {best['c1_init']:.4f}, range: {best['range_label']})")
        print(f"  c2 = {best['c2_fit']:.6f} ± {best['c2_err']:.6f}  "
              f"(init: {best['c2_init']:.5f})")
        print(f"  N1 = {best['n1_fit']:.1f} ± {best['n1_err']:.1f}")
        print(f"  N2 = {best['n2_fit']:.1f} ± {best['n2_err']:.1f}")
        print(f"  NLL = {best['nll']:.2f}, status = {best['status']}, "
              f"covQual = {best['cov_qual']}")
    else:
        print("\n  ** No non-degenerate fits found — all converge to one component **")

    print("=" * 110 + "\n")


# ---------------------------------------------------------------------------
# Component fits
# ---------------------------------------------------------------------------

# Map from ROOT file names to short labels used in output filenames and plots
COMPONENT_FILES = {
    "DYJets": "WRAnalyzer_DYJets.root",
    "tt_tW": "WRAnalyzer_tt_tW.root",
    "Other": "WRAnalyzer_Other.root",
    "Nonprompt": "WRAnalyzer_Nonprompt.root",
}

# Components that get individual single-exp fits
FIT_COMPONENTS = ["DYJets", "tt_tW"]


def fit_single_component(
    observable: ROOT.RooRealVar,
    input_dir: Path,
    hist_key: str,
    label: str,
    rebin: int,
    output_dir: Path,
    *,
    lumi: float,
    com: float,
    channel: str,
    topology: str,
    era: str,
    bin_width_gev: float,
    y_range: tuple[float, float] | None = None,
) -> tuple[dict, ROOT.TH1D]:
    """Fit a single background component with a single exponential.

    Returns (results_dict, rebinned_histogram).
    """
    logger.info("--- Fitting component: %s ---", label)

    hist = load_single_background(input_dir, hist_key, COMPONENT_FILES[label])
    rebinned = hist.Rebin(rebin, f"h_{label}_rebinned")

    comp_data = create_roodatahist(rebinned, observable, name=f"data_{label}")
    n_data = comp_data.sumEntries()
    logger.info("  %s events in fit range: %.1f", label, n_data)

    if n_data < 1:
        logger.warning("  Skipping %s — too few events", label)
        return {}, rebinned

    pdf, params = build_single_exp_model(observable, n_data)
    fit_result = run_fit(pdf, comp_data, params)

    results = extract_fit_results(fit_result, params)

    # Chi2/ndf
    tmp_frame = observable.frame()
    comp_data.plotOn(tmp_frame, ROOT.RooFit.Name("tmp_data"))
    pdf.plotOn(tmp_frame, ROOT.RooFit.Name("tmp_fit"))
    roo_hist = tmp_frame.getHist("tmp_data")
    n_bins = roo_hist.GetN()
    ndf = n_bins - len(params)
    chi2_ndf = tmp_frame.chiSquare("tmp_fit", "tmp_data", len(params))
    results["goodness_of_fit"] = {
        "chi2_per_ndf": chi2_ndf,
        "ndf": ndf,
        "n_bins": n_bins,
        "chi2": chi2_ndf * ndf,
    }

    # Save JSON
    json_path = output_dir / f"fit_{label}_{channel}.json"
    mass_lo = observable.getMin()
    mass_hi = observable.getMax()
    full_results = {
        "metadata": {
            "component": label,
            "channel": channel,
            "topology": topology,
            "era": era,
            "model": "single-exp",
            "mass_range": [mass_lo, mass_hi],
            "bin_width_gev": bin_width_gev,
            "n_events_fit_range": float(n_data),
        },
        **results,
    }
    save_results_json(full_results, json_path)

    # Plot
    plot_path = output_dir / f"fit_{label}_{channel}.pdf"
    plot_fit(
        observable, comp_data, pdf, fit_result, params, plot_path,
        lumi=lumi, com=com, channel=channel, topology=topology, era=era,
        model_label=f"Single exp ({label})",
        bin_width_gev=bin_width_gev,
        y_range=y_range,
    )

    return results, rebinned


def plot_background_composition(
    component_hists: dict[str, ROOT.TH1D],
    total_hist: ROOT.TH1D,
    observable: ROOT.RooRealVar,
    output_path: Path,
    *,
    lumi: float,
    com: float,
    channel: str,
    topology: str,
    era: str,
    bin_width_gev: float,
) -> None:
    """Plot the fractional composition of each background vs mass.

    Upper panel: fraction of total for each component (stacked area).
    Lower panel: absolute event counts per bin (log scale).
    """
    mass_lo = observable.getMin()
    mass_hi = observable.getMax()

    # Extract bin centers and contents
    n_bins = total_hist.GetNbinsX()
    bin_centers = np.array([total_hist.GetBinCenter(i) for i in range(1, n_bins + 1)])
    total_content = np.array([total_hist.GetBinContent(i) for i in range(1, n_bins + 1)])

    # Restrict to fit range
    mask = (bin_centers >= mass_lo) & (bin_centers <= mass_hi)
    bin_centers = bin_centers[mask]
    total_content = total_content[mask]

    comp_contents = {}
    for label, hist in component_hists.items():
        content = np.array([hist.GetBinContent(i) for i in range(1, n_bins + 1)])
        comp_contents[label] = content[mask]

    # Compute fractions (avoid div-by-zero)
    safe_total = np.where(total_content > 0, total_content, 1.0)
    comp_fractions = {
        label: content / safe_total for label, content in comp_contents.items()
    }

    # --- Plot ---
    hep.style.use("CMS")
    fig, (ax, rax) = plt.subplots(
        2, 1,
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1),
        sharex=True,
    )

    colors = {
        "DYJets": "#1f77b4",
        "tt_tW": "#e41a1c",
        "Other": "#4daf4a",
        "Nonprompt": "#ff7f0e",
    }
    # Plot fractions as stacked area
    bottom = np.zeros_like(bin_centers)
    for label in ["DYJets", "tt_tW", "Other", "Nonprompt"]:
        if label not in comp_fractions:
            continue
        frac = comp_fractions[label]
        ax.fill_between(
            bin_centers, bottom, bottom + frac,
            color=colors[label], alpha=0.7, label=label, step="mid",
        )
        bottom = bottom + frac

    ax.set_ylabel("Fraction of total")
    ax.set_ylim(0, 1.45)
    ax.set_xlim(mass_lo, mass_hi)
    ax.legend(loc="upper right", fontsize=16)

    ch_label = {"ee": "ee", "mumu": r"$\mu\mu$"}[channel]
    region_label = f"{ch_label}\n{topology.capitalize()} SR\n{era}"
    ax.text(
        0.05, 0.96, region_label,
        transform=ax.transAxes, fontsize=20, verticalalignment="top",
    )
    hep.cms.label(
        loc=0, ax=ax, data=False,
        label="Work in Progress",
        lumi=f"{lumi:.2f}", com=com, fontsize=20,
    )

    # Lower panel: absolute counts (log scale)
    for label in ["DYJets", "tt_tW", "Other", "Nonprompt"]:
        if label not in comp_contents:
            continue
        rax.step(
            bin_centers, comp_contents[label],
            where="mid", color=colors[label], linewidth=1.5, label=label,
        )
    rax.step(bin_centers, total_content, where="mid",
             color="black", linewidth=2, linestyle="--", label="Total")
    rax.set_yscale("log")
    rax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    rax.set_ylabel(f"Events / {bin_width_gev:.0f} GeV")
    rax.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    # Add headroom for the legend
    all_positive = total_content[total_content > 0]
    if len(all_positive) > 0:
        rax.set_ylim(None, all_positive.max() * 50)
    rax.legend(loc="upper right", fontsize=12, ncol=2)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved composition plot: %s (.pdf and .png)", output_path.stem)


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def extract_fit_results(
    fit_result: ROOT.RooFitResult,
    params: dict[str, ROOT.RooRealVar],
) -> dict:
    """Package fit output into a JSON-serializable dict."""
    param_names = sorted(params.keys())

    parameters = {}
    for name in param_names:
        p = params[name]
        parameters[name] = {
            "value": p.getVal(),
            "error": p.getError(),
            "error_lo": p.getErrorLo(),
            "error_hi": p.getErrorHi(),
            "range": [p.getMin(), p.getMax()],
        }

    # Build correlation matrix
    corr = []
    for ni in param_names:
        row = []
        for nj in param_names:
            row.append(fit_result.correlation(params[ni], params[nj]))
        corr.append(row)

    return {
        "fit_status": {
            "status": fit_result.status(),
            "edm": fit_result.edm(),
            "cov_quality": fit_result.covQual(),
            "min_nll": fit_result.minNll(),
        },
        "parameters": parameters,
        "correlation_matrix": corr,
    }


def save_results_json(results: dict, output_path: Path) -> None:
    """Write fit results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved fit results: %s", output_path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _extract_curve(roo_curve) -> tuple[np.ndarray, np.ndarray]:
    """Extract (x, y) arrays from a RooCurve on a RooPlot frame."""
    n = roo_curve.GetN()
    x = np.array([roo_curve.GetPointX(i) for i in range(n)])
    y = np.array([roo_curve.GetPointY(i) for i in range(n)])
    return x, y


def _extract_data_points(roo_hist) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (x, y, yerr_lo, yerr_hi) arrays from a RooHist on a RooPlot frame."""
    n = roo_hist.GetN()
    x = np.array([roo_hist.GetPointX(i) for i in range(n)])
    y = np.array([roo_hist.GetPointY(i) for i in range(n)])
    yerr_lo = np.array([roo_hist.GetErrorYlow(i) for i in range(n)])
    yerr_hi = np.array([roo_hist.GetErrorYhigh(i) for i in range(n)])
    return x, y, yerr_lo, yerr_hi


def plot_fit(
    observable: ROOT.RooRealVar,
    data: ROOT.RooDataHist,
    pdf: ROOT.RooAbsPdf,
    fit_result: ROOT.RooFitResult,
    params: dict[str, ROOT.RooRealVar],
    output_path: Path,
    *,
    lumi: float,
    com: float,
    channel: str,
    topology: str,
    era: str,
    model_label: str,
    bin_width_gev: float,
    y_range: tuple[float, float] | None = None,
) -> None:
    """Create a fit overlay plot with a pull panel using mplhep.

    Uses RooPlot internally to get properly normalized curves, then
    extracts arrays and plots with matplotlib for consistent style.

    Upper panel: data points + fit curve (+ components for double-exp).
    Lower panel: pull = (data - fit) / error per bin.
    Saves as both PDF and PNG.
    """
    # --- Build a RooPlot frame to get properly normalized arrays ---
    frame = observable.frame()
    data.plotOn(frame, ROOT.RooFit.Name("data"))
    pdf.plotOn(frame, ROOT.RooFit.Name("fit_curve"))

    # Extract data points
    data_x, data_y, data_yerr_lo, data_yerr_hi = _extract_data_points(
        frame.getHist("data")
    )

    # Extract fit curve (smooth, many points)
    fit_x, fit_y = _extract_curve(frame.getCurve("fit_curve"))

    # Compute component curves for double-exp using numpy.
    # Scale each component relative to the total fit curve so they share
    # the same normalization (events / bin width) as the plotted curve.
    components = {}
    if "n1" in params:
        c1_val = params["c1"].getVal()
        c2_val = params["c2"].getVal()
        n1_val = params["n1"].getVal()
        n2_val = params["n2"].getVal()
        total_unnorm = n1_val * np.exp(c1_val * fit_x) + n2_val * np.exp(c2_val * fit_x)
        # Avoid division by zero in regions where total is negligible
        safe_total = np.where(total_unnorm > 0, total_unnorm, 1.0)
        comp1_frac = (n1_val * np.exp(c1_val * fit_x)) / safe_total
        comp2_frac = (n2_val * np.exp(c2_val * fit_x)) / safe_total
        components["Steep component"] = (fit_x, fit_y * comp1_frac)
        components["Shallow component"] = (fit_x, fit_y * comp2_frac)

    # Extract pulls
    pull_hist = frame.pullHist("data", "fit_curve")
    pull_x, pull_y, pull_yerr_lo, pull_yerr_hi = _extract_data_points(pull_hist)

    # Chi2/ndf
    npar = len(params)
    chi2_ndf = frame.chiSquare("fit_curve", "data", npar)

    # --- Plot with mplhep (matching make_stackplots.py style) ---
    hep.style.use("CMS")
    fig, (ax, rax) = plt.subplots(
        2, 1,
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1),
        sharex=True,
    )

    # -- Upper panel: data + fit --
    ax.errorbar(
        data_x, data_y,
        yerr=[data_yerr_lo, data_yerr_hi],
        fmt="ko", markersize=5, capsize=0, linewidth=1.2,
        label="MC background", zorder=5,
    )
    ax.plot(fit_x, fit_y, color="#1f77b4", linewidth=2, label="Fit", zorder=4)

    comp_styles = [
        {"color": "#e41a1c", "linestyle": "--"},
        {"color": "#4daf4a", "linestyle": "--"},
    ]
    for (comp_label, (cx, cy)), style in zip(components.items(), comp_styles):
        ax.plot(cx, cy, linewidth=1.5, label=comp_label, zorder=3, **style)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylabel(f"Events / {bin_width_gev:.0f} GeV")
    ax.set_xlim(observable.getMin(), observable.getMax())

    # Y-axis limits: use explicit range if provided, otherwise auto-compute
    if y_range is not None:
        ax.set_ylim(*y_range)
    else:
        visible = data_y[data_y > 0]
        fit_positive = fit_y[fit_y > 0]
        y_lo = min(visible.min(), fit_positive.min()) * 0.3 if len(fit_positive) > 0 else visible.min() * 0.3
        if len(visible) > 0:
            ax.set_ylim(y_lo, 5000 * visible.max())

    ax.legend(loc="upper right", fontsize=18)

    # Region label — same style as make_stackplots.py (plotting_helpers.plot_stack)
    ch_label = {"ee": "ee", "mumu": r"$\mu\mu$"}[channel]
    region_label = f"{ch_label}\n{topology.capitalize()} SR\n{era}"
    ax.text(
        0.05, 0.96, region_label,
        transform=ax.transAxes, fontsize=20,
        verticalalignment="top",
    )

    hep.cms.label(
        loc=0, ax=ax, data=False,
        label="Work in Progress",
        lumi=f"{lumi:.2f}", com=com, fontsize=20,
    )

    # Fit info + parameters — right side, below the legend
    model_equations = {
        "Single exponential": r"$f(m) = e^{c \cdot m}$",
        "Double exponential": r"$f(m) = N_1 \cdot e^{c_1 \cdot m} + N_2 \cdot e^{c_2 \cdot m}$",
        "Power-law exp": r"$f(m) = m^{a} \cdot e^{c \cdot m}$",
    }
    # LaTeX names for parameters (match the equation notation)
    latex_names = {
        "c": "c", "c1": "c_1", "c2": "c_2", "a": "a",
        "n_bkg": "N_{bkg}", "n1": "N_1", "n2": "N_2",
    }
    fit_lines = [
        model_label + " Fit",
        model_equations.get(model_label, ""),
    ]
    for name in sorted(params.keys()):
        p = params[name]
        lname = latex_names.get(name, name)
        # Yield parameters are event counts — show as integers
        if name in ("n_bkg", "n1", "n2"):
            fit_lines.append(rf"${lname} = {p.getVal():.0f} \pm {p.getError():.0f}$")
        else:
            fit_lines.append(rf"${lname} = {p.getVal():.5f} \pm {p.getError():.5f}$")
    fit_lines.append(rf"$\chi^2 / \mathrm{{ndf}} = {chi2_ndf:.2f}$")
    ax.text(
        0.95, 0.81,
        "\n".join(fit_lines),
        transform=ax.transAxes, fontsize=18,
        verticalalignment="top", horizontalalignment="right",
        multialignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.8),
    )

    # -- Lower panel: pulls --
    rax.axhspan(-2, 2, color="gold", alpha=0.3, zorder=0)    # ±2σ band
    rax.axhspan(-1, 1, color="green", alpha=0.3, zorder=1)   # ±1σ band
    rax.errorbar(
        pull_x, pull_y,
        yerr=[pull_yerr_lo, pull_yerr_hi],
        fmt="ko", markersize=4, capsize=0, linewidth=1, zorder=2,
    )
    rax.axhline(0, color="gray", linestyle="--", linewidth=1)
    rax.set_ylabel("Pull", fontsize=20, loc="center")
    rax.set_ylim(-5, 5)
    rax.set_yticks([-4, -2, 0, 2, 4])
    rax.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    logger.info("Saved fit plot: %s (.pdf and .png)", output_path.stem)


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
    # For a single era, there's one input directory
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
        if model_name == "single-exp":
            pdf, params = build_single_exp_model(mass, n_data)
        elif model_name == "double-exp":
            pdf, params = build_double_exp_model(mass, n_data)
        elif model_name == "pow-exp":
            pdf, params = build_pow_exp_model(mass, n_data)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        logger.info("Model: %s (%d parameters)", model_name, len(params))

        # --- Component fits (optional) ---
        component_results = {}
        if args.component_fits:
            logger.info("=" * 60)
            logger.info("Running per-component single exponential fits...")
            logger.info("=" * 60)

            fit_kwargs = dict(
                observable=mass,
                input_dir=input_dir,
                hist_key=hist_key,
                rebin=args.rebin,
                output_dir=chan_output_dir,
                lumi=info["lumi"],
                com=info.get("com", 13.6),
                channel=channel,
                topology=args.topology,
                era=args.era,
                bin_width_gev=rebinned.GetBinWidth(1),
                y_range=global_y_range,
            )

            component_hists = {}
            for label in FIT_COMPONENTS:
                comp_res, comp_hist = fit_single_component(label=label, **fit_kwargs)
                component_results[label] = comp_res
                component_hists[label] = comp_hist

            # Load remaining components for the composition plot
            for label in ["Other", "Nonprompt"]:
                try:
                    h = load_single_background(input_dir, hist_key, COMPONENT_FILES[label])
                    component_hists[label] = h.Rebin(args.rebin, f"h_{label}_rebinned")
                except RuntimeError:
                    logger.warning("Could not load %s for composition plot", label)

            # Composition plot
            plot_background_composition(
                component_hists, rebinned, mass,
                chan_output_dir / f"composition_{channel}.pdf",
                lumi=info["lumi"],
                com=info.get("com", 13.6),
                channel=channel,
                topology=args.topology,
                era=args.era,
                bin_width_gev=rebinned.GetBinWidth(1),
            )

            # Composition JSON — per-bin fractions for each component
            mass_lo, mass_hi = args.mass_range
            comp_json = {"metadata": {
                "channel": channel,
                "topology": args.topology,
                "era": args.era,
                "mass_range": [mass_lo, mass_hi],
                "bin_width_gev": rebinned.GetBinWidth(1),
            }, "bins": []}
            for i in range(1, rebinned.GetNbinsX() + 1):
                center = rebinned.GetBinCenter(i)
                total = rebinned.GetBinContent(i)
                bin_entry = {"bin_center": center, "total": total, "fractions": {}}
                for lbl, ch in component_hists.items():
                    val = ch.GetBinContent(i)
                    bin_entry["fractions"][lbl] = {
                        "yield": val,
                        "fraction": val / total if total > 0 else 0.0,
                    }
                comp_json["bins"].append(bin_entry)
            save_results_json(comp_json, chan_output_dir / f"composition_{channel}.json")

            # Comparison summary
            logger.info("=" * 60)
            logger.info("Component fit comparison:")
            for label, res in component_results.items():
                if not res:
                    continue
                c_val = res["parameters"]["c"]["value"]
                c_err = res["parameters"]["c"]["error"]
                n_val = res["parameters"]["n_bkg"]["value"]
                n_err = res["parameters"]["n_bkg"]["error"]
                chi2 = res["goodness_of_fit"]["chi2_per_ndf"]
                logger.info(
                    "  %-10s  c = %.6f +/- %.6f  |  n_bkg = %5.0f +/- %3.0f  |  chi2/ndf = %.2f",
                    label, c_val, c_err, n_val, n_err, chi2,
                )

            # Identify steeper vs shallower
            fitted = {
                label: res["parameters"]["c"]["value"]
                for label, res in component_results.items()
                if res
            }
            if len(fitted) == 2:
                labels_sorted = sorted(fitted, key=lambda k: fitted[k])  # most negative first
                logger.info(
                    "  Suggestion: %s (c=%.6f) → c1 (steep),  %s (c=%.6f) → c2 (shallow)",
                    labels_sorted[0], fitted[labels_sorted[0]],
                    labels_sorted[1], fitted[labels_sorted[1]],
                )
            logger.info("=" * 60)

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

        if component_results:
            results["component_fits"] = component_results

        # Compute goodness of fit from the plot frame.
        # frame.chiSquare() returns chi2/ndf using the actual non-empty data bins,
        # NOT the frame's default binning. We count data bins from the RooHist.
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
                "single-exp": "Single exponential",
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
