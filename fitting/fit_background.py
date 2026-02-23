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
        "--channel", choices=["ee", "mumu"], default="ee",
        help="Lepton channel (default: ee)",
    )
    parser.add_argument(
        "--topology", choices=["resolved", "boosted"], default="resolved",
        help="Event topology (default: resolved)",
    )
    parser.add_argument(
        "--model", choices=["single-exp", "double-exp"], default="single-exp",
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
) -> tuple[ROOT.RooExponential, dict[str, ROOT.RooRealVar]]:
    """v0: Build f(m) = exp(c * m).

    RooExponential is normalized over the observable's range, so the pdf is:
        pdf(m) = c * exp(c * m) / (exp(c*max) - exp(c*min))
    The slope c must be negative for a falling distribution.

    Initial value -0.003 is estimated from the data: the background falls
    from ~420/100GeV at 800 GeV to ~12/100GeV at 2000 GeV,
    giving ln(12/420) / 1200 ≈ -0.003.
    """
    c = ROOT.RooRealVar("c", "exponential slope", -0.003, -0.01, -0.0001)

    pdf = ROOT.RooExponential("single_exp", "single exponential", observable, c)

    return pdf, {"c": c}


def build_double_exp_model(
    observable: ROOT.RooRealVar,
) -> tuple[ROOT.RooAddPdf, dict[str, ROOT.RooRealVar]]:
    """v1: Build f(m) = frac * exp(c1*m) + (1-frac) * exp(c2*m).

    Uses RooAddPdf to combine two RooExponentials. When one fraction is
    given for two pdfs, RooAddPdf automatically sets the second = 1-frac.

    c1 is the "steep" component (more negative, dominates at lower masses).
    c2 is the "shallow" component (less negative, dominates the tail).
    The parameter ranges are deliberately non-overlapping to prevent
    Minuit from swapping the two components (a common degeneracy issue).
    """
    c1 = ROOT.RooRealVar("c1", "steep slope", -0.005, -0.02, -0.002)
    c2 = ROOT.RooRealVar("c2", "shallow slope", -0.001, -0.002, -0.00005)
    frac = ROOT.RooRealVar("frac", "fraction of steep component", 0.7, 0.01, 0.99)

    exp1 = ROOT.RooExponential("exp1", "steep component", observable, c1)
    exp2 = ROOT.RooExponential("exp2", "shallow component", observable, c2)

    # RooAddPdf with one fraction: pdf = frac*exp1 + (1-frac)*exp2
    pdf = ROOT.RooAddPdf(
        "double_exp", "double exponential",
        ROOT.RooArgList(exp1, exp2),
        ROOT.RooArgList(frac),
    )

    return pdf, {"c1": c1, "c2": c2, "frac": frac}


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
    - Minimizer("Minuit2"): more robust than classic Minuit for multi-param fits
    - SumW2Error(True): correct errors for weighted MC (uses sum-of-weights-squared
      instead of assuming Poisson statistics — essential for MC fits)
    - PrintLevel(1): moderate console output from Minuit
    """
    result = pdf.fitTo(
        data,
        ROOT.RooFit.Save(True),
        ROOT.RooFit.Strategy(strategy),
        ROOT.RooFit.Minimizer("Minuit2", "Migrad"),
        ROOT.RooFit.SumW2Error(True),
        ROOT.RooFit.PrintLevel(1),
    )

    logger.info("Fit status: %d (0 = converged)", result.status())
    logger.info("EDM: %.2e", result.edm())
    logger.info("Covariance quality: %d (3 = full accurate)", result.covQual())

    # EDM can be inflated when using SumW2Error (corrected Hessian for weighted
    # data). A high EDM with status=0 and covQual=3 is usually not a real problem,
    # but we flag it so it's visible in the logs.
    if result.edm() > 1.0:
        logger.warning(
            "EDM = %.2e is elevated (> 1). This is common with SumW2Error on "
            "weighted MC. Check that status=0 and covQual=3.",
            result.edm(),
        )

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
            ROOT.RooFit.Minimizer("Minuit2", "Migrad"),
            ROOT.RooFit.SumW2Error(True),
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
      1. Fix the shallow component, fit only the steep slope
      2. Release all parameters and do the full fit
      3. If that fails, retry with the most thorough strategy
    """
    logger.info("Stage 1: fitting steep component only (c2 and frac fixed)...")
    params["c2"].setConstant(True)
    params["frac"].setConstant(True)
    pdf.fitTo(
        data,
        ROOT.RooFit.Strategy(0),
        ROOT.RooFit.Minimizer("Minuit2", "Migrad"),
        ROOT.RooFit.SumW2Error(True),
        ROOT.RooFit.PrintLevel(-1),
    )
    logger.info("  c1 after stage 1: %.6f", params["c1"].getVal())

    logger.info("Stage 2: releasing all parameters for full fit...")
    params["c2"].setConstant(False)
    params["frac"].setConstant(False)
    result = run_fit(pdf, data, params, strategy=1)

    return result


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

    if "c1" in params:
        pdf.plotOn(frame, ROOT.RooFit.Components("exp1"), ROOT.RooFit.Name("comp1"))
        pdf.plotOn(frame, ROOT.RooFit.Components("exp2"), ROOT.RooFit.Name("comp2"))

    # Extract data points
    data_x, data_y, data_yerr_lo, data_yerr_hi = _extract_data_points(
        frame.getHist("data")
    )

    # Extract fit curve (smooth, many points)
    fit_x, fit_y = _extract_curve(frame.getCurve("fit_curve"))

    # Extract components if present
    components = {}
    if "c1" in params:
        components["Steep component"] = _extract_curve(frame.getCurve("comp1"))
        components["Shallow component"] = _extract_curve(frame.getCurve("comp2"))

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

    # Auto y-limits: use fit curve minimum so the full fit is visible
    visible = data_y[data_y > 0]
    fit_positive = fit_y[fit_y > 0]
    y_lo = min(visible.min(), fit_positive.min()) * 0.3 if len(fit_positive) > 0 else visible.min() * 0.3
    if len(visible) > 0:
        ax.set_ylim(y_lo, 200 * visible.max())

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
        "Double exponential": r"$f(m) = f \cdot e^{c_1 \cdot m} + (1-f) \cdot e^{c_2 \cdot m}$",
    }
    # LaTeX names for parameters (match the equation notation)
    latex_names = {"c": "c", "c1": "c_1", "c2": "c_2", "frac": "f"}
    fit_lines = [
        model_label + " Fit",
        model_equations.get(model_label, ""),
    ]
    for name in sorted(params.keys()):
        p = params[name]
        lname = latex_names.get(name, name)
        fit_lines.append(rf"${lname} = {p.getVal():.5f} \pm {p.getError():.5f}$")
    fit_lines.append(rf"$\chi^2 / \mathrm{{ndf}} = {chi2_ndf:.2f}$")
    ax.text(
        0.95, 0.75,
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

    output_dir = Path(
        args.output_dir or str(repo_root() / "fitting" / "outputs" / args.era)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build histogram path ---
    region = build_region_name(args.channel, args.topology)
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

    # --- Create RooDataHist ---
    data_hist = create_roodatahist(rebinned, mass)
    logger.info("RooDataHist entries: %.1f (in fit range)", data_hist.sumEntries())

    # --- Build model ---
    model_name = args.model
    if model_name == "single-exp":
        pdf, params = build_single_exp_model(mass)
    elif model_name == "double-exp":
        pdf, params = build_double_exp_model(mass)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    logger.info("Model: %s (%d parameters)", model_name, len(params))

    # --- Fit ---
    if model_name == "double-exp":
        fit_result = run_staged_double_exp_fit(pdf, data_hist, params)
    else:
        fit_result = run_fit(pdf, data_hist, params)

    # --- Extract and save results ---
    results = {
        "metadata": {
            "era": args.era,
            "channel": args.channel,
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

    slug = model_name.replace("-", "_")
    json_path = output_dir / f"fit_{slug}.json"
    save_results_json(results, json_path)

    # --- Plot ---
    plot_path = output_dir / f"fit_{slug}.pdf"
    plot_fit(
        mass, data_hist, pdf, fit_result, params, plot_path,
        lumi=info["lumi"],
        com=info.get("com", 13.6),
        channel=args.channel,
        topology=args.topology,
        era=args.era,
        model_label={
            "single-exp": "Single exponential",
            "double-exp": "Double exponential",
        }[model_name],
        bin_width_gev=rebinned.GetBinWidth(1),
    )

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("Fit summary for %s in %s %s SR:", model_name, args.channel, args.topology)
    logger.info("  Status: %d (0 = converged)", results["fit_status"]["status"])
    logger.info("  chi2/ndf: %.2f (%d dof)", chi2_ndf, ndf)
    for name, p in results["parameters"].items():
        logger.info("  %s = %.6f +/- %.6f", name, p["value"], p["error"])
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
