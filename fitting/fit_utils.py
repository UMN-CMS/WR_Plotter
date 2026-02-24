"""Shared infrastructure for background fitting scripts.

Provides histogram loading, RooFit helpers, result extraction, and plotting
functions used by all model-specific fitting scripts (fit_single_exp.py, etc.).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
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

from wrplotter.cli_utils import add_era_args
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

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments shared across all fitting scripts."""
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
        help="Output directory for plots and JSON (default: fitting/outputs/<era>/<model>)",
    )


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
    rax.axhspan(-2, 2, color="gold", alpha=0.3, zorder=0)    # +/-2sigma band
    rax.axhspan(-1, 1, color="green", alpha=0.3, zorder=1)   # +/-1sigma band
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
