"""Shared infrastructure for fitting_v2 background fitting scripts.

Provides histogram loading, RooFit helpers, result extraction, and plotting
functions used by all fitting_v2 scripts.
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
    """Add CLI arguments shared across all fitting_v2 scripts."""
    add_era_args(parser, required=True)

    parser.add_argument(
        "--signal", type=str, default=None,
        metavar="TAG",
        help="Signal tag, e.g. 'WR2000_N1100'. "
             "Loads WRAnalyzer_signal_<TAG>.root from the input directory.",
    )
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
        "--rebin", type=int, default=2,
        help="Rebin factor — merge N original 10-GeV bins (default: 2 → 20 GeV/bin)",
    )
    parser.add_argument(
        "--n-sigma", type=float, default=3.0,
        help="Half-width of fitting window in units of sigma (default: 3.0)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for plots and JSON (default: fitting_v2/outputs/<era>/<stage>)",
    )
    parser.add_argument(
        "--box-left", action="store_true", default=False,
        help="Place the fit-info textbox on the left side of the plot",
    )


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------

def build_region_name(channel: str, topology: str) -> str:
    """Map (channel, topology) to the ROOT histogram region directory name."""
    return f"wr_{channel}_{topology}_sr"


def build_hist_key(region: str, variable: str = "mass_fourobject") -> str:
    """Construct the full TH1 key inside the ROOT file.

    Naming convention: {region}/{variable}_{region}
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


def load_signal(
    input_dir: Path,
    hist_key: str,
    signal_tag: str,
) -> ROOT.TH1D:
    """Load signal histogram for a given signal tag.

    Args:
        input_dir: Directory containing ROOT files.
        hist_key: Full histogram path inside the ROOT file.
        signal_tag: e.g. 'WR2000_N1100' → loads WRAnalyzer_signal_WR2000_N1100.root
    """
    fname = f"WRAnalyzer_signal_{signal_tag}.root"
    return load_single_background(input_dir, hist_key, fname)


# ---------------------------------------------------------------------------
# RooFit data
# ---------------------------------------------------------------------------

def create_roodatahist(
    hist: ROOT.TH1D,
    observable: ROOT.RooRealVar,
    name: str = "data_hist",
) -> ROOT.RooDataHist:
    """Convert a TH1D into a RooDataHist for use in RooFit.

    RooDataHist imports the bin contents and errors from the TH1.
    Only bins within the observable's range are included in the fit.
    """
    return ROOT.RooDataHist(name, name, ROOT.RooArgList(observable), hist)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_single_exp_model(
    observable: ROOT.RooRealVar,
    n_data: float,
) -> tuple[ROOT.RooExtendPdf, dict[str, ROOT.RooRealVar]]:
    """Build N_bkg * exp(c * m) as an extended PDF."""
    c = ROOT.RooRealVar("c", "exponential slope", -0.003, -0.01, -0.0001)
    n_bkg = ROOT.RooRealVar("n_bkg", "background yield", n_data, 0, 50000)

    shape = ROOT.RooExponential("single_exp", "single exponential", observable, c)
    ROOT.SetOwnership(shape, False)

    pdf = ROOT.RooExtendPdf(
        "ext_single_exp", "extended single exponential", shape, n_bkg,
    )
    return pdf, {"c": c, "n_bkg": n_bkg}


def build_double_exp_model(
    observable: ROOT.RooRealVar,
    n_data: float,
) -> tuple[ROOT.RooAddPdf, dict[str, ROOT.RooRealVar]]:
    """Build N1 * exp(c1*m) + N2 * exp(c2*m) as an extended PDF.

    Two RooExponential shapes, each wrapped in RooExtendPdf with its own
    yield parameter (N1, N2), combined via RooAddPdf.  The extended
    likelihood constrains both shapes and yields simultaneously.

    Initial values informed by per-component single-exp fits:
      tt+tW  -> c ~ -0.0032, ~70% of events  (steep)
      DYJets -> c ~ -0.0024, ~30% of events  (shallow)
    """
    c1 = ROOT.RooRealVar("c1", "steep slope", -0.0032, -0.02, 0)
    c2 = ROOT.RooRealVar("c2", "shallow slope", -0.0024, -0.02, 0)
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


def run_fit(
    pdf: ROOT.RooAbsPdf,
    data: ROOT.RooDataHist,
    params: dict[str, ROOT.RooRealVar],
    strategy: int = 1,
    fit_range: str | None = None,
    print_level: int = 1,
    use_chi2: bool = False,
) -> ROOT.RooFitResult:
    """Perform a binned fit to a RooDataHist.

    Parameters
    ----------
    use_chi2 : bool
        If False (default), minimize the binned extended negative
        log-likelihood (``fitTo``).  If True, minimize a Neyman
        chi-squared using observed bin errors (``chi2FitTo`` with
        ``DataError(RooAbsData::SumW2)``), which matches the scipy
        least-squares cross-check.

    Other key RooFit options:
    - Save(True): return the RooFitResult object with status and correlations
    - Strategy(1): default Minuit2 strategy (balance of speed and accuracy)
    - PrintLevel: Minuit console output level (1=moderate, -1=silent)
    - Range(fit_range): restrict fit to named ranges (e.g. "loSB,hiSB")
    """
    fit_opts = [
        ROOT.RooFit.Save(True),
        ROOT.RooFit.Strategy(strategy),
        ROOT.RooFit.PrintLevel(print_level),
    ]
    if fit_range is not None:
        fit_opts.append(ROOT.RooFit.Range(fit_range))

    if use_chi2:
        # Neyman chi2: sum (obs - exp)^2 / sigma_obs^2, matching scipy
        fit_opts.append(ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2))
        result = pdf.chi2FitTo(data, *fit_opts)
    else:
        result = pdf.fitTo(data, *fit_opts)

    logger.info("Fit status: %d (0 = converged)", result.status())
    logger.info("EDM: %.2e", result.edm())
    logger.info("Covariance quality: %d (3 = full accurate)", result.covQual())

    # If fit failed, retry with more thorough strategy
    if result.status() != 0 and strategy < 2:
        logger.warning(
            "Fit did not converge (status=%d). Retrying with strategy=2...",
            result.status(),
        )
        retry_opts = [
            ROOT.RooFit.Save(True),
            ROOT.RooFit.Strategy(2),
            ROOT.RooFit.PrintLevel(print_level),
        ]
        if fit_range is not None:
            retry_opts.append(ROOT.RooFit.Range(fit_range))
        if use_chi2:
            retry_opts.append(ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2))
            result = pdf.chi2FitTo(data, *retry_opts)
        else:
            result = pdf.fitTo(data, *retry_opts)
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
    region_type: str = "SR",
    data_label: str = "MC signal",
    blinding_window: tuple[float, float] | None = None,
    norm_range: str | None = None,
    scipy_curve: tuple[np.ndarray, np.ndarray] | None = None,
    bkg_curve: tuple[np.ndarray, np.ndarray] | None = None,
    chi2_ndf_override: float | None = None,
    x_range: tuple[float, float] | None = None,
    extra_data: ROOT.TH1 | None = None,
    box_left: bool = False,
) -> None:
    """Create a fit overlay plot with a pull panel using mplhep.

    Uses RooPlot internally to get properly normalized curves, then
    extracts arrays and plots with matplotlib for consistent style.

    Upper panel: data points + fit curve.
    Lower panel: pull = (data - fit) / error per bin.
    If blinding_window is provided, shades the blinded region on both panels.
    If norm_range is provided (e.g. "loSB,hiSB"), normalizes the PDF curve
    to the fit range but draws it over the full observable range (extrapolation).
    Saves as both PDF and PNG.
    """
    # --- Build a RooPlot frame to get properly normalized arrays ---
    frame = observable.frame()
    data.plotOn(frame, ROOT.RooFit.Name("data"))
    if norm_range is not None:
        observable.setRange("full", observable.getMin(), observable.getMax())
        pdf.plotOn(
            frame,
            ROOT.RooFit.Name("fit_curve"),
            ROOT.RooFit.NormRange(norm_range),
            ROOT.RooFit.Range("full"),
        )
    else:
        pdf.plotOn(frame, ROOT.RooFit.Name("fit_curve"))

    # Extract data points, filtering out empty bins (which may have
    # artificially large errors from the chi2FitTo empty-bin workaround).
    data_x, data_y, data_yerr_lo, data_yerr_hi = _extract_data_points(
        frame.getHist("data")
    )
    data_mask = data_y > 0
    data_x = data_x[data_mask]
    data_y = data_y[data_mask]
    data_yerr_lo = data_yerr_lo[data_mask]
    data_yerr_hi = data_yerr_hi[data_mask]

    # Extract fit curve (smooth, many points)
    fit_x, fit_y = _extract_curve(frame.getCurve("fit_curve"))

    # Extract pulls (same mask as data)
    pull_hist = frame.pullHist("data", "fit_curve")
    pull_x, pull_y, pull_yerr_lo, pull_yerr_hi = _extract_data_points(pull_hist)
    pull_x = pull_x[data_mask]
    pull_y = pull_y[data_mask]
    pull_yerr_lo = pull_yerr_lo[data_mask]
    pull_yerr_hi = pull_yerr_hi[data_mask]

    # Chi2/ndf — use the caller's value if provided, otherwise fall back
    # to the RooPlot estimate (which includes all bins and uses RooHist errors).
    if chi2_ndf_override is not None:
        chi2_ndf = chi2_ndf_override
    else:
        npar = len(params)
        chi2_ndf = frame.chiSquare("fit_curve", "data", npar)

    # --- Plot with mplhep ---
    hep.style.use("CMS")
    fig, (ax, rax) = plt.subplots(
        2, 1,
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1),
        sharex=True,
    )

    # -- Upper panel: data + fit --
    # Extra data points outside the fit range (from full histogram)
    if extra_data is not None:
        fit_lo, fit_hi = observable.getMin(), observable.getMax()
        ex_x, ex_y, ex_err = [], [], []
        for i in range(1, extra_data.GetNbinsX() + 1):
            c = extra_data.GetBinCenter(i)
            if c < fit_lo or c > fit_hi:
                y = extra_data.GetBinContent(i)
                if y > 0:
                    ex_x.append(c)
                    ex_y.append(y)
                    ex_err.append(extra_data.GetBinError(i))
        if ex_x:
            ax.errorbar(
                ex_x, ex_y, yerr=ex_err,
                fmt="o", color="gray", markersize=4, capsize=0, linewidth=1,
                zorder=4,
            )

    ax.errorbar(
        data_x, data_y,
        yerr=[data_yerr_lo, data_yerr_hi],
        fmt="ko", markersize=5, capsize=0, linewidth=1.2,
        label=data_label, zorder=5,
    )
    ax.plot(fit_x, fit_y, color="#1f77b4", linewidth=2, label="RooFit", zorder=4)

    # Background component curve (for S+B fits)
    if bkg_curve is not None:
        bkg_x, bkg_y = bkg_curve
        ax.plot(bkg_x, bkg_y, color="#ff7f0e", linewidth=2, linestyle="--",
                label="Background", zorder=3)

    # Scipy curve overlay
    if scipy_curve is not None:
        sp_x, sp_y = scipy_curve
        ax.plot(sp_x, sp_y, color="#d62728", linewidth=2, linestyle="--",
                label="Scipy", zorder=3)

    # Shaded blinding window
    if blinding_window is not None:
        blo, bhi = blinding_window
        ax.axvspan(blo, bhi, color="gray", alpha=0.25, zorder=2, label="Fit region")
        rax.axvspan(blo, bhi, color="gray", alpha=0.25, zorder=0)

    ax.set_ylabel(f"Events / {bin_width_gev:.0f} GeV")
    if x_range is not None:
        ax.set_xlim(*x_range)
    else:
        ax.set_xlim(observable.getMin(), observable.getMax())

    if y_range is not None:
        # Linear scale if lower bound is 0, otherwise log
        if y_range[0] > 0:
            ax.set_yscale("log")
        ax.set_ylim(*y_range)
    else:
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 1e4)

    ax.legend(loc="upper right", fontsize=18)

    ch_labels = {"ee": "ee", "mumu": r"$\mu\mu$", "emu": r"$e\mu$"}
    ch_label = ch_labels.get(channel, channel)
    region_label = f"{ch_label}\n{topology.capitalize()} {region_type}\n{era}"
    ax.text(
        0.05, 0.96, region_label,
        transform=ax.transAxes, fontsize=20,
        verticalalignment="top",
    )

    hep.cms.label(
        loc=0, ax=ax, data=False,
        label="Work in Progress",
        com=com, fontsize=20,
    )

    # Parameter box
    model_equations = {
        "Gaussian": r"$f(m) = \mathcal{N}(\mu, \sigma)$",
        "Crystal Ball": r"$f(m) = \mathrm{CB}(\mu, \sigma, \alpha, n)$",
        "Voigtian": r"$f(m) = \mathrm{BW}(\mu, \Gamma) \otimes \mathcal{N}(\sigma)$",
        "Breit-Wigner": r"$f(m) \propto 1/((m^2-\mu^2)^2 + \mu^2\Gamma^2)$",
        "Bifurcated Gaussian": r"$f(m) = \mathcal{N}(\mu,\sigma_L\,|\,\sigma_R)$",
        "Single exponential": r"$f(m) = N_{\mathrm{bkg}} \cdot e^{c \cdot m}$",
        "Double exponential": r"$f(m) = N_1 e^{c_1 m} + N_2 e^{c_2 m}$",
        "S+B": r"$f(m) = N_{\mathrm{bkg}} \cdot e^{c \cdot m} + N_{\mathrm{sig}} \cdot S(m)$",
    }
    latex_names = {
        "mean": r"\mu",
        "sigma": r"\sigma",
        "sigma_lo": r"\sigma_L",
        "sigma_hi": r"\sigma_R",
        "width": r"\Gamma",
        "alpha": r"\alpha",
        "cb_n": r"n",
        "c": r"c",
        "c1": r"c_1",
        "c2": r"c_2",
        "n1": r"N_1",
        "n2": r"N_2",
        "mu": r"\mu",
        "n_sig": r"N_{\mathrm{sig}}",
        "n_bkg": r"N_{\mathrm{bkg}}",
    }
    fit_lines = [
        model_label + " Fit",
        model_equations.get(model_label, ""),
    ]
    for name in sorted(params.keys()):
        p = params[name]
        lname = latex_names.get(name, name)
        if name.startswith("n_"):
            fit_lines.append(rf"${lname} = {p.getVal():.0f} \pm {p.getError():.0f}$")
        elif name in ("mean", "sigma", "sigma_lo", "sigma_hi", "width"):
            fit_lines.append(rf"${lname} = {p.getVal():.1f} \pm {p.getError():.1f}$")
        else:
            fit_lines.append(rf"${lname} = {p.getVal():.3f} \pm {p.getError():.3f}$")
    fit_lines.append(rf"$\chi^2 / \mathrm{{ndf}} = {chi2_ndf:.2f}$")
    if blinding_window is not None:
        blo_val, bhi_val = blinding_window
        fit_lines.append(rf"Window: $[{blo_val:.0f},\, {bhi_val:.0f}]$ GeV")
    if box_left:
        box_x, box_y = (0.05, 0.65)
        box_ha = "left"
    else:
        box_x, box_y = (0.95, 0.65)
        box_ha = "right"
    ax.text(
        box_x, box_y,
        "\n".join(fit_lines),
        transform=ax.transAxes, fontsize=16,
        verticalalignment="top", horizontalalignment=box_ha,
        multialignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.8),
    )

    # -- Lower panel: pulls --
    rax.axhspan(-2, 2, color="gold", alpha=0.3, zorder=0)
    rax.axhspan(-1, 1, color="green", alpha=0.3, zorder=1)
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
