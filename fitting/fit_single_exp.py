#!/usr/bin/env python3
"""
Fit summed MC background m_lljj distribution with a single exponential.

Usage examples:
    # Default (both channels, 800-6000 GeV, 200 GeV bins):
    python fitting/fit_single_exp.py --era RunIII2024Summer24 --dir 20260223_lo_dy

    # Custom mass range and rebinning:
    python fitting/fit_single_exp.py --era RunIII2024Summer24 --dir 20260223_lo_dy \
        --mass-range 1000 4000 --rebin 10

    # With per-component fits and composition plot:
    python fitting/fit_single_exp.py --era RunIII2024Summer24 --dir 20260223_lo_dy \
        --component-fits
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

from wrplotter.cli_utils import setup_logging
from wrplotter.config import load_lumi
from wrplotter.paths import input_dirs_for_era, repo_root
from wrplotter.plotting_helpers import custom_log_formatter

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
    BACKGROUND_FILES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_single_exp_model(
    observable: ROOT.RooRealVar,
    n_data: float,
) -> tuple[ROOT.RooExtendPdf, dict[str, ROOT.RooRealVar]]:
    """Build N_bkg * exp(c * m) as an extended PDF.

    RooExponential provides the shape (normalized over the observable range),
    and RooExtendPdf adds a Poisson term for the total event count so the
    likelihood constrains both the shape and the normalization.

    Initial value -0.003 is estimated from the data: the background falls
    from ~420/100GeV at 800 GeV to ~12/100GeV at 2000 GeV,
    giving ln(12/420) / 1200 ~ -0.003.
    """
    c = ROOT.RooRealVar("c", "exponential slope", -0.003, -0.01, -0.0001)
    n_bkg = ROOT.RooRealVar("n_bkg", "background yield", n_data, 0, 50000)

    shape = ROOT.RooExponential("single_exp", "single exponential", observable, c)
    ROOT.SetOwnership(shape, False)  # prevent Python GC; C++ RooExtendPdf holds a ref

    pdf = ROOT.RooExtendPdf("ext_single_exp", "extended single exponential", shape, n_bkg)

    return pdf, {"c": c, "n_bkg": n_bkg}


# ---------------------------------------------------------------------------
# Component fits
# ---------------------------------------------------------------------------

# Map from short labels to ROOT file names
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


# ---------------------------------------------------------------------------
# Composition plot
# ---------------------------------------------------------------------------

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
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the m_lljj background distribution with a single exponential."
    )
    add_common_args(parser)

    parser.add_argument(
        "--component-fits", action="store_true",
        help="Fit DYJets and tt+tW individually with single exponentials, "
             "and plot background composition vs mass.",
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

    base_output_dir = Path(
        args.output_dir
        or str(repo_root() / "fitting" / "outputs" / args.era / "single_exp")
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
        pdf, params = build_single_exp_model(mass, n_data)
        logger.info("Model: single-exp (%d parameters)", len(params))

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
                "model": "single-exp",
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
            model_label="Single exponential",
            bin_width_gev=rebinned.GetBinWidth(1),
            y_range=global_y_range,
        )

        # --- Summary ---
        logger.info("=" * 60)
        logger.info("Fit summary for single-exp in %s %s SR:", channel, args.topology)
        logger.info("  Status: %d (0 = converged)", results["fit_status"]["status"])
        logger.info("  chi2/ndf: %.2f (%d dof)", chi2_ndf, ndf)
        for name, p in results["parameters"].items():
            logger.info("  %s = %.6f +/- %.6f", name, p["value"], p["error"])
        logger.info("  Output: %s", chan_output_dir)
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
