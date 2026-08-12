#!/usr/bin/env python3
"""
Background fit in the signal window defined by signal_window.py.

Fits an analytic background model to MC background m_lljj in the full
fitting window (no sidebands, no blinding). The window is derived from
the signal MC histogram RMS via signal_window.py.

Two modes:
  - Background-only (default): fit bkg model to bkg MC
  - S+B (--inject-signal): inject signal MC into pseudodata, fit bkg + signal

Supports single-exp and double-exp background models via --model.

Usage:
    # Background-only:
    python fitting_v2/fit_background.py \\
        --era RunIII2024Summer24 \\
        --dir 20260223_nlo_dy \\
        --signal WR2000_N1100

    # S+B with full signal injection (mu=1):
    python fitting_v2/fit_background.py \\
        --era RunIII2024Summer24 \\
        --dir 20260223_nlo_dy \\
        --signal WR2000_N1100 \\
        --inject-signal

    # S+B with partial injection (mu=0.1):
    python fitting_v2/fit_background.py \\
        --era RunIII2024Summer24 \\
        --dir 20260223_nlo_dy \\
        --signal WR2000_N1100 \\
        --inject-signal --inject-mu 0.1

    # Double-exp background model:
    python fitting_v2/fit_background.py \\
        --era RunIII2024Summer24 \\
        --dir 20260223_nlo_dy \\
        --signal WR2000_N1100 \\
        --model double-exp
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

# Add repo root to path so wrplotter and fitting_v2 imports work
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

from fitting_v2.fit_utils import (
    _extract_curve,
    add_common_args,
    build_region_name,
    build_hist_key,
    load_and_sum_backgrounds,
    load_signal,
    create_roodatahist,
    build_single_exp_model,
    build_double_exp_model,
    run_fit,
    extract_fit_results,
    save_results_json,
    plot_fit,
)

logger = logging.getLogger(__name__)

_MODEL_LABELS = {
    "single-exp": "Single exponential",
    "double-exp": "Double exponential",
}

# RooFit component names used for plotting the background piece of S+B
_BKG_COMPONENT_NAMES = {
    "single-exp": "ext_single_exp",
    "double-exp": "double_exp",
}


# ---------------------------------------------------------------------------
# Signal window JSON helpers
# ---------------------------------------------------------------------------

def discover_window_json(
    era: str,
    signal_tag: str,
    channel: str,
) -> Path:
    """Auto-resolve path to signal_window.py output JSON."""
    json_path = (
        repo_root() / "fitting_v2" / "outputs" / era
        / "signal_window" / signal_tag / f"window_{channel}.json"
    )
    if not json_path.exists():
        raise FileNotFoundError(
            f"Signal window JSON not found: {json_path}\n"
            "Run signal_window.py first, or pass --signal-json explicitly."
        )
    return json_path


def load_window_json(json_path: Path) -> dict:
    """Load and validate signal window JSON from signal_window.py."""
    with open(json_path) as f:
        data = json.load(f)

    # Validate required structure
    for key in ("signal_stats", "fit_window"):
        if key not in data:
            raise ValueError(f"Signal window JSON missing '{key}': {json_path}")

    stats = data["signal_stats"]
    for key in ("mean", "rms"):
        if key not in stats:
            raise ValueError(
                f"Signal window JSON missing signal_stats.{key}: {json_path}"
            )

    window = data["fit_window"]
    for key in ("lo", "hi", "n_sigma"):
        if key not in window:
            raise ValueError(
                f"Signal window JSON missing fit_window.{key}: {json_path}"
            )

    return data


# ---------------------------------------------------------------------------
# S+B model builder
# ---------------------------------------------------------------------------

def build_sb_model(
    observable: ROOT.RooRealVar,
    bkg_model: str,
    n_bkg_data: float,
    sig_hist: ROOT.TH1D,
    inject_mu: float,
) -> tuple[ROOT.RooAddPdf, dict[str, ROOT.RooRealVar], float]:
    """Build signal+background model.

    Returns (sb_pdf, all_params, n_sig_expected) where:
    - sb_pdf: RooAddPdf combining background + signal
    - all_params: dict of all free RooRealVars (bkg params + n_sig)
    - n_sig_expected: signal events in the window at mu=1 (for mu calculation)
    """
    # --- Background component ---
    model_builders = {
        "single-exp": build_single_exp_model,
        "double-exp": build_double_exp_model,
    }
    bkg_pdf, bkg_params = model_builders[bkg_model](observable, n_bkg_data)
    # Prevent Python GC — RooAddPdf will hold a reference
    ROOT.SetOwnership(bkg_pdf, False)

    # --- Signal template from MC histogram ---
    sig_data_hist = create_roodatahist(sig_hist, observable, name="sig_template")
    n_sig_expected = sig_data_hist.sumEntries()
    # Prevent Python GC — RooHistPdf reads from this
    ROOT.SetOwnership(sig_data_hist, False)

    sig_pdf = ROOT.RooHistPdf(
        "sig_pdf", "signal template",
        ROOT.RooArgSet(observable), sig_data_hist,
    )
    ROOT.SetOwnership(sig_pdf, False)

    # Signal yield — initialize at injected value, allow negative for closure
    n_sig = ROOT.RooRealVar(
        "n_sig", "signal yield",
        inject_mu * n_sig_expected,
        -5 * n_sig_expected,
        5 * n_sig_expected,
    )
    ext_sig = ROOT.RooExtendPdf("ext_sig", "extended signal", sig_pdf, n_sig)
    ROOT.SetOwnership(ext_sig, False)

    # --- Combine ---
    sb_pdf = ROOT.RooAddPdf(
        "sb_model", "S+B model",
        ROOT.RooArgList(bkg_pdf, ext_sig),
    )

    all_params = {**bkg_params, "n_sig": n_sig}
    return sb_pdf, all_params, n_sig_expected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Background (or S+B) fit in the signal window."
        )
    )
    add_common_args(parser)

    parser.add_argument(
        "--signal-json", type=str, default=None,
        help="Explicit path to signal window JSON. "
             "Overrides auto-discovery from --signal.",
    )
    parser.add_argument(
        "--model",
        choices=["single-exp", "double-exp"],
        default="single-exp",
        help="Background model (default: single-exp)",
    )
    parser.add_argument(
        "--inject-signal", action="store_true",
        help="Activate S+B mode: inject signal MC into pseudodata "
             "and fit a signal+background model.",
    )
    parser.add_argument(
        "--inject-mu", type=float, default=1.0,
        help="Signal strength multiplier for injection (default: 1.0). "
             "Only used with --inject-signal.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if not args.signal and not args.signal_json:
        sys.exit(
            "ERROR: --signal or --signal-json is required.\n"
            "  Example: --signal WR2000_N1100"
        )

    ROOT.gROOT.SetBatch(True)
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

    sb_mode = args.inject_signal
    mode_label = "S+B" if sb_mode else "bkg-only"

    # --- Resolve input path ---
    info = load_lumi(args.era)
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    input_dir = input_dirs[0]
    logger.info("Input directory: %s", input_dir)

    # --- Output directory ---
    signal_tag = args.signal or "custom"
    stage_dir = "sb_injection" if sb_mode else "background"
    base_output_dir = Path(
        args.output_dir
        or str(repo_root() / "fitting_v2" / "outputs" / args.era / stage_dir / signal_tag)
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    channels = [args.channel] if args.channel else ["ee", "mumu"]

    for channel in channels:
        logger.info("=" * 60)
        logger.info(
            "Channel: %s  Model: %s  Mode: %s",
            channel, args.model, mode_label,
        )

        # --- Load signal window ---
        if args.signal_json:
            win_json_path = Path(args.signal_json)
        else:
            win_json_path = discover_window_json(args.era, args.signal, channel)
        logger.info("Signal window JSON: %s", win_json_path)

        win_data = load_window_json(win_json_path)
        sig_stats = win_data["signal_stats"]
        fit_window = win_data["fit_window"]

        sig_mean = sig_stats["mean"]
        sig_rms = sig_stats["rms"]
        fit_lo = fit_window["lo"]
        fit_hi = fit_window["hi"]
        win_n_sigma = fit_window["n_sigma"]

        logger.info(
            "Signal: mean=%.1f GeV, RMS=%.1f GeV",
            sig_mean, sig_rms,
        )
        logger.info(
            "Fit window (%.1f * RMS): [%.1f, %.1f] GeV",
            win_n_sigma, fit_lo, fit_hi,
        )

        # --- Build histogram key ---
        region = build_region_name(channel, args.topology)
        mass_var = "mass_twoobject" if args.topology == "boosted" else "mass_fourobject"
        hist_key = build_hist_key(region, mass_var)
        logger.info("Histogram key: %s", hist_key)

        # --- Load MC background ---
        bkg_hist = load_and_sum_backgrounds(input_dir, hist_key)

        # --- Rebin ---
        rebinned = bkg_hist.Rebin(args.rebin, f"h_bkg_{channel}_rebinned")
        bin_width_gev = rebinned.GetBinWidth(1)
        logger.info(
            "Rebinned: %d bins (factor %d, %.0f GeV/bin)",
            rebinned.GetNbinsX(), args.rebin, bin_width_gev,
        )

        # --- Load and prepare signal MC (needed for S+B mode) ---
        sig_rebinned = None
        n_sig_expected = 0.0
        if sb_mode:
            sig_hist_raw = load_signal(input_dir, hist_key, signal_tag)
            sig_rebinned = sig_hist_raw.Rebin(
                args.rebin, f"h_sig_{channel}_rebinned",
            )

        # --- Create pseudodata histogram ---
        if sb_mode:
            # bkg + inject_mu * signal
            pseudodata = rebinned.Clone(f"h_pseudodata_{channel}")
            pseudodata.Add(sig_rebinned, args.inject_mu)
            logger.info(
                "Pseudodata: bkg (%.1f) + %.2f * signal (%.1f) = %.1f events",
                rebinned.Integral(),
                args.inject_mu,
                sig_rebinned.Integral(),
                pseudodata.Integral(),
            )
            fit_hist = pseudodata
        else:
            fit_hist = rebinned

        # --- Define observable with fit window range ---
        mass = ROOT.RooRealVar("mass", r"m_{lljj}", fit_lo, fit_hi, "GeV")

        # --- Create RooDataHist ---
        data_hist = create_roodatahist(fit_hist, mass, name=f"data_{channel}")
        n_total = data_hist.sumEntries()
        logger.info("Events in fit window: %.1f", n_total)

        if n_total < 10:
            logger.warning("Too few events in %s — skipping.", channel)
            continue

        # --- Build model ---
        bkg_curve_arrays = None
        if sb_mode:
            # S+B model
            pdf, params, n_sig_expected = build_sb_model(
                mass, args.model, n_total, sig_rebinned, args.inject_mu,
            )
            logger.info(
                "Signal template: %.1f events in window (at mu=1)",
                n_sig_expected,
            )
        else:
            # Background-only model
            model_builders = {
                "single-exp": build_single_exp_model,
                "double-exp": build_double_exp_model,
            }
            pdf, params = model_builders[args.model](mass, n_total)

        # --- Fit full window ---
        fit_result = run_fit(pdf, data_hist, params)

        # --- Extract results ---
        fit_results = extract_fit_results(fit_result, params)

        # --- Chi2/ndf ---
        tmp_frame = mass.frame()
        data_hist.plotOn(tmp_frame, ROOT.RooFit.Name("tmp_data"))
        pdf.plotOn(tmp_frame, ROOT.RooFit.Name("tmp_fit"))
        chi2_ndf = tmp_frame.chiSquare("tmp_fit", "tmp_data", len(params))

        roo_hist = tmp_frame.getHist("tmp_data")
        n_bins = roo_hist.GetN()
        ndf = n_bins - len(params)
        fit_results["goodness_of_fit"] = {
            "chi2_per_ndf": chi2_ndf,
            "ndf": ndf,
            "n_bins": n_bins,
            "chi2": chi2_ndf * ndf,
        }

        # --- Extract background component curve for S+B plots ---
        if sb_mode:
            bkg_comp_name = _BKG_COMPONENT_NAMES[args.model]
            bkg_frame = mass.frame()
            data_hist.plotOn(bkg_frame, ROOT.RooFit.Name("bkg_data"))
            pdf.plotOn(
                bkg_frame,
                ROOT.RooFit.Name("bkg_comp"),
                ROOT.RooFit.Components(bkg_comp_name),
            )
            bkg_curve_arrays = _extract_curve(bkg_frame.getCurve("bkg_comp"))

        # --- Assemble full results dict ---
        results = {
            "metadata": {
                "era": args.era,
                "channel": channel,
                "topology": args.topology,
                "region": region,
                "variable": "mass_fourobject",
                "model": args.model,
                "mode": mode_label,
                "signal_tag": signal_tag,
                "signal_window_json": str(win_json_path),
                "signal_mean": sig_mean,
                "signal_rms": sig_rms,
                "fit_window": [fit_lo, fit_hi],
                "fit_window_n_sigma": win_n_sigma,
                "rebin": args.rebin,
                "bin_width_gev": bin_width_gev,
                "n_events_total": float(n_total),
                "timestamp": datetime.now().isoformat(),
            },
            "roofit": fit_results,
        }

        if sb_mode:
            n_sig_fitted = params["n_sig"].getVal()
            n_sig_error = params["n_sig"].getError()
            mu_fitted = n_sig_fitted / n_sig_expected if n_sig_expected > 0 else 0
            mu_error = n_sig_error / n_sig_expected if n_sig_expected > 0 else 0
            results["signal_injection"] = {
                "inject_mu": args.inject_mu,
                "n_sig_expected": n_sig_expected,
                "n_sig_fitted": n_sig_fitted,
                "n_sig_error": n_sig_error,
                "mu_fitted": mu_fitted,
                "mu_error": mu_error,
            }

        mu_suffix = f"_mu{args.inject_mu:.2f}" if sb_mode else ""
        prefix = "sb_fit" if sb_mode else "bkg_fit"
        json_path = base_output_dir / f"{prefix}_{signal_tag}_{channel}{mu_suffix}.json"
        save_results_json(results, json_path)

        # --- Plot ---
        plot_path = base_output_dir / f"{prefix}_{signal_tag}_{channel}{mu_suffix}.pdf"
        if sb_mode:
            data_label_str = (
                f"MC bkg + {args.inject_mu:.2f}" + r"$\times$signal"
                if args.inject_mu != 1.0
                else "MC bkg + signal"
            )
            model_label_str = "S+B"
        else:
            data_label_str = "MC background"
            model_label_str = _MODEL_LABELS[args.model]

        plot_fit(
            mass, data_hist, pdf, fit_result, params, plot_path,
            lumi=info["lumi"],
            com=info.get("com", 13.6),
            channel=channel,
            topology=args.topology,
            era=args.era,
            model_label=model_label_str,
            bin_width_gev=bin_width_gev,
            data_label=data_label_str,
            bkg_curve=bkg_curve_arrays,
            y_range=(1e-1, 1e9) if sb_mode else None,
        )

        # --- Summary ---
        logger.info("=" * 60)
        logger.info(
            "Fit: %s  [signal: %s, model: %s, mode: %s]",
            channel, signal_tag, args.model, mode_label,
        )
        logger.info("  Status:    %d (0 = converged)", fit_results["fit_status"]["status"])
        logger.info("  covQual:   %d (3 = full)", fit_results["fit_status"]["cov_quality"])
        logger.info("  chi2/ndf:  %.2f (%d dof)", chi2_ndf, ndf)
        for pname in sorted(params.keys()):
            p = params[pname]
            if pname.startswith("n"):
                logger.info("  %-10s %.0f +/- %.0f", pname + ":", p.getVal(), p.getError())
            else:
                logger.info("  %-10s %.5f +/- %.5f", pname + ":", p.getVal(), p.getError())
        if sb_mode:
            logger.info("  --- Signal injection ---")
            logger.info("  inject_mu:   %.3f", args.inject_mu)
            logger.info("  n_sig_exp:   %.1f (at mu=1)", n_sig_expected)
            logger.info("  mu_fitted:   %.3f +/- %.3f", mu_fitted, mu_error)
        logger.info("  Fit window: [%.1f, %.1f] GeV", fit_lo, fit_hi)
        logger.info("  Output:    %s", base_output_dir)
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
