#!/usr/bin/env python3
"""Physics-motivated parametric model of m_lljj resolution.

Decomposes the W_R -> l N -> l l j j cascade into three energy buckets
(primary lepton, secondary lepton, jet system) using simple two-body
kinematics, applies CMS-style per-component fractional resolution
functions, and combines them in quadrature to predict sigma(m_lljj)
as a function of (M_WR, M_N).

Two parallel joint chi^2 fits are run per era: one against the RMS
column and one against the bin-scan FWHM column from
shape_summary.json. The RMS fit is the physics answer; the FWHM fit
is the diagnostic. Six free parameters per fit:
    a_e, b_e   electron stochastic + constant terms
    a_mu, b_mu muon linear + constant terms
    a_j, b_j   jet stochastic + constant terms (shared between channels)

See fitting_v2/resolution_model_guide.md for the full description.

Usage:
    python fitting_v2/resolution_model.py --era RunIII2024Summer24 -v
    python fitting_v2/resolution_model.py --era RunIISummer20UL18 -v
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wrplotter.cli_utils import setup_logging
from wrplotter.config import load_lumi
from wrplotter.paths import repo_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-era WR mass list (mirrors COMPARISON_WR_MASSES_BY_ERA in
# signal_shape_study.py). Adding a new era is a one-line change.
# ---------------------------------------------------------------------------
WR_MASSES_BY_ERA: dict[str, list[int]] = {
    "RunIII2024Summer24": [2000, 4000, 6000],
    "RunIISummer20UL18": [1000, 2000, 3000, 4000, 5000, 6000],
}

CHANNELS = ("ee", "mumu")
PARAM_NAMES = ("a_e", "b_e", "a_mu", "b_mu", "a_j", "b_j")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_masses(signal_tag: str) -> tuple[int, int]:
    m = re.match(r"WR(\d+)_N(\d+)", signal_tag)
    if not m:
        raise ValueError(f"Cannot parse {signal_tag}")
    return int(m.group(1)), int(m.group(2))


# ---------------------------------------------------------------------------
# Kinematics: W_R -> l_1 N -> l_1 l_2 j j
# ---------------------------------------------------------------------------

def primary_lepton_energy(m_wr: np.ndarray, m_n: np.ndarray) -> np.ndarray:
    """E_l1 = (M_WR^2 - M_N^2) / (2 M_WR). Decreases as M_N grows."""
    return (m_wr ** 2 - m_n ** 2) / (2.0 * m_wr)


def secondary_lepton_energy(m_wr: np.ndarray, m_n: np.ndarray) -> np.ndarray:
    """E_l2 ~ gamma_N * (M_N/3) = (M_WR^2 + M_N^2) / (6 M_WR).

    Equipartition approximation for the 3-body N -> l q q' decay
    (V-A massless 3-body average to within ~10%).
    """
    return (m_wr ** 2 + m_n ** 2) / (6.0 * m_wr)


def jet_system_energy(m_wr: np.ndarray, m_n: np.ndarray) -> np.ndarray:
    """E_jj = M_WR - E_l1 - E_l2 = (M_WR^2 + M_N^2) / (3 M_WR)."""
    return (m_wr ** 2 + m_n ** 2) / (3.0 * m_wr)


# ---------------------------------------------------------------------------
# Per-component fractional resolutions
# ---------------------------------------------------------------------------

def electron_frac_resolution(E: np.ndarray, a_e: float, b_e: float) -> np.ndarray:
    """sigma(E)/E = sqrt((a_e/sqrt(E))^2 + b_e^2). ECAL stochastic + constant."""
    return np.sqrt((a_e ** 2) / E + b_e ** 2)


def muon_frac_resolution(p: np.ndarray, a_mu: float, b_mu: float) -> np.ndarray:
    """sigma(p)/p = sqrt((a_mu * p)^2 + b_mu^2). Sagitta-dominated linear term."""
    return np.sqrt((a_mu * p) ** 2 + b_mu ** 2)


def jet_frac_resolution(E: np.ndarray, a_j: float, b_j: float) -> np.ndarray:
    """sigma(E)/E = sqrt((a_j/sqrt(E))^2 + b_j^2). Same form as electron."""
    return np.sqrt((a_j ** 2) / E + b_j ** 2)


# ---------------------------------------------------------------------------
# m_lljj resolution prediction
# ---------------------------------------------------------------------------

def predict_sigma_mlljj(
    m_wr: np.ndarray,
    m_n: np.ndarray,
    channel: str,
    a_e: float, b_e: float,
    a_mu: float, b_mu: float,
    a_j: float, b_j: float,
) -> np.ndarray:
    """sigma(m_lljj)^2 = sigma(E_l1)^2 + sigma(E_l2)^2 + sigma(E_jj)^2.

    Quadrature sum of independent component resolutions evaluated at
    typical lab-frame energies from the kinematic decomposition.
    Channel selects which lepton resolution applies to both leptons.
    """
    e_l1 = primary_lepton_energy(m_wr, m_n)
    e_l2 = secondary_lepton_energy(m_wr, m_n)
    e_jj = jet_system_energy(m_wr, m_n)

    if channel == "ee":
        s_l1 = e_l1 * electron_frac_resolution(e_l1, a_e, b_e)
        s_l2 = e_l2 * electron_frac_resolution(e_l2, a_e, b_e)
    elif channel == "mumu":
        s_l1 = e_l1 * muon_frac_resolution(e_l1, a_mu, b_mu)
        s_l2 = e_l2 * muon_frac_resolution(e_l2, a_mu, b_mu)
    else:
        raise ValueError(f"Unknown channel {channel!r}")

    s_jj = e_jj * jet_frac_resolution(e_jj, a_j, b_j)
    return np.sqrt(s_l1 ** 2 + s_l2 ** 2 + s_jj ** 2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_shape_summary(era: str, sub_dir: str) -> Path:
    """Resolve the path to shape_summary.json for the given era."""
    base = repo_root() / "fitting_v2" / "outputs" / era / "signal_shape_study_onshell"
    if sub_dir:
        # signal_shape_study writes a flat output dir; sub_dir is unused there
        # but we accept it for parity with other scripts.
        pass
    path = base / "shape_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input: {path}\n"
            f"Run signal_shape_study.py --era {era} first to produce it."
        )
    return path


def assemble_dataset(
    summary_path: Path,
    wr_masses: list[int],
    width_key: str,
    topology: str = "resolved",
    min_ratio: float = 0.0,
) -> dict:
    """Build flat parallel arrays per channel from the shape summary JSON.

    width_key: "rms" or "fwhm".
    min_ratio: minimum M_N/M_WR to include (useful for excluding the
        boosted off-shell regime from the RMS fit).
    Returns dict keyed by channel with (m_wr, m_n, width, n_eff) arrays.
    """
    with open(summary_path) as f:
        d = json.load(f)

    out: dict[str, dict] = {}
    for ch in CHANNELS:
        m_wr_l, m_n_l, w_l, n_l, tags = [], [], [], [], []
        for tag, ch_data in d.items():
            wr, n = parse_masses(tag)
            if wr not in wr_masses:
                continue
            if n / wr < min_ratio:
                continue
            if ch not in ch_data or topology not in ch_data[ch]:
                continue
            entry = ch_data[ch][topology]
            w = entry.get(width_key)
            integral = entry.get("integral", 0.0)
            if w is None or not np.isfinite(w) or w <= 0 or integral <= 0:
                continue
            m_wr_l.append(wr)
            m_n_l.append(n)
            w_l.append(w)
            n_l.append(integral)
            tags.append(tag)
        out[ch] = {
            "m_wr": np.array(m_wr_l, dtype=float),
            "m_n": np.array(m_n_l, dtype=float),
            "width": np.array(w_l, dtype=float),
            "n_eff": np.array(n_l, dtype=float),
            "tags": tags,
        }
        logger.info("  %s/%s: %d points loaded", ch, width_key, len(m_wr_l))
    return out


def width_uncertainty(width: np.ndarray, n_eff: np.ndarray) -> np.ndarray:
    """Stat uncertainty on RMS/FWHM, with a 5% modeling-uncertainty floor.

    The pure-statistical part is `width / sqrt(2N)` (standard rule for
    Gaussian sample stdev). For ~1e4 events this is ~3 GeV, much smaller
    than the modeling uncertainty of a back-of-envelope quadrature-sum
    resolution model. We add a 5% fractional floor in quadrature so chi^2
    is dominated by modeling discrepancy rather than statistical noise.
    """
    stat = width / np.sqrt(np.maximum(2.0 * n_eff, 1.0))
    floor = 0.05 * width
    return np.sqrt(stat ** 2 + floor ** 2)


# ---------------------------------------------------------------------------
# Joint chi^2 fit (curve_fit on stacked ee+mumu)
# ---------------------------------------------------------------------------

def stack_channels(dataset: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack ee and mumu into a single (X, y, yerr, channel_id) array."""
    m_wr = np.concatenate([dataset[ch]["m_wr"] for ch in CHANNELS])
    m_n = np.concatenate([dataset[ch]["m_n"] for ch in CHANNELS])
    y = np.concatenate([dataset[ch]["width"] for ch in CHANNELS])
    n_eff = np.concatenate([dataset[ch]["n_eff"] for ch in CHANNELS])
    yerr = width_uncertainty(y, n_eff)
    ch_id = np.concatenate([
        np.full(len(dataset[ch]["m_wr"]), i, dtype=int)
        for i, ch in enumerate(CHANNELS)
    ])
    X = np.vstack([m_wr, m_n, ch_id])
    return X, y, yerr, ch_id


def joint_model(X: np.ndarray, a_e: float, b_e: float,
                a_mu: float, b_mu: float, a_j: float, b_j: float) -> np.ndarray:
    m_wr, m_n, ch_id = X
    out = np.empty_like(m_wr)
    ee_mask = (ch_id == 0)
    mumu_mask = (ch_id == 1)
    if ee_mask.any():
        out[ee_mask] = predict_sigma_mlljj(
            m_wr[ee_mask], m_n[ee_mask], "ee",
            a_e, b_e, a_mu, b_mu, a_j, b_j,
        )
    if mumu_mask.any():
        out[mumu_mask] = predict_sigma_mlljj(
            m_wr[mumu_mask], m_n[mumu_mask], "mumu",
            a_e, b_e, a_mu, b_mu, a_j, b_j,
        )
    return out


def run_fit(dataset: dict, label: str) -> dict:
    """Run a single joint chi^2 fit and return parameter dict."""
    X, y, yerr, _ = stack_channels(dataset)
    n_pts = len(y)
    if n_pts < len(PARAM_NAMES) + 2:
        raise RuntimeError(f"Too few points ({n_pts}) for {label} fit")

    # Initial guesses motivated by CMS published values:
    # ECAL: ~3%/sqrt(E) stochastic, ~1% constant
    # Muon: ~1.5e-4/GeV linear, ~1% constant
    # Jet:  ~1.0/sqrt(E) stochastic (large), ~5% constant
    p0 = [0.03, 0.01, 1.5e-4, 0.01, 1.0, 0.05]
    bounds = (
        [0.0,  0.0,  0.0,    0.0,  0.0, 0.0],
        [1.0,  0.5,  1.0e-2, 0.5,  10.0, 0.5],
    )

    logger.info("[%s fit] %d points, p0 = %s", label, n_pts, p0)
    popt, pcov = curve_fit(
        joint_model, X, y, sigma=yerr, p0=p0, bounds=bounds,
        absolute_sigma=True, maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))

    y_pred = joint_model(X, *popt)
    residuals = (y - y_pred) / yerr
    chi2 = float(np.sum(residuals ** 2))
    ndf = n_pts - len(PARAM_NAMES)

    result = {
        "params": {name: float(val) for name, val in zip(PARAM_NAMES, popt)},
        "errors": {name: float(err) for name, err in zip(PARAM_NAMES, perr)},
        "chi2": chi2,
        "ndf": ndf,
        "chi2_per_ndf": chi2 / ndf if ndf > 0 else float("nan"),
        "n_points": n_pts,
    }

    logger.info("[%s fit] chi2/ndf = %.2f / %d = %.2f",
                label, chi2, ndf, result["chi2_per_ndf"])
    for name, val, err in zip(PARAM_NAMES, popt, perr):
        logger.info("  %-5s = %.4g +/- %.4g", name, val, err)

    return result


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

CMS_COLORS = ["#3f90da", "#bd1f01", "#832db6", "#e76300",
              "#3f8f3a", "#a96b59", "#717581", "#ffa90e"]


def _channel_label(ch: str) -> str:
    return {"ee": "ee", "mumu": r"$\mu\mu$"}[ch]


def _setup_axes(ax, channel: str, era: str, com: float):
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  com=com, fontsize=20)
    ax.text(0.05, 0.96, f"{_channel_label(channel)}\nResolved SR\n{era}",
            transform=ax.transAxes, fontsize=18, verticalalignment="top")


def plot_energy_decomposition(wr_masses: list[int], era: str, com: float,
                              output_dir: Path) -> None:
    """Three separate plots of E_l1, E_l2, E_jj vs M_N/M_WR."""
    # Dense grid from 2000 to 6000 in 200 GeV steps (pure kinematics,
    # no signal files needed).
    wr_plot = list(range(2000, 6001, 200))

    components = [
        ("primary_lepton", r"Primary lepton $E_{\ell_1}$",
         primary_lepton_energy,
         r"$E_{\ell_1} = \frac{M_{W_R}^2 - M_N^2}{2\,M_{W_R}}$"),
        ("secondary_lepton", r"Secondary lepton $E_{\ell_2}$",
         secondary_lepton_energy,
         r"$E_{\ell_2} = \frac{M_{W_R}^2 + M_N^2}{6\,M_{W_R}}$"),
        ("jet_system", r"Jet system $E_{jj}$",
         jet_system_energy,
         r"$E_{jj} = \frac{M_{W_R}^2 + M_N^2}{3\,M_{W_R}}$"),
    ]

    ratios = np.linspace(0.05, 0.99, 200)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors

    cmap = mcm.viridis
    norm = mcolors.Normalize(vmin=wr_plot[0], vmax=wr_plot[-1])

    for tag, title, energy_fn, formula in components:
        hep.style.use("CMS")
        fig, ax = plt.subplots()

        for wr in wr_plot:
            m_wr = np.full_like(ratios, float(wr))
            m_n = ratios * m_wr
            e = energy_fn(m_wr, m_n)
            ax.plot(ratios, e, "-", color=cmap(norm(wr)), linewidth=2)

        ax.set_xlabel(r"$M_N / M_{W_R}$")
        ax.set_ylabel("Energy [GeV]")
        ax.grid(alpha=0.3)
        hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                      com=com, fontsize=20)
        ax.text(0.05, 0.96, f"{title}\nKinematics only\n{era}",
                transform=ax.transAxes, fontsize=16, verticalalignment="top")
        ax.text(0.95, 0.95, formula, transform=ax.transAxes, fontsize=20,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="gray", alpha=0.8))

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=14)

        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(0, y_hi * 1.25)

        out = output_dir / f"energy_decomposition_{tag}.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out.name)

    # Combined plot: all three components on one axis
    hep.style.use("CMS")
    fig, ax = plt.subplots()

    linestyles = {
        "primary_lepton": ("-", r"$E_{\ell_1}$ (primary lepton)"),
        "secondary_lepton": ("--", r"$E_{\ell_2}$ (secondary lepton)"),
        "jet_system": ("-.", r"$E_{jj}$ (jet system)"),
    }

    for wr in wr_plot:
        m_wr = np.full_like(ratios, float(wr))
        m_n = ratios * m_wr
        c = cmap(norm(wr))
        for tag, _, energy_fn, _ in components:
            ls, _ = linestyles[tag]
            ax.plot(ratios, energy_fn(m_wr, m_n), ls, color=c, linewidth=2)

    ax.set_xlabel(r"$M_N / M_{W_R}$")
    ax.set_ylabel("Energy [GeV]")
    ax.grid(alpha=0.3)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  com=com, fontsize=20)
    ax.text(0.05, 0.96, f"Energy decomposition\nKinematics only\n{era}",
            transform=ax.transAxes, fontsize=16, verticalalignment="top")

    # Linestyle legend (use gray lines so it's independent of color)
    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color="gray", linestyle=ls, linewidth=2, label=lbl)
        for ls, lbl in linestyles.values()
    ]
    ax.legend(handles=style_handles, fontsize=12, loc="upper right")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=14)

    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(0, y_hi * 1.25)

    out = output_dir / "energy_decomposition.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out.name)


def plot_resolution_overlay(
    dataset: dict,
    fit_result: dict,
    wr_masses: list[int],
    channel: str,
    era: str,
    com: float,
    output_path: Path,
    *,
    ylabel: str,
    extra_curve: dict | None = None,
    extra_label: str | None = None,
) -> None:
    """Predicted curves overlaid on measured points, one curve per WR mass.

    Model curves are drawn at 2/3/4/5/6 TeV with a discrete legend.
    Data points are plotted at whatever WR masses have data.
    """
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ds = dataset[channel]
    p = fit_result["params"]

    # WR masses for model curves
    wr_curve_grid = [2000, 3000, 4000, 5000, 6000]
    # Discrete colors: dark purple to gold
    curve_colors = {
        2000: "#440154",  # dark purple
        3000: "#31688e",  # teal-blue
        4000: "#35b779",  # green
        5000: "#c8a415",  # dark gold
        6000: "#b8860b",  # darker gold / dark goldenrod
    }

    n_grid = np.linspace(0.10, 0.99, 200)

    # Model curves
    for wr in wr_curve_grid:
        c = curve_colors[wr]
        m_wr = np.full_like(n_grid, float(wr))
        m_n = n_grid * m_wr
        sigma = predict_sigma_mlljj(
            m_wr, m_n, channel,
            p["a_e"], p["b_e"], p["a_mu"], p["b_mu"], p["a_j"], p["b_j"],
        )
        ax.plot(n_grid, sigma, "-", color=c, linewidth=2, zorder=2)

    # Data points at available WR masses
    for wr in wr_masses:
        c = curve_colors.get(wr, "#333333")
        mask = (ds["m_wr"] == float(wr))
        if mask.any():
            ratios_meas = ds["m_n"][mask] / wr
            widths_meas = ds["width"][mask]
            errs_meas = width_uncertainty(widths_meas, ds["n_eff"][mask])
            ax.errorbar(ratios_meas, widths_meas, yerr=errs_meas, fmt="o",
                        color=c, markersize=6,
                        markeredgecolor="black", markeredgewidth=0.5,
                        zorder=3, label=rf"$W_R$ {wr}")

    ax.set_xlabel(r"$M_N / M_{W_R}$")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    _setup_axes(ax, channel, era, com)

    ax.legend(fontsize=12, loc="upper right")

    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(0, y_hi * 1.30)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


def plot_component_contribution(
    fit_result: dict,
    rep_wr: int,
    channel: str,
    era: str,
    com: float,
    output_path: Path,
) -> None:
    """sigma(E_l1)^2, sigma(E_l2)^2, sigma(E_jj)^2 and their sum vs M_N/M_WR."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    p = fit_result["params"]
    ratios = np.linspace(0.10, 0.99, 200)
    m_wr = np.full_like(ratios, float(rep_wr))
    m_n = ratios * m_wr

    e_l1 = primary_lepton_energy(m_wr, m_n)
    e_l2 = secondary_lepton_energy(m_wr, m_n)
    e_jj = jet_system_energy(m_wr, m_n)

    if channel == "ee":
        s_l1 = e_l1 * electron_frac_resolution(e_l1, p["a_e"], p["b_e"])
        s_l2 = e_l2 * electron_frac_resolution(e_l2, p["a_e"], p["b_e"])
    else:
        s_l1 = e_l1 * muon_frac_resolution(e_l1, p["a_mu"], p["b_mu"])
        s_l2 = e_l2 * muon_frac_resolution(e_l2, p["a_mu"], p["b_mu"])
    s_jj = e_jj * jet_frac_resolution(e_jj, p["a_j"], p["b_j"])
    s_tot = np.sqrt(s_l1 ** 2 + s_l2 ** 2 + s_jj ** 2)

    ax.plot(ratios, s_l1, "-", color=CMS_COLORS[0], linewidth=2,
            label=r"$\sigma(E_{\ell_1})$ primary lepton")
    ax.plot(ratios, s_l2, "-", color=CMS_COLORS[1], linewidth=2,
            label=r"$\sigma(E_{\ell_2})$ secondary lepton")
    ax.plot(ratios, s_jj, "-", color=CMS_COLORS[2], linewidth=2,
            label=r"$\sigma(E_{jj})$ jet system")
    ax.plot(ratios, s_tot, "-", color="black", linewidth=2.5,
            label=r"Quadrature sum $\sigma(m_{\ell\ell jj})$")

    ax.set_xlabel(r"$M_N / M_{W_R}$")
    ax.set_ylabel("Component contribution to width [GeV]")
    ax.grid(alpha=0.3)
    _setup_axes(ax, channel, era, com)
    ax.text(0.05, 0.78, rf"$W_R$ {rep_wr} GeV", transform=ax.transAxes,
            fontsize=16, verticalalignment="top")
    ax.legend(fontsize=12, loc="upper right")

    # Fitted parameter values below legend
    if channel == "ee":
        param_lines = (
            rf"$a_e = {p['a_e']:.2g}$, $b_e = {100*p['b_e']:.1f}\%$"
            "\n"
            rf"$a_j = {p['a_j']:.2f}$, $b_j = {100*p['b_j']:.1f}\%$"
        )
    else:
        param_lines = (
            rf"$a_\mu = {p['a_mu']:.2g}$/GeV, $b_\mu = {100*p['b_mu']:.1f}\%$"
            "\n"
            rf"$a_j = {p['a_j']:.2f}$, $b_j = {100*p['b_j']:.1f}\%$"
        )
    ax.text(
        0.95, 0.72, param_lines,
        transform=ax.transAxes, fontsize=12,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="gray", alpha=0.8),
    )

    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(0, y_hi * 1.30)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path.name)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks() -> None:
    logger.info("--- Sanity checks ---")
    # 1. Energy conservation
    for wr, n in [(2000, 1100), (4000, 1900), (6000, 3000)]:
        e1 = primary_lepton_energy(np.array([wr]), np.array([n]))[0]
        e2 = secondary_lepton_energy(np.array([wr]), np.array([n]))[0]
        ej = jet_system_energy(np.array([wr]), np.array([n]))[0]
        total = e1 + e2 + ej
        logger.info("  WR%d N%d: E_l1=%.1f E_l2=%.1f E_jj=%.1f sum=%.1f (expect %d)",
                    wr, n, e1, e2, ej, total, wr)
        assert abs(total - wr) < 1e-6

    # 2. Boundary behavior
    e1_zero = primary_lepton_energy(np.array([4000.0]), np.array([0.0]))[0]
    e1_one = primary_lepton_energy(np.array([4000.0]), np.array([4000.0]))[0]
    logger.info("  E_l1(M_N=0) = %.1f (expect 2000), E_l1(M_N=M_WR) = %.1f (expect 0)",
                e1_zero, e1_one)
    assert abs(e1_zero - 2000.0) < 1e-6
    assert abs(e1_one) < 1e-6

    # 3. Reference resolution at WR4000 N1900, ee channel
    ref_sigma = predict_sigma_mlljj(
        np.array([4000.0]), np.array([1900.0]), "ee",
        a_e=0.03, b_e=0.01, a_mu=1.5e-4, b_mu=0.01, a_j=1.0, b_j=0.05,
    )[0]
    logger.info("  Reference sigma(m_lljj) at WR4000 N1900 ee = %.1f GeV", ref_sigma)
    # Reference uses constant-term-only resolutions; expect O(100 GeV) -- well
    # below the measured ~470 GeV, which is what motivates the fit floating the
    # parameters higher.
    assert 50 < ref_sigma < 1000, f"unexpected reference sigma {ref_sigma}"
    logger.info("--- Sanity checks passed ---")


# ---------------------------------------------------------------------------
# Output dump
# ---------------------------------------------------------------------------

def write_fit_json(rms_fit: dict, output_path: Path) -> None:
    blob = {"rms_fit": rms_fit}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(blob, f, indent=2)
    logger.info("Saved %s", output_path.name)


def print_fit_results(rms_fit: dict) -> None:
    logger.info("--- RMS fit results ---")
    logger.info("  %-6s  %-22s", "param", "value +/- error")
    for name in PARAM_NAMES:
        rv = rms_fit["params"][name]
        re_ = rms_fit["errors"][name]
        logger.info("  %-6s  %.4g +/- %.2g", name, rv, re_)
    logger.info("  chi2/ndf = %.2f / %d = %.2f",
                rms_fit["chi2"], rms_fit["ndf"], rms_fit["chi2_per_ndf"])


def plot_per_wr_resolution(
    dataset: dict,
    fit_result: dict,
    wr_masses: list[int],
    channel: str,
    era: str,
    com: float,
    output_dir: Path,
) -> None:
    """One plot per WR mass showing data + model curve + stats box."""
    ds = dataset[channel]
    p = fit_result["params"]
    n_grid = np.linspace(0.10, 0.99, 200)

    for wr in wr_masses:
        hep.style.use("CMS")
        fig, ax = plt.subplots()

        mask = (ds["m_wr"] == float(wr))
        if not mask.any():
            continue

        ratios_meas = ds["m_n"][mask] / wr
        widths_meas = ds["width"][mask]
        errs_meas = width_uncertainty(widths_meas, ds["n_eff"][mask])

        # Data points
        ax.errorbar(ratios_meas, widths_meas, yerr=errs_meas, fmt="o",
                     color="#3f90da", markersize=6,
                     markeredgecolor="black", markeredgewidth=0.5,
                     zorder=3, label="Data")

        # Model curve
        m_wr = np.full_like(n_grid, float(wr))
        m_n = n_grid * m_wr
        sigma = predict_sigma_mlljj(
            m_wr, m_n, channel,
            p["a_e"], p["b_e"], p["a_mu"], p["b_mu"], p["a_j"], p["b_j"],
        )
        ax.plot(n_grid, sigma, "-", color="#bd1f01", linewidth=2,
                zorder=2, label="Model")

        # Per-WR chi2
        y_pred = predict_sigma_mlljj(
            ds["m_wr"][mask], ds["m_n"][mask], channel,
            p["a_e"], p["b_e"], p["a_mu"], p["b_mu"], p["a_j"], p["b_j"],
        )
        residuals = (widths_meas - y_pred) / errs_meas
        chi2_wr = float(np.sum(residuals ** 2))
        n_pts = int(mask.sum())

        ax.set_xlabel(r"$M_N / M_{W_R}$")
        ax.set_ylabel(r"RMS$_{\mathrm{onshell}}(m_{\ell\ell jj})$ [GeV]")
        ax.grid(alpha=0.3)
        _setup_axes(ax, channel, era, com)

        ch_labels = {"ee": "ee", "mumu": r"$\mu\mu$"}
        ch_label = ch_labels.get(channel, channel)
        ax.text(0.05, 0.78, rf"$M_{{W_R}}$ = {wr} GeV",
                transform=ax.transAxes, fontsize=16, verticalalignment="top")

        # Stats box
        stats_text = (
            rf"$N_{{\mathrm{{pts}}}}$ = {n_pts}"
            "\n"
            rf"$\chi^2$ = {chi2_wr:.1f}"
            "\n"
            rf"$\chi^2 / N$ = {chi2_wr / n_pts:.2f}"
        )
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes, fontsize=13,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.8),
        )

        ax.legend(fontsize=12, loc="center right")
        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(0, y_hi * 1.30)

        out = output_dir / f"resolution_model_WR{wr}_{channel}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out.name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physics-motivated m_lljj resolution model.",
    )
    parser.add_argument("--era", required=True, choices=sorted(WR_MASSES_BY_ERA),
                        help="Era tag")
    parser.add_argument("--dir", default="",
                        help="Optional subdirectory under input/output paths")
    parser.add_argument("--rep-wr-mass", type=int, default=4000,
                        help="WR mass used for the component-contribution plot "
                             "(default: 4000)")
    parser.add_argument("--rms-min-ratio", type=float, default=0.0,
                        help="Minimum M_N/M_WR for the RMS core fit "
                             "(default: 0.0, no cut)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory "
                             "(default: fitting_v2/outputs/<era>/resolution_model/)")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    run_sanity_checks()

    era = args.era
    wr_masses = WR_MASSES_BY_ERA[era]
    info = load_lumi(era)
    com = info.get("com", 13.6)

    summary_path = load_shape_summary(era, args.dir)
    logger.info("Reading %s", summary_path)
    logger.info("WR masses for %s: %s", era, wr_masses)

    base_output = Path(args.output_dir or
                       str(repo_root() / "fitting_v2" / "outputs" /
                           era / "resolution_model_onshell"))
    base_output.mkdir(parents=True, exist_ok=True)

    # Load RMS onshell dataset
    rms_min_ratio = args.rms_min_ratio
    logger.info("Loading RMS onshell dataset (min ratio %.2f)", rms_min_ratio)
    rms_data = assemble_dataset(summary_path, wr_masses, "rms_onshell",
                                min_ratio=rms_min_ratio)

    # Run RMS fit
    rms_fit = run_fit(rms_data, "RMS")
    print_fit_results(rms_fit)

    # Plot 1: energy decomposition (kinematics only)
    plot_energy_decomposition(wr_masses, era, com, base_output)

    # Plot 2: resolution overlays per channel
    rep_wr = args.rep_wr_mass
    if rep_wr not in wr_masses:
        logger.warning("--rep-wr-mass %d not in WR list %s; using %d",
                       rep_wr, wr_masses, wr_masses[0])
        rep_wr = wr_masses[0]

    for ch in CHANNELS:
        plot_resolution_overlay(
            rms_data, rms_fit, wr_masses, ch, era, com,
            base_output / f"resolution_model_rms_onshell_{ch}.pdf",
            ylabel=r"RMS$_{\mathrm{onshell}}(m_{\ell\ell jj})$ [GeV]",
        )
        # Plot 3: per-WR mass diagnostic plots with stats
        plot_per_wr_resolution(
            rms_data, rms_fit, wr_masses, ch, era, com, base_output,
        )
        # Plot 4: component contribution — one per WR mass
        for wr in wr_masses:
            plot_component_contribution(
                rms_fit, wr, ch, era, com,
                base_output / f"component_contribution_WR{wr}_{ch}.pdf",
            )

    # Fit JSON
    write_fit_json(rms_fit, base_output / "resolution_model_fit.json")

    logger.info("Done. Outputs in %s", base_output)


if __name__ == "__main__":
    main()
