#!/usr/bin/env python3
"""Shared infrastructure for the on-shell signal-width study.

Backbone for the two window-robust width definitions we kept — the iterative
Gaussian core fit and the RooKeysPdf FWHM — powering the deep dives
`1_signal_widths/gaussian/detail_gauss_fit.py`, `fwhm/detail_fwhm.py`, and
`compare_gauss_fwhm.py`. Each deep dive fits its width at several seed windows
and calls `window_stability_report` for the pre-window-robustness check. (The
earlier σ_eff / σ_eff-sym / RMS / best-Gaussian estimators were dropped after
that study showed they were window-driven.)

What lives here:

  * Cell discovery / loading
      - `load_master_masses` — read the production mass grid from a CSV,
        grouped by topology (`resolved` / `boosted`). `load_master_resolved_masses`
        is a back-compat shim returning just the resolved tags.
      - `discover_masses` — full on-disk grid (incl. the sub-0.1 boosted points).
      - `collect_cells` — load the native (un-rebinned) MC mass histogram for
        every (channel, topology, mass) cell (boosted x<0.1), reporting skips.

  * Width primitives (each operates on the native edges/vals)
      - `windowed_moments`    — (mean, std-dev) inside [lo, hi] via TH1.
      - `keys_fwhm_detail` / `keys_peak_and_fwhm` — RooKeysPdf peak + FWHM.
      - `gaussian_core_fit`   — iterative single-Gaussian core fit (mu±2σ).
      - `gaussian_chi2_ndf`   — weighted χ²/ndf of a Gaussian over a window.

  * Plotting / reporting (generic, M_WR-colorbar aware)
      - `plot_scalar_vs_x_by_mwr`, `plot_ratio_overlay_by_mwr`,
        `plot_series_vs_x`, `plot_signal_with_interval`, `print_summary_table`.
      - `window_stability_report` — wide-pivot the per-window widths to a CSV +
        print the pre-window-robustness verdict (median / max |ratio − 1|).

ALL widths consume the native, un-rebinned histogram. Rebinning would smear the
FWHM; the factor-6 rebin used elsewhere in the pipeline is a fit-performance
convenience and is intentionally not used here.

Setup:
    source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import ROOT
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "ERROR: PyROOT unavailable. Source LCG_106 first:\n"
        "  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh"
    ) from exc

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)

from measure_fwhm import (  # noqa: E402  (after the ROOT guard, by design)
    ONSHELL_WINDOW_LO_FRAC,
    ONSHELL_WINDOW_HI_FRAC,
    build_hist_key,
    build_region_name,
    load_and_combine_signal,
    parse_masses,
)

logger = logging.getLogger(__name__)

# Gaussian FWHM-to-sigma conversion (FWHM = 2*sqrt(2 ln 2) * sigma).
FWHM_TO_GAUSS_SIGMA: float = 2.0 * np.sqrt(2.0 * np.log(2.0))  # 2.3548

# Region scanned for the smooth peak / FWHM (clipped to the histogram range).
PEAK_SCAN_LO_FRAC: float = 0.3
PEAK_SCAN_HI_FRAC: float = 1.7

# Mass-variable per topology.
MASS_VAR = {"resolved": "mass_fourobject", "boosted": "mass_twoobject"}

CH_LAB = {"ee": "ee", "mumu": r"$\mu\mu$"}

# Monotonic counter so per-cell RooFit objects get unique names (avoids the
# "object already exists" churn when looping over hundreds of cells).
_UID = 0


def _next_uid() -> int:
    global _UID
    _UID += 1
    return _UID


# ---------------------------------------------------------------------------
# Cell discovery / loading
# ---------------------------------------------------------------------------

def load_master_masses(
    csv_path: Path, topology: str | None = None,
) -> dict[str, list[str]] | list[str]:
    """Read mass tags from a production CSV, grouped by topology.

    The master CSV has a `mass` column and a `topology` column tagging each
    point `resolved` or `boosted` (the regime split at x = M_N/M_WR = 0.1). With
    `topology=None` (default) returns ``{topology: [sorted tags]}`` covering both
    regimes; with a specific topology returns just that sorted list.

    A legacy single-column CSV (no `topology`) is treated as all-resolved, so old
    grids keep working.
    """
    df = pd.read_csv(csv_path)
    if "topology" not in df.columns:
        df = df.assign(topology="resolved")
    df = df.dropna(subset=["mass"])

    def _sorted(tags: Iterable[str]) -> list[str]:
        return sorted(set(tags), key=lambda t: parse_masses(t))

    if topology is not None:
        return _sorted(df.loc[df["topology"] == topology, "mass"])
    return {topo: _sorted(grp["mass"]) for topo, grp in df.groupby("topology")}


def load_master_resolved_masses(csv_path: Path) -> list[str]:
    """Backward-compatible shim: the resolved-topology tags only."""
    return load_master_masses(csv_path, topology="resolved")


def discover_masses(input_dirs: Sequence[Path]) -> list[str]:
    """All WR<m>_N<n> signal tags present on disk, sorted by (M_WR, M_N).

    Unlike the production CSV (filtered to M_N/M_WR >= 0.10), this sees the full
    grid — including the very-low-x (highly-boosted) N100/N200 points that the
    boosted analysis needs.
    """
    import re
    pat = re.compile(r"WRAnalyzer_signal_(WR\d+_N\d+)\.root$")
    tags: set[str] = set()
    for d in input_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for f in d.glob("WRAnalyzer_signal_WR*_N*.root"):
            mm = pat.search(f.name)
            if mm:
                tags.add(mm.group(1))
    return sorted(tags, key=parse_masses)


@dataclasses.dataclass(frozen=True)
class CellData:
    """One (channel, topology, mass) MC cell with its native histogram."""
    channel: str
    topology: str
    mass: str
    M_WR: float
    M_N: float
    x: float
    n_events: float
    edges: np.ndarray
    vals: np.ndarray
    variances: np.ndarray  # per-bin sum of w^2 (for weighted-MC errors / chi2)


def collect_cells(
    input_dirs: Sequence[Path],
    channels: Sequence[str],
    topologies: Sequence[str],
    masses: Sequence[str] | Mapping[str, Sequence[str]],
    *,
    min_events: float = 100.0,
    boosted_max_x: float | None = 0.1,
) -> tuple[list[CellData], dict[tuple[str, str], dict[str, int]]]:
    """Load native MC histograms for every (channel, topology, mass) cell.

    `masses` selects which mass tags to load per topology and may be either:

      * a ``{topology: [tags]}`` mapping — each topology draws its own tags from
        the master CSV (see `load_master_masses`). This is the uniform path:
        boosted cells come straight from the CSV's `boosted` rows.
      * a flat sequence (legacy) — those tags are used for RESOLVED, while
        BOOSTED is discovered from disk (`discover_masses`).

    Either way, boosted cells are still kept only in the highly-boosted regime
    x = M_N/M_WR < `boosted_max_x` (default 0.1); that low-x set includes the
    N100/N200 points, and the cut drops higher-x cells whose boosted 68% window
    latches onto off-shell structure. Pass `boosted_max_x=None` to keep all
    boosted cells.

    A cell is also skipped (and counted) when its histogram is missing or its
    full-range integral is below `min_events`. Returns the surviving cells plus
    a per-(channel, topology) skip tally.
    """
    masses_by_topo = masses if isinstance(masses, Mapping) else None

    cells: list[CellData] = []
    skips: dict[tuple[str, str], dict[str, int]] = {}
    disk_masses: list[str] | None = None  # lazily discovered, for legacy boosted

    for channel in channels:
        for topology in topologies:
            key = (channel, topology)
            skips[key] = {"missing": 0, "low_stat": 0, "high_x": 0, "kept": 0}
            region = build_region_name(channel, topology)
            hist_key = build_hist_key(region, MASS_VAR[topology])

            if masses_by_topo is not None:
                topo_masses = masses_by_topo.get(topology, [])
            elif topology == "boosted":
                if disk_masses is None:
                    disk_masses = discover_masses(input_dirs)
                topo_masses = disk_masses
            else:
                topo_masses = masses

            for mass in topo_masses:
                try:
                    M_WR, M_N = parse_masses(mass)
                except Exception:
                    logger.warning("bad mass tag %r, skipping", mass)
                    continue
                if (topology == "boosted" and boosted_max_x is not None
                        and (M_N / M_WR) >= boosted_max_x):
                    skips[key]["high_x"] += 1
                    continue
                try:
                    edges, vals, variances = load_and_combine_signal(
                        input_dirs, hist_key, mass,
                    )
                except Exception as exc:
                    skips[key]["missing"] += 1
                    logger.debug("[%s/%s] %s missing: %s",
                                 channel, topology, mass, exc)
                    continue

                n_events = float(np.maximum(vals, 0.0).sum())
                if n_events < min_events:
                    skips[key]["low_stat"] += 1
                    logger.debug("[%s/%s] %s under-populated (%.1f < %.0f)",
                                 channel, topology, mass, n_events, min_events)
                    continue

                skips[key]["kept"] += 1
                cells.append(CellData(
                    channel=channel, topology=topology, mass=mass,
                    M_WR=float(M_WR), M_N=float(M_N),
                    x=float(M_N) / float(M_WR),
                    n_events=n_events,
                    edges=np.ascontiguousarray(edges, dtype=np.float64),
                    vals=np.ascontiguousarray(vals, dtype=np.float64),
                    variances=np.ascontiguousarray(variances, dtype=np.float64),
                ))

            s = skips[key]
            logger.info("[%s/%s] kept %d, skipped %d missing + %d under-populated "
                        "+ %d high-x", channel, topology, s["kept"], s["missing"],
                        s["low_stat"], s["high_x"])

    return cells, skips


# ---------------------------------------------------------------------------
# ROOT TH1 helper
# ---------------------------------------------------------------------------

def make_th1(edges: np.ndarray, vals: np.ndarray) -> "ROOT.TH1D":
    """Build a detached TH1D from native edges/values (negatives clamped to 0)."""
    n_bins = len(vals)
    edges_arr = np.ascontiguousarray(edges, dtype=np.float64)
    h = ROOT.TH1D(f"h_{_next_uid()}", "", n_bins, edges_arr)
    h.SetDirectory(0)
    for i in range(n_bins):
        h.SetBinContent(i + 1, max(float(vals[i]), 0.0))
    return h


# ---------------------------------------------------------------------------
# Estimator primitives
# ---------------------------------------------------------------------------

def windowed_moments(
    edges: np.ndarray, vals: np.ndarray, lo: float, hi: float,
) -> tuple[float, float]:
    """(mean, std-dev) of the histogram inside [lo, hi] via TH1 getters."""
    h = make_th1(edges, vals)
    h.GetXaxis().SetRangeUser(lo, hi)
    if h.Integral() <= 0:
        return float("nan"), float("nan")
    return float(h.GetMean()), float(h.GetStdDev())


def keys_fwhm_detail(
    edges: np.ndarray, vals: np.ndarray, M_WR: float, *,
    lo_frac: float = PEAK_SCAN_LO_FRAC, hi_frac: float = PEAK_SCAN_HI_FRAC,
    n_grid: int = 2001,
) -> dict:
    """RooKeysPdf peak, FWHM, half-max crossings, and the evaluated curve.

    Builds an adaptive-KDE density from the events in [lo_frac, hi_frac]*M_WR,
    evaluates it on a fine grid over that range, takes the maximum as the peak,
    and walks outward to the half-maximum crossings (x_lo, x_hi). The curve is
    returned in events/bin units (KDE density × n_in × bin_width) for overlaying
    on the histogram. Returns a dict with peak, fwhm, x_lo, x_hi, half_max
    (events/bin), xs, ys (events/bin), n_in, bin_w. NaNs if too few events or a
    crossing is not found inside the range.
    """
    nan = {"peak": float("nan"), "fwhm": float("nan"),
           "x_lo": float("nan"), "x_hi": float("nan"),
           "half_max": float("nan"), "xs": None, "ys": None,
           "n_in": 0.0, "bin_w": float(edges[1] - edges[0])}
    xlo = lo_frac * M_WR
    xhi = min(hi_frac * M_WR, float(edges[-1]))
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.maximum(vals, 0.0)
    sel = (centers >= xlo) & (centers <= xhi) & (counts > 0)
    if sel.sum() < 3:
        return nan

    uid = _next_uid()
    m = ROOT.RooRealVar(f"m_keys_{uid}", "m", float(xlo), float(xhi))
    w = ROOT.RooRealVar(f"w_keys_{uid}", "w", 0.0, 1e12)
    ds = ROOT.RooDataSet(f"ds_keys_{uid}", "ds",
                         ROOT.RooArgSet(m, w), ROOT.RooFit.WeightVar(w))
    arg = ROOT.RooArgSet(m, w)
    for c, v in zip(centers[sel], counts[sel]):
        m.setVal(float(c))
        ds.add(arg, float(v))
    keys = ROOT.RooKeysPdf(f"keys_{uid}", "keys", m, ds,
                           ROOT.RooKeysPdf.MirrorBoth, 1.0)

    xs = np.linspace(float(xlo), float(xhi), int(n_grid))
    ys = np.empty_like(xs)
    mset = ROOT.RooArgSet(m)
    for i, x in enumerate(xs):
        m.setVal(float(x))
        ys[i] = keys.getVal(mset)
    del keys, ds, m, w  # let RooFit release the per-cell objects

    # Scale the normalized KDE density to events/bin for overlay/plotting.
    bin_w = float(edges[1] - edges[0])
    n_in = float(counts[sel].sum())
    ys = ys * n_in * bin_w

    ipk = int(np.argmax(ys))
    peak = float(xs[ipk])
    half = ys[ipk] / 2.0

    x_lo = float("nan")
    for i in range(ipk, 0, -1):
        if ys[i - 1] <= half:
            f = (half - ys[i - 1]) / (ys[i] - ys[i - 1])
            x_lo = xs[i - 1] + f * (xs[i] - xs[i - 1])
            break
    x_hi = float("nan")
    for i in range(ipk, len(xs) - 1):
        if ys[i + 1] <= half:
            f = (half - ys[i + 1]) / (ys[i] - ys[i + 1])
            x_hi = xs[i] + f * (xs[i + 1] - xs[i])
            break

    fwhm = (x_hi - x_lo) if (np.isfinite(x_lo) and np.isfinite(x_hi)) else float("nan")
    return {"peak": peak, "fwhm": float(fwhm), "x_lo": x_lo, "x_hi": x_hi,
            "half_max": float(half), "xs": xs, "ys": ys,
            "n_in": n_in, "bin_w": bin_w}


def keys_peak_and_fwhm(
    edges: np.ndarray, vals: np.ndarray, M_WR: float, *,
    lo_frac: float = PEAK_SCAN_LO_FRAC, hi_frac: float = PEAK_SCAN_HI_FRAC,
    n_grid: int = 2001,
) -> tuple[float, float]:
    """(peak, FWHM) convenience wrapper over `keys_fwhm_detail`."""
    d = keys_fwhm_detail(edges, vals, M_WR, lo_frac=lo_frac, hi_frac=hi_frac,
                         n_grid=n_grid)
    return d["peak"], d["fwhm"]


def gaussian_chi2_ndf(
    edges: np.ndarray, vals: np.ndarray, variances: np.ndarray,
    mu: float, sigma: float, lo: float, hi: float, n_params: int = 2,
) -> tuple[float, int]:
    """Weighted χ² / ndf of a Gaussian (mu, sigma) over [lo, hi].

    Uses proper MC errors (sqrt of the per-bin variance = sum of w²). The
    Gaussian is normalized to the observed integral inside [lo, hi], so it is
    a shape-only comparison. ndf = (bins in range) − n_params. Returns
    (chi2_over_ndf, ndf).
    """
    import math
    centers = 0.5 * (edges[:-1] + edges[1:])
    sel = (centers >= lo) & (centers <= hi) & (variances > 0)
    if sel.sum() <= n_params or sigma <= 0:
        return float("nan"), 0
    obs = np.maximum(vals[sel], 0.0)
    err2 = variances[sel]
    e_lo, e_hi = edges[:-1][sel], edges[1:][sel]

    def cdf(x):
        return 0.5 * (1.0 + np.vectorize(math.erf)((x - mu) / (sigma * np.sqrt(2.0))))

    prob = cdf(e_hi) - cdf(e_lo)
    norm = cdf(hi) - cdf(lo)
    if norm <= 0:
        return float("nan"), 0
    exp = obs.sum() * prob / norm
    chi2 = float(np.sum((obs - exp) ** 2 / err2))
    ndf = int(sel.sum() - n_params)
    return (chi2 / ndf if ndf > 0 else float("nan")), ndf


def gaussian_core_fit(
    edges: np.ndarray, vals: np.ndarray, M_WR: float,
    lo_frac: float, hi_frac: float, *,
    n_sigma: float = 2.0, max_iter: int = 8, tol: float = 0.02,
) -> dict:
    """Iterative single-Gaussian *core* fit (RooFit).

    Seed the fit range at [lo_frac, hi_frac]*M_WR, fit a RooGaussian with mu
    floating, then refit in mu ± n_sigma*sigma and iterate until BOTH mu and
    sigma change by < tol (relative) or max_iter is reached. Because the final
    range is set by the data's own core, the converged sigma should be largely
    independent of the seed window.

    Returns a dict: mu, sigma, sigma_err, n_iter, fit_status, cov_status,
    lo, hi, converged.
    """
    lo = lo_frac * M_WR
    hi = min(hi_frac * M_WR, float(edges[-1]))
    h = make_th1(edges, vals)

    uid = _next_uid()
    m = ROOT.RooRealVar(f"m_gc_{uid}", "m", float(edges[0]), float(edges[-1]))
    dh = ROOT.RooDataHist(f"dh_gc_{uid}", "dh", ROOT.RooArgList(m), h)

    mu0, s0 = windowed_moments(edges, vals, lo, hi)
    if not np.isfinite(mu0):
        mu0 = float(M_WR)
    if not (np.isfinite(s0) and s0 > 0):
        s0 = 0.1 * M_WR
    mu = ROOT.RooRealVar(f"mu_gc_{uid}", "mu", float(mu0),
                         float(edges[0]), float(edges[-1]))
    sigma = ROOT.RooRealVar(f"sigma_gc_{uid}", "sigma", float(s0),
                            1.0, float(edges[-1] - edges[0]))
    gauss = ROOT.RooGaussian(f"g_gc_{uid}", "g", m, mu, sigma)

    min_width = 4.0 * float(edges[1] - edges[0])  # keep a few bins in the range
    fit_status, cov_status, n_iter = -1, -1, 0
    prev_mu = prev_sig = None
    converged = False
    sigma_err = float("nan")
    for n_iter in range(1, max_iter + 1):
        if hi - lo < min_width:
            break
        m.setRange(f"core_{uid}", float(lo), float(hi))
        res = gauss.fitTo(
            dh, ROOT.RooFit.Range(f"core_{uid}"),
            ROOT.RooFit.SumW2Error(False),
            ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-1),
        )
        fit_status = int(res.status()) if res else -1
        cov_status = int(res.covQual()) if res else -1
        mu_v, sig_v = float(mu.getVal()), float(sigma.getVal())
        sigma_err = float(sigma.getError())
        lo = max(float(edges[0]), mu_v - n_sigma * sig_v)
        hi = min(float(edges[-1]), mu_v + n_sigma * sig_v)
        converged = (
            prev_sig is not None
            and abs(sig_v - prev_sig) <= tol * prev_sig
            and abs(mu_v - prev_mu) <= tol * prev_mu
        )
        prev_mu, prev_sig = mu_v, sig_v
        del res
        if converged:
            break

    out = {
        "mu": float(mu.getVal()), "sigma": float(sigma.getVal()),
        "sigma_err": sigma_err, "n_iter": n_iter,
        "fit_status": fit_status, "cov_status": cov_status,
        "lo": lo, "hi": hi, "converged": converged,
    }
    del gauss, dh, m, mu, sigma, h
    return out


# ---------------------------------------------------------------------------
# Plotting / reporting (generic over a list of estimator specs)
# ---------------------------------------------------------------------------
#
# An "estimator spec" is a (column, label, color) triple. The same machinery
# serves the width study now and the mean study later — just pass a different
# list of specs and a different y-axis label.

def _topology_label(topology: str) -> str:
    return topology.capitalize()


def plot_overlay_colored_by_mwr(
    df_ct: pd.DataFrame,
    specs: Sequence[tuple[str, str, str]],
    out_path: Path, *,
    channel: str, topology: str, era: str, com: float,
    ylabel: str = "Width estimate [GeV]",
):
    """Overlay every estimator vs x = M_N/M_WR, colored by M_WR (the required plot).

    Marker shape encodes the estimator (legend); the viridis colorbar encodes
    M_WR. Matches the mplhep CMS style of plot_fit_vs_truth.py / leave_one_out.py.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")
    LBL_FS, TICK_FS = 18, 16
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    x = df_ct["x"].to_numpy()
    m_wr = df_ct["M_WR"].to_numpy()
    vmin, vmax = float(np.min(m_wr)), float(np.max(m_wr))

    # Clip the y-axis to the bulk (99th pct over all estimators) so a few
    # extreme cells — e.g. the symmetric-window blow-up at very low x — don't
    # compress the rest; off-scale points are counted and annotated.
    all_vals = np.concatenate([df_ct[c].to_numpy() for c, _l, _c in specs])
    all_vals = all_vals[np.isfinite(all_vals)]
    y_cap = float(np.percentile(all_vals, 99)) * 1.05 if all_vals.size else None

    sc = None
    n_off = 0
    for i, (col, label, _color) in enumerate(specs):
        y = df_ct[col].to_numpy()
        good = np.isfinite(y)
        if y_cap is not None:
            n_off += int(np.sum(y[good] > y_cap))
        sc = ax.scatter(
            x[good], y[good], c=m_wr[good], cmap="viridis",
            vmin=vmin, vmax=vmax, marker=markers[i % len(markers)],
            s=34, edgecolor="black", linewidth=0.35, label=label, zorder=2,
        )

    ax.set_xlabel(r"$x = M_N / M_{W_R}$", fontsize=LBL_FS)
    ax.set_ylabel(ylabel, fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(alpha=0.3)
    if y_cap is not None:
        ax.set_ylim(0, y_cap)
        if n_off > 0:
            ax.text(0.98, 0.04, f"{n_off} pts off-scale",
                    transform=ax.transAxes, fontsize=11, color="gray",
                    verticalalignment="bottom", horizontalalignment="right")
    else:
        ax.set_ylim(bottom=0)

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
        cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=LBL_FS)
        cbar.ax.tick_params(labelsize=TICK_FS)

    # Estimator legend with neutral (gray) markers so it reads as shape, not color.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=markers[i % len(markers)], linestyle="",
               markerfacecolor="0.6", markeredgecolor="black",
               markersize=9, label=label)
        for i, (_col, label, _c) in enumerate(specs)
    ]
    ax.legend(handles=handles, fontsize=13, loc="upper right",
              framealpha=0.9, title="estimator")

    # mplhep prepends "Simulation" automatically when data=False, so passing
    # "Work in Progress" renders as "Simulation Work in Progress".
    hep.cms.label(loc=0, ax=ax, data=False,
                  label="Work in Progress", com=com, fontsize=LBL_FS)
    ax.text(0.04, 0.96, f"{CH_LAB[channel]}  {_topology_label(topology)} SR\n{era}",
            transform=ax.transAxes, fontsize=LBL_FS,
            verticalalignment="top", horizontalalignment="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def plot_ratio_to_reference(
    df_ct: pd.DataFrame,
    specs: Sequence[tuple[str, str, str]],
    ref_col: str, ref_label: str,
    out_path: Path, *,
    channel: str, topology: str, era: str, com: float,
):
    """Each estimator / reference vs x, one fixed color per estimator.

    Directly visualizes the non-Gaussianity: for a perfectly Gaussian shape all
    ratios collapse to 1. Easier to read than the M_WR-colored overlay.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")
    LBL_FS, TICK_FS = 18, 16

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    x = df_ct["x"].to_numpy()
    ref = df_ct[ref_col].to_numpy()

    for col, label, color in specs:
        if col == ref_col:
            continue
        y = df_ct[col].to_numpy() / ref
        good = np.isfinite(y)
        ax.scatter(x[good], y[good], s=28, color=color, edgecolor="black",
                   linewidth=0.3, alpha=0.85, label=label, zorder=2)

    ax.axhline(1.0, color="red", linewidth=1.5, alpha=0.7, zorder=1)
    ax.set_xlabel(r"$x = M_N / M_{W_R}$", fontsize=LBL_FS)
    ax.set_ylabel(rf"estimator / {ref_label}", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=13, loc="upper right", framealpha=0.9)

    hep.cms.label(loc=0, ax=ax, data=False,
                  label="Work in Progress", com=com, fontsize=LBL_FS)
    ax.text(0.04, 0.96, f"{CH_LAB[channel]}  {_topology_label(topology)} SR\n{era}",
            transform=ax.transAxes, fontsize=LBL_FS,
            verticalalignment="top", horizontalalignment="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def plot_ratio_overlay_by_mwr(
    df_ct: pd.DataFrame,
    series: Sequence[tuple[str, str]],
    out_path: Path, *,
    channel: str, topology: str, era: str, com: float,
    ylabel: str, ylim: tuple[float, float] | None = None,
    mwr_lim: tuple[float, float] | None = None,
    hline: float = 1.0, band: float | None = None,
):
    """Ratio series vs x, one marker per series, points colored by M_WR (colorbar).

    `series` is a list of (column, label); all series share the viridis M_WR
    colorbar and marker shape distinguishes them (legend). Pass a shared `ylim`
    and `mwr_lim` to keep several panels on identical axes/colorbars. Points
    outside `ylim` are counted and annotated rather than rescaling the axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.lines import Line2D

    hep.style.use("CMS")
    LBL_FS, TICK_FS = 18, 16
    markers = ["o", "^", "s", "D", "v"]

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    if band is not None:
        ax.axhspan(hline - band, hline + band, color="green", alpha=0.10, zorder=0)
    ax.axhline(hline, color="red", linewidth=1.4, alpha=0.7, zorder=1)

    x = df_ct["x"].to_numpy()
    m_wr = df_ct["M_WR"].to_numpy()
    vmin, vmax = mwr_lim if mwr_lim else (float(np.min(m_wr)), float(np.max(m_wr)))

    sc = None
    n_off = 0
    for i, (col, _label) in enumerate(series):
        y = df_ct[col].to_numpy()
        good = np.isfinite(y)
        if ylim is not None:
            n_off += int(np.sum((y[good] < ylim[0]) | (y[good] > ylim[1])))
        sc = ax.scatter(x[good], y[good], c=m_wr[good], cmap="viridis",
                        vmin=vmin, vmax=vmax, marker=markers[i % len(markers)],
                        s=42, edgecolor="black", linewidth=0.35, zorder=3)

    ax.set_xlabel(r"$x = M_N / M_{W_R}$", fontsize=LBL_FS)
    ax.set_ylabel(ylabel, fontsize=LBL_FS - 2)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)
        if n_off:
            ax.text(0.98, 0.04, f"{n_off} pts off-scale", transform=ax.transAxes,
                    fontsize=11, color="gray", va="bottom", ha="right")

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
        cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=LBL_FS)
        cbar.ax.tick_params(labelsize=TICK_FS)

    handles = [
        Line2D([0], [0], marker=markers[i % len(markers)], linestyle="",
               markerfacecolor="0.6", markeredgecolor="black",
               markersize=10, label=label)
        for i, (_col, label) in enumerate(series)
    ]
    ax.legend(handles=handles, fontsize=12, loc="upper right", framealpha=0.95,
              title="window")

    hep.cms.label(loc=0, ax=ax, data=False,
                  label="Work in Progress", com=com, fontsize=LBL_FS)
    ax.text(0.04, 0.96, f"{CH_LAB[channel]}  {_topology_label(topology)} SR\n{era}",
            transform=ax.transAxes, fontsize=LBL_FS - 2,
            verticalalignment="top", horizontalalignment="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def print_summary_table(
    df_ct: pd.DataFrame,
    specs: Sequence[tuple[str, str, str]],
    *, channel: str, topology: str, quantity: str = "width",
):
    """Console table: median of each estimator + median pairwise ratio matrix."""
    cols = [c for c, _l, _c in specs]
    labels = [l for _c, l, _cc in specs]

    print(f"\n=== {channel} / {topology}  ({len(df_ct)} cells) — {quantity} estimators ===")
    print("median of each estimator [GeV]:")
    for col, lab in zip(cols, labels):
        med = float(np.nanmedian(df_ct[col].to_numpy()))
        print(f"  {lab:<22} {med:8.1f}")

    # Compact labels for the ratio matrix header.
    short = [c.replace("sigma_", "").replace("mean_", "") for c in cols]
    print("\nmedian pairwise ratio  (row / column):")
    print("            " + "".join(f"{s:>12}" for s in short))
    for ci, (col_i, si) in enumerate(zip(cols, short)):
        row = f"  {si:>10}"
        for cj, col_j in enumerate(cols):
            r = df_ct[col_i].to_numpy() / df_ct[col_j].to_numpy()
            row += f"{float(np.nanmedian(r)):>12.3f}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Per-method diagnostic plots (used by the per-method deep-dive scripts, e.g.
# detail_eff_gauss.py). Generic enough that the gauss_fit / rms / fwhm analogs
# can reuse them with different columns and reference overlays.
# ---------------------------------------------------------------------------

def plot_signal_with_interval(
    edges: np.ndarray, vals: np.ndarray,
    x_low: float, x_high: float, out_path: Path, *,
    mass: str, channel: str, topology: str, era: str, com: float,
    mass_var_label: str = r"$m_{\ell\ell jj}$",
    sigma_label: str = r"\sigma_{\rm eff}",
    extra_lines: Sequence[float] | None = None,
    extra_lines_label: str | None = None,
):
    """Raw native MC histogram with the [x_low, x_high] interval shaded.

    The headline per-cell diagnostic: does the 68% interval actually sit on the
    physics peak, or get pulled toward off-shell / second-peak structure? Used
    for every signal point (important for two-peak boosted cells). σ_eff and the
    interval edges are annotated. `extra_lines` draws extra vertical references
    (e.g. M_WR, the on-shell window edges); `extra_lines_label` documents them in
    the legend.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.lines import Line2D

    M_WR, M_N = parse_masses(mass)
    sigma_eff = 0.5 * (x_high - x_low)
    x_center = 0.5 * (x_low + x_high)
    centers = 0.5 * (edges[:-1] + edges[1:])

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    ax.stairs(np.maximum(vals, 0.0), edges, color="#3f90da", linewidth=1.5,
              label="MC")
    mask = (centers >= x_low) & (centers <= x_high)
    if mask.any():
        ax.fill_between(centers[mask], 0, np.maximum(vals[mask], 0.0),
                        step="mid", color="#3f90da", alpha=0.30, zorder=0,
                        label=r"68.27% interval")
    for xv, lab in ((x_low, r"$x_{\rm low},\,x_{\rm high}$"), (x_high, None)):
        ax.axvline(xv, color="#bd1f01", linestyle="--", linewidth=1.5,
                   zorder=3, label=lab)
    ax.axvline(x_center, color="black", linestyle=":", linewidth=1.3, zorder=3,
               label=r"$x_{\rm center}$")
    for xv in (extra_lines or ()):
        ax.axvline(xv, color="0.30", linestyle=(0, (6, 3)), linewidth=2.2,
                   alpha=0.95, zorder=2)

    # x-range: show the interval plus context (and any low-mass second peak).
    x_hi_view = min(float(edges[-1]), max(1.4 * M_WR, x_high * 1.15))
    ax.set_xlim(0, x_hi_view)
    # Generous top headroom so neither the info text nor the legend (both upper
    # corners) collide with the peak.
    ax.set_ylim(bottom=0)
    _, y_hi = ax.get_ylim()
    ax.set_ylim(0, y_hi * 1.50)

    ax.set_xlabel(mass_var_label + " [GeV]")
    ax.set_ylabel("Events / bin")
    ax.grid(alpha=0.3)

    hep.cms.label(loc=0, ax=ax, data=False,
                  label="Work in Progress", com=com, fontsize=16)
    ax.text(
        0.04, 0.96,
        f"{CH_LAB[channel]}  {_topology_label(topology)} SR\n{era}\n"
        rf"$M_{{W_R}}={M_WR}$, $M_N={M_N}$ GeV",
        transform=ax.transAxes, fontsize=13, verticalalignment="top",
    )
    # Stat box: top-right.
    ax.text(
        0.96, 0.96,
        rf"$x_{{\rm low}}={x_low:.0f}$, $x_{{\rm high}}={x_high:.0f}$ GeV"
        "\n"
        rf"$x_{{\rm center}}={x_center:.0f}$ GeV"
        "\n"
        rf"${sigma_label}=(x_{{\rm high}}-x_{{\rm low}})/2={sigma_eff:.0f}$ GeV",
        transform=ax.transAxes, fontsize=12,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="gray", alpha=0.85),
    )
    # Legend: upper-left, tucked below the info text on the (empty) low-mass
    # side, with a proxy entry documenting the gray reference lines.
    handles, labels = ax.get_legend_handles_labels()
    if extra_lines_label and (extra_lines or ()):
        handles.append(Line2D([0], [0], color="0.30", linestyle=(0, (6, 3)),
                              linewidth=2.2, alpha=0.95))
        labels.append(extra_lines_label)
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.03, 0.78),
              fontsize=11, framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_scalar_vs_x_by_mwr(
    df_ct: pd.DataFrame, col: str, out_path: Path, *,
    channel: str, topology: str, era: str, com: float,
    ylabel: str, hlines: Sequence[tuple[float, str]] | None = None,
    clip_bulk: bool = True, ylim: tuple[float, float] | None = None,
):
    """One quantity vs x = M_N/M_WR, colored by M_WR (CMS house style).

    `ylim` (explicit) overrides the auto-range — e.g. to center the data band
    vertically or to share a range across panels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")
    LBL_FS, TICK_FS = 18, 16
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    x = df_ct["x"].to_numpy()
    y = df_ct[col].to_numpy()
    m_wr = df_ct["M_WR"].to_numpy()
    good = np.isfinite(y)
    sc = ax.scatter(x[good], y[good], c=m_wr[good], cmap="viridis",
                    s=34, edgecolor="black", linewidth=0.35, zorder=2)

    for yv, lab in (hlines or ()):
        ax.axhline(yv, color="red", linestyle="--", linewidth=1.3, alpha=0.7)
        ax.text(0.005, yv, lab, transform=ax.get_yaxis_transform(),
                fontsize=11, color="red", va="bottom", ha="left")

    if ylim is not None:
        ax.set_ylim(*ylim)
        if good.sum():
            n_off = int(np.sum((y[good] < ylim[0]) | (y[good] > ylim[1])))
            if n_off:
                ax.text(0.98, 0.04, f"{n_off} pts off-scale",
                        transform=ax.transAxes, fontsize=11, color="gray",
                        verticalalignment="bottom", horizontalalignment="right")
    elif clip_bulk and good.sum():
        y_cap = float(np.percentile(y[good], 99)) * 1.08
        n_off = int(np.sum(y[good] > y_cap))
        ax.set_ylim(bottom=min(0.0, float(np.min(y[good]))), top=y_cap)
        if n_off > 0:
            ax.text(0.98, 0.04, f"{n_off} pts off-scale",
                    transform=ax.transAxes, fontsize=11, color="gray",
                    verticalalignment="bottom", horizontalalignment="right")

    ax.set_xlabel(r"$x = M_N / M_{W_R}$", fontsize=LBL_FS)
    ax.set_ylabel(ylabel, fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=LBL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)

    hep.cms.label(loc=0, ax=ax, data=False,
                  label="Work in Progress", com=com, fontsize=LBL_FS)
    ax.text(0.04, 0.96, f"{CH_LAB[channel]}  {_topology_label(topology)} SR\n{era}",
            transform=ax.transAxes, fontsize=LBL_FS,
            verticalalignment="top", horizontalalignment="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def plot_series_vs_x(
    df_ct: pd.DataFrame,
    series: Sequence[tuple[str, str, str]],
    out_path: Path, *,
    channel: str, topology: str, era: str, com: float,
    ylabel: str, hlines: Sequence[tuple[float, str]] | None = None,
    legend_loc: str = "best", legend_bbox: tuple[float, float] | None = None,
):
    """Several fixed-color quantities vs x on one axis (e.g. x_low/center/high)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")
    LBL_FS, TICK_FS = 18, 16
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    x = df_ct["x"].to_numpy()
    for col, label, color in series:
        y = df_ct[col].to_numpy()
        good = np.isfinite(y)
        ax.scatter(x[good], y[good], s=28, color=color, edgecolor="black",
                   linewidth=0.3, alpha=0.85, label=label, zorder=2)

    for yv, lab in (hlines or ()):
        ax.axhline(yv, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(0.005, yv, lab, transform=ax.get_yaxis_transform(),
                fontsize=11, color="gray", va="bottom", ha="left")

    ax.set_xlabel(r"$x = M_N / M_{W_R}$", fontsize=LBL_FS)
    ax.set_ylabel(ylabel, fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=13, loc=legend_loc, bbox_to_anchor=legend_bbox,
              framealpha=0.9)

    hep.cms.label(loc=0, ax=ax, data=False,
                  label="Work in Progress", com=com, fontsize=LBL_FS)
    ax.text(0.04, 0.96, f"{CH_LAB[channel]}  {_topology_label(topology)} SR\n{era}",
            transform=ax.transAxes, fontsize=LBL_FS,
            verticalalignment="top", horizontalalignment="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# Window-robustness report (folded in from the old width_window_stability.py)
# ---------------------------------------------------------------------------

def window_stability_report(
    df, value_col: str,
    windows: Sequence[tuple], baseline: tuple, out_csv: Path, *,
    channels: Sequence[str], topologies: Sequence[str], est_name: str,
):
    """Wide-pivot the per-window widths to a CSV and print the robustness verdict.

    A deep-dive table already fits its width at every seed window, so this is a
    re-pivot of data it has in hand: one row per cell, one `val_<window>` column
    per window. It also prints, per (channel, topology), each window's median
    width and its median / max |ratio − 1| vs the `baseline` window, then an
    overall verdict — the quantitative "is this width window-driven?" check.

    `df` is the long per-(cell, window) table with columns channel, category,
    mWR, mN, fit_range (= "[lo,hi]") and `value_col` (e.g. sigma_gaus). Returns
    the wide table it writes.
    """
    def _wkey(lo, hi):  return f"{lo:g}_{hi:g}".replace(".", "p")
    def _wstr(lo, hi):  return f"[{lo:g},{hi:g}]"
    def _wpretty(lo, hi):  return f"[{lo:g}, {hi:g}]"

    wide = df.pivot_table(index=["channel", "category", "mWR", "mN"],
                          columns="fit_range", values=value_col).reset_index()
    out = pd.DataFrame({
        "channel": wide["channel"], "topology": wide["category"],
        "mass": [f"WR{int(w)}_N{int(n)}" for w, n in zip(wide["mWR"], wide["mN"])],
        "M_WR": wide["mWR"], "M_N": wide["mN"], "x": wide["mN"] / wide["mWR"],
    })
    for lo, hi in windows:
        out[f"val_{_wkey(lo, hi)}"] = wide[_wstr(lo, hi)].to_numpy()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    logger.info("Wrote %s (%d rows)", out_csv, len(out))

    base_col = f"val_{_wkey(*baseline)}"
    print(f"\n=== {est_name} pre-window stability  "
          f"(baseline {_wpretty(*baseline)}) ===")
    overall: dict[str, list] = {}
    for channel in channels:
        for topology in topologies:
            sub = out[(out.channel == channel) & (out.topology == topology)]
            if sub.empty:
                continue
            print(f"\n  {channel} / {topology}  ({len(sub)} cells)")
            print(f"    {'window':>14} {'median val':>11} "
                  f"{'med |Δ|/base':>13} {'max |Δ|/base':>13}")
            for lo, hi in windows:
                col = f"val_{_wkey(lo, hi)}"
                dev = np.abs(sub[col].to_numpy() / sub[base_col].to_numpy() - 1.0)
                tag = " (base)" if (lo, hi) == baseline else ""
                print(f"    {_wpretty(lo, hi):>14} "
                      f"{np.nanmedian(sub[col]):>9.1f}G "
                      f"{np.nanmedian(dev)*100:>11.2f}% "
                      f"{np.nanmax(dev)*100:>11.2f}%{tag}")
                if (lo, hi) != baseline:
                    overall.setdefault(_wpretty(lo, hi), []).extend(dev.tolist())

    print("\n  --- overall (all cells, non-baseline windows vs baseline) ---")
    worst = 0.0
    for lab, devs in overall.items():
        arr = np.array(devs)
        worst = max(worst, float(np.nanmedian(arr)) * 100)
        print(f"    {lab:>14}: median |Δ|/base = {np.nanmedian(arr)*100:5.2f}%, "
              f"max = {np.nanmax(arr)*100:5.2f}%")
    verdict = ("ROBUST (windows agree within a few %)" if worst < 5.0
               else "PRE-WINDOW DRIVEN (windows differ substantially)")
    print(f"\n  VERDICT: worst median deviation = {worst:.2f}%  ->  {verdict}\n")
    return out
