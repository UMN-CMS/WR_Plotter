"""Signal plots of one era from the analyzer's signal histograms (see bin/make_signal_plots.py).

The single-era counterpart of :mod:`wrplotter.signal_compare`, which compares
sets of eras; the definitions (acceptance x efficiency, sigma x L, the cutflow
efficiencies, the mass-peak summary) are that module's and are reused here.

Acceptance x efficiency
    SR yield (every analyzer weight) / sigma x L of the channel's lepton
    flavour, per mass point, topology (resolved, boosted, their sum) and
    channel, as AN-19-083 App. B (per m_WR, against m_N) and as a map over
    the (m_WR, m_N) grid.
Signal in the control regions
    The same fraction for the DY and flavour control regions: how much of the
    produced signal of a flavour lands in each CR (the Run 2 plotter drew
    these next to the signal-region curves).
Cross sections
    sigma(LO) and sigma(LO) x k(NLO) of every point from the analyzer's signal
    config (the k-factor derivation in docs/studies/signal_kfactor.md), and k
    itself, against m_N per m_WR.
Shapes
    The signal-region mass of several points at one m_WR (and of several m_WR
    at a fixed m_N / m_WR), unit area; the position and width of the peak
    against m_N; the other signal-region variables of a few points; the
    cumulative cutflow efficiency of a few points.
"""
from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker
import mplhep as hep
import numpy as np
from hist import Hist

from .config import load_lumi
from .cutflow import cut_label
from .inputs import mass_point, read_manifest
from .paths import SIGNAL_OUTPUTS, plots_base, repo_root
from .plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
from .signal_compare import (
    CHANNEL_LABELS, CHANNELS, COLORS, COMBINED, MARKERS, MASS_LABELS, MASS_VARIABLES, TOPOLOGIES,
    TOPOLOGY_LABELS, CutflowEfficiency, EraSet, Produced, Shape, Value, _normalized, display_range, efficiency,
    flavour_shares, flow_arrays, load_point_hist, make_era_sets, overflow_fraction, point_files, produced_events,
    shape_summary,
)
from .stack import hist_key

logger = logging.getLogger(__name__)

# Control regions of each topology.  A DY CR is a channel's region and its yield is divided by
# sigma x L of that flavour; a flavour CR (no channel in its name) takes events of both flavours
# of the point's file and its yield is divided by sigma x L of both.
CONTROL_REGIONS = {
    "resolved": {"wr_{ch}_resolved_dy_cr": "DY CR", "wr_resolved_flavor_cr": "flavour CR"},
    "boosted": {"wr_{ch}_boosted_dy_cr": "DY CR", "wr_emu_boosted_flavor_cr": "flavour CR (e, $\\mu$-jet)",
                "wr_mue_boosted_flavor_cr": "flavour CR ($\\mu$, e-jet)"},
}
KINEMATIC_VARIABLES = {
    "resolved": ["pt_leading_lepton", "pt_subleading_lepton", "eta_leading_lepton", "eta_subleading_lepton",
                 "pt_leading_jet", "pt_subleading_jet", "eta_leading_jet", "mass_dilepton", "mass_dijet",
                 "pt_dilepton", "mass_threeobject_leadlep", "mass_threeobject_subleadlep"],
    "boosted": ["pt_leading_lepton", "pt_subleading_lepton", "eta_leading_lepton", "eta_subleading_lepton",
                "pt_leading_AK8Jets", "eta_leading_AK8Jets", "mass_dilepton", "pt_dilepton",
                "LSF_leading_AK8Jets", "dPhi_leading_tightlepton_AK8Jet"],
}
VARIABLE_LABELS = {
    "pt_leading_lepton": r"$p_T$ of the leading lepton [GeV]",
    "pt_subleading_lepton": r"$p_T$ of the subleading lepton [GeV]",
    "eta_leading_lepton": r"$\eta$ of the leading lepton",
    "eta_subleading_lepton": r"$\eta$ of the subleading lepton",
    "pt_leading_jet": r"$p_T$ of the leading AK4 jet [GeV]",
    "pt_subleading_jet": r"$p_T$ of the subleading AK4 jet [GeV]",
    "eta_leading_jet": r"$\eta$ of the leading AK4 jet",
    "pt_leading_AK8Jets": r"$p_T$ of the AK8 jet [GeV]",
    "eta_leading_AK8Jets": r"$\eta$ of the AK8 jet",
    "mass_dilepton": r"$m_{\ell\ell}$ [GeV]",
    "mass_dijet": r"$m_{jj}$ [GeV]",
    "pt_dilepton": r"$p_{T,\ell\ell}$ [GeV]",
    "mass_threeobject_leadlep": r"$m_{\ell_1 jj}$ [GeV]",
    "mass_threeobject_subleadlep": r"$m_{\ell_2 jj}$ [GeV]",
    "LSF_leading_AK8Jets": r"$\mathrm{LSF}_3$ of the AK8 jet",
    "dPhi_leading_tightlepton_AK8Jet": r"$|\Delta\phi(\ell, J)|$",
    "mass_fourobject": r"$m_{\ell\ell jj}$ [GeV]",
    "mass_twoobject": r"$m_{\ell J}$ [GeV]",
}


# ---------------------------------------------------------------------------
# Inputs and output folders
# ---------------------------------------------------------------------------

def signal_output_dirs(era: str, dir_name: str, part: str, *, local_plots: bool = False) -> tuple[Path, Path]:
    """(plots, tables) folders of one part of a production's signal outputs:
    ``<plots>/<Run>/<Year>/<Era>/<dir>/signal_outputs/<part>`` (EOS, ``plots/`` or ``outputs/``, as
    :func:`wrplotter.paths.plots_base`) and ``tables/<Run>/<Year>/<Era>/<dir>/signal_outputs/<part>``."""
    info = load_lumi(era)
    plots = plots_base(info["run"], info["year"], era, dir_name or None, local=local_plots) / SIGNAL_OUTPUTS / part
    tables = repo_root() / "tables" / info["run"] / info["year"] / era
    tables = (tables / dir_name if dir_name else tables) / SIGNAL_OUTPUTS / part
    return plots, tables



def load_set(era: str, dir_name: str, working_dir: str | Path | None = None) -> EraSet:
    (era_set,) = make_era_sets([(era, dir_name)], working_dir)
    return era_set


def points_by_mwr(points: Sequence[str]) -> dict[int, list[str]]:
    """{m_WR: points at that m_WR, by m_N}."""
    out: dict[int, list[str]] = {}
    for p in sorted(points, key=mass_point):
        out.setdefault(mass_point(p)[0], []).append(p)
    return out


def point_label(point: str) -> str:
    wr, n = mass_point(point)
    return rf"$m_{{W_R}}$ = {wr}, $m_N$ = {n} GeV"


def spread(points: Sequence[str], n: int) -> list[str]:
    """Up to *n* of *points*, evenly spread, first and last kept."""
    if len(points) <= n:
        return list(points)
    idx = np.unique(np.round(np.linspace(0, len(points) - 1, n)).astype(int))
    return [points[i] for i in idx]


def nearest_fraction(points: Sequence[str], fraction: float) -> str | None:
    """The point whose m_N / m_WR is closest to *fraction*."""
    if not points:
        return None
    return min(points, key=lambda p: abs(mass_point(p)[1] / mass_point(p)[0] - fraction))


# ---------------------------------------------------------------------------
# Efficiencies and CR fractions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PointSummary:
    point: str
    m_wr: int
    m_n: int
    produced: Produced
    efficiency: dict[tuple[str, str], Value | None]       # (topology or COMBINED, channel)
    cr_fraction: dict[tuple[str, str, str], Value | None]  # (topology, region, channel)
    shape: dict[tuple[str, str], Shape | None]            # (topology, channel)


def region_fraction(era_set: EraSet, point: str, region: str, variable: str, channel: str,
                    produced: Produced) -> Value | None:
    """Yield of *region* (integral of its *variable* histogram, flow bins included) over sigma x L of
    the channel's flavour, or of both flavours for a region without a channel (the flavour CRs);
    None without the histogram or without signal of that flavour."""
    n = produced.channels[channel] if "{ch}" in region or f"_{channel}_" in region else produced.total
    if n <= 0:
        return None
    h = load_point_hist(era_set, point, hist_key(region, variable))
    if h is None:
        return None
    _, values, variances = flow_arrays(h)
    return Value(float(values.sum()) / n, math.sqrt(max(float(variances.sum()), 0.0)) / n)


def summarize_point(era_set: EraSet, point: str, *, analyzer_root: str | Path | None = None,
                    fileset: Path | None = None) -> PointSummary:
    produced = produced_events(era_set, point, analyzer_root=analyzer_root,
                               filesets={era_set.era: fileset} if fileset else None)
    eff = {(t, ch): efficiency(era_set, point, t, ch, produced)
           for t in (*TOPOLOGIES, COMBINED) for ch in CHANNELS}
    crs = {}
    for topology, regions in CONTROL_REGIONS.items():
        for pattern in regions:
            for ch in CHANNELS:
                crs[(topology, pattern, ch)] = region_fraction(era_set, point, pattern.format(ch=ch),
                                                               MASS_VARIABLES[topology], ch, produced)
    shapes = {}
    for t in TOPOLOGIES:
        for ch in CHANNELS:
            h = load_point_hist(era_set, point, hist_key(f"wr_{ch}_{t}_sr", MASS_VARIABLES[t]))
            shapes[(t, ch)] = shape_summary(h) if h is not None and float(h.values(flow=True).sum()) > 0 else None
    wr, n = mass_point(point)
    return PointSummary(point, wr, n, produced, eff, crs, shapes)


def cross_sections(config_path: Path, era_set: EraSet, points: Sequence[str]
                   ) -> dict[str, dict[str, dict[str, float]]]:
    """{point: {channel: {"lo": pb, "nlo": pb, "k": k}}} from the analyzer's signal config, for the
    datasets each point's manifest lists.  A dataset without a generator value gives NaN for lo/k."""
    import json
    with open(config_path) as f:
        config = json.load(f)
    by_name: dict[str, dict] = {}
    for key, entry in config.items():
        for name in (key, entry.get("dataset"), entry.get("das_name"), entry.get("sample")):
            if name:
                by_name[str(name).lstrip("/")] = entry
    out: dict[str, dict[str, dict[str, float]]] = {}
    for point in points:
        files = point_files(era_set, point)
        if files is None:
            continue
        for path in files:
            for dataset in (read_manifest(path) or {}).get("datasets", []):
                entry = by_name.get(str(dataset).lstrip("/"))
                if entry is None or entry.get("xsec") is None:
                    continue
                source = entry.get("xsec_source") or {}
                nlo = float(entry["xsec"])
                lo = source.get("generator")
                lo = float(lo) * float(source.get("factor", 1.0)) if lo is not None else math.nan
                k = float(source.get("kfactor", nlo / lo if lo and math.isfinite(lo) else math.nan))
                for ch, share in flavour_shares(dataset).items():
                    if share > 0:
                        out.setdefault(point, {})[ch] = {"lo": lo, "nlo": nlo, "k": k}
    return out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _figure(figsize=(10, 9)):
    hep.style.use("CMS")
    return plt.subplots(figsize=figsize)


def _header(ax, era_set: EraSet, text: str, **kw) -> None:
    cms_label(ax, None, data=False, rlabel=f"({era_set.com:g} TeV)", text=text, **kw)


def _style(i: int) -> dict:
    return {"color": COLORS[i % len(COLORS)], "marker": MARKERS[i % len(MARKERS)]}


def _series(values: Sequence[tuple[int, Value | None]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = [(m, v) for m, v in values if v is not None and math.isfinite(v.value)]
    return (np.array([m for m, _ in pts], dtype=float), np.array([v.value for _, v in pts]),
            np.array([v.error for _, v in pts]))


def label_text(channel: str, *, topology: str | None = None, m_wr: int | None = None,
               point: str | None = None) -> str:
    lines = [CHANNEL_LABELS[channel] + (f", {TOPOLOGY_LABELS[topology]}" if topology else "")]
    if point:
        lines.append(point_label(point))
    elif m_wr:
        lines.append(rf"$m_{{W_R}}$ = {m_wr} GeV")
    return "\n".join(lines)


def plot_efficiency_vs_mn(era_set: EraSet, summaries: Sequence[PointSummary], m_wr: int, channel: str):
    """A x epsilon of the resolved SR, the boosted SR and their sum against m_N (AN-19-083 App. B)."""
    fig, ax = _figure()
    for i, topology in enumerate((COMBINED, "resolved", "boosted")):
        mn, val, err = _series([(s.m_n, s.efficiency[(topology, channel)]) for s in summaries])
        st = _style(i)
        ax.errorbar(mn, 100 * val, yerr=100 * err, color=st["color"], marker=st["marker"], markersize=6, lw=1.5,
                    capsize=2, label=TOPOLOGY_LABELS[topology] if topology != COMBINED else "Combined")
    ax.set_xlim(0, m_wr)
    top = max((100 * s.efficiency[(COMBINED, channel)].value for s in summaries
               if s.efficiency[(COMBINED, channel)] is not None), default=1.0)
    ax.set_ylim(0, 1.6 * top)
    ax.set_xlabel(r"$m_N$ [GeV]")
    ax.set_ylabel(r"$A\epsilon$ [%]")
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    _header(ax, era_set, label_text(channel, m_wr=m_wr) + "\nSR yield / $\\sigma L$ of the flavour")
    fit_header(ax)
    return fig


def plot_cr_fraction_vs_mn(era_set: EraSet, summaries: Sequence[PointSummary], m_wr: int, channel: str,
                           topology: str):
    """Signal in each control region of *topology* over sigma x L of the flavour, against m_N."""
    fig, ax = _figure()
    for i, (pattern, name) in enumerate(CONTROL_REGIONS[topology].items()):
        mn, val, err = _series([(s.m_n, s.cr_fraction[(topology, pattern, channel)]) for s in summaries])
        st = _style(i)
        ax.errorbar(mn, 100 * val, yerr=100 * err, color=st["color"], marker=st["marker"], markersize=6, lw=1.5,
                    capsize=2, label=f"{TOPOLOGY_LABELS[topology].split(' ')[0]} {name}")
    mn, val, err = _series([(s.m_n, s.efficiency[(topology, channel)]) for s in summaries])
    ax.errorbar(mn, 100 * val, yerr=100 * err, color="grey", marker="x", markersize=6, lw=1.0, ls=":",
                capsize=2, label=TOPOLOGY_LABELS[topology])
    ax.set_xlim(0, m_wr)
    ax.set_yscale("log")
    shown = np.concatenate([100 * _series([(s.m_n, s.cr_fraction[(topology, p, channel)]) for s in summaries])[1]
                            for p in CONTROL_REGIONS[topology]] + [100 * val])
    shown = shown[shown > 0]
    ax.set_ylim(shown.min() / 5 if shown.size else 1e-4, (shown.max() if shown.size else 1) * 10)
    ax.set_xlabel(r"$m_N$ [GeV]")
    ax.set_ylabel(r"Yield / $\sigma L$ [%]")
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND - 4)
    _header(ax, era_set, label_text(channel, m_wr=m_wr) + "\nSignal in the control regions\n"
            "(flavour CRs: both flavours over $\\sigma L$ of both)")
    fit_header(ax)
    return fig


def plot_efficiency_map(era_set: EraSet, summaries: Sequence[PointSummary], topology: str, channel: str):
    """A x epsilon over the (m_WR, m_N) grid, colour-coded."""
    fig, ax = _figure((11, 9))
    pts = [(s.m_wr, s.m_n, s.efficiency[(topology, channel)]) for s in summaries]
    pts = [(wr, n, v.value) for wr, n, v in pts if v is not None and math.isfinite(v.value)]
    if not pts:
        return fig
    wr = np.array([p[0] for p in pts], dtype=float)
    n = np.array([p[1] for p in pts], dtype=float)
    val = 100 * np.array([p[2] for p in pts])
    sc = ax.scatter(wr, n, c=val, s=90, cmap="viridis", vmin=0, vmax=max(val.max(), 1e-3), edgecolors="k",
                    linewidths=0.4)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$A\epsilon$ [%]")
    lim = max(wr.max(), n.max()) * 1.08
    ax.plot([0, lim], [0, lim], color="grey", lw=1, ls="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"$m_N$ [GeV]")
    _header(ax, era_set, label_text(channel, topology=topology) + "\nSR yield / $\\sigma L$ of the flavour")
    fit_header(ax)
    return fig


def plot_xsec_vs_mn(era_set: EraSet, xsecs: Mapping[str, Mapping[str, Mapping[str, float]]], m_wr: int,
                    points: Sequence[str], channel: str, quantity: str):
    """sigma (pb; LO and LO x k) or k against m_N at one m_WR (AN-19-083 Fig. 6)."""
    fig, ax = _figure()
    rows = [(mass_point(p)[1], xsecs[p][channel]) for p in points if p in xsecs and channel in xsecs[p]]
    if not rows:
        return fig
    mn = np.array([m for m, _ in rows], dtype=float)
    if quantity == "xsec":
        for i, (key, label) in enumerate((("nlo", r"$\sigma_{LO} \times k$ (NLO), the analysis normalisation"),
                                          ("lo", r"$\sigma_{LO}$ (generator)"))):
            y = np.array([r[key] for _, r in rows]) * 1000.0     # pb -> fb
            st = _style(i)
            ax.plot(mn, y, color=st["color"], marker=st["marker"], markersize=6, lw=1.5, label=label)
        ax.set_yscale("log")
        ax.set_ylabel(rf"$\sigma(pp \to W_R \to {CHANNEL_LABELS[channel]} jj)$ [fb]")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 30)
    else:
        y = np.array([r["k"] for _, r in rows])
        ax.plot(mn, y, color=COLORS[0], marker=MARKERS[0], markersize=6, lw=1.5, label=r"$k = \sigma_{NLO} / \sigma_{LO}$")
        ax.set_ylabel(r"$k$-factor")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.5 * (hi - lo))
    ax.set_xlim(0, m_wr)
    ax.set_xlabel(r"$m_N$ [GeV]")
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND - 2)
    _header(ax, era_set, label_text(channel, m_wr=m_wr))
    fit_header(ax)
    return fig


def plot_shapes(era_set: EraSet, entries: Sequence[tuple[str, Hist]], *, xlabel: str, text: str,
                target_bins: int = 40, xlim: tuple[float, float] | None = None, logy: bool = False):
    """Histograms scaled to unit area, overlaid (events beyond the range in the last bin)."""
    hists = [h for _, h in entries]
    factor, auto_xlim = display_range(hists, target_bins=target_bins)
    xlim = xlim or auto_xlim
    if max(overflow_fraction(h, xlim[1]) for h in hists) > 0.01:
        text += "\nLast bin: overflow"
    fig, ax = _figure()
    top = 0.0
    for i, (label, h) in enumerate(entries):
        edges, vals, errs = _normalized(h, factor, xlim)
        shown = vals[(edges[:-1] < xlim[1]) & (edges[1:] > xlim[0])]
        top = max(top, float(shown.max()) if shown.size else 0.0)
        hep.histplot(vals, edges, yerr=errs, histtype="step", color=COLORS[i % len(COLORS)], linewidth=2,
                     label=label, ax=ax)
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(max(top * 1e-4, 1e-6), top * 50 if top > 0 else 1)
    else:
        ax.set_ylim(0, 1.55 * top if top > 0 else 1)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of events / bin")
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND - 4)
    _header(ax, era_set, text)
    fit_header(ax)
    return fig


def plot_resolution_vs_mn(era_set: EraSet, summaries: Sequence[PointSummary], m_wr: int, topology: str,
                          channel: str):
    """Median / m_WR and sigma_eff / median of the signal-region mass peak against m_N."""
    hep.style.use("CMS")
    fig, (ax_pos, ax_wid) = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[1, 1], hspace=0.1), sharex=True,
                                         figsize=(10, 11))
    rows = [(s.m_n, s.shape[(topology, channel)]) for s in summaries if s.shape[(topology, channel)] is not None]
    if rows:
        mn = np.array([m for m, _ in rows], dtype=float)
        pos = [sh.median / Value(m_wr, 0.0) for _, sh in rows]
        wid = [sh.sigma_eff / sh.median for _, sh in rows]
        kw = dict(color=COLORS[0], marker=MARKERS[0], markersize=6, lw=1.5, capsize=2)
        ax_pos.errorbar(mn, [v.value for v in pos], yerr=[v.error for v in pos], **kw)
        ax_wid.errorbar(mn, [100 * v.value for v in wid], yerr=[100 * v.error for v in wid], **kw)
    ax_pos.set_ylabel(r"Median / $m_{W_R}$", fontsize=18)
    lo, hi = ax_pos.get_ylim()
    ax_pos.set_ylim(lo, hi + 0.8 * (hi - lo))
    ax_wid.set_yscale("log")
    ax_wid.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax_wid.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax_wid.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_wid.set_ylabel(r"$\sigma_{\mathrm{eff}}$ / median [%]", fontsize=18)
    ax_wid.set_xlabel(r"$m_N$ [GeV]")
    ax_wid.set_xlim(0, m_wr)
    _header(ax_pos, era_set, label_text(channel, topology=topology, m_wr=m_wr) + "\n" + MASS_LABELS[topology] + " peak",
            fontsize=18, text_fontsize=15)
    fit_header(ax_pos)
    return fig


def _fit_bottom(fig, ax, pad: float = 0.015) -> None:
    """Move the axes up and right until its x tick labels lie inside the figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels() if t.get_text()]
    if not boxes:
        return
    pos = ax.get_position()
    low = min(b.y0 for b in boxes) / fig.bbox.height
    left = min(b.x0 for b in boxes) / fig.bbox.width
    if low < pad:
        fig.subplots_adjust(bottom=pos.y0 + pad - low)
    if left < pad:
        fig.subplots_adjust(left=pos.x0 + pad - left)


def _efficiency_ticks(ax, log: bool) -> None:
    """Efficiency ticks only up to 100 %: the axis may run higher to hold the CMS label and legend."""
    ticker = matplotlib.ticker
    lo = ax.get_ylim()[0]
    if log:
        first = math.floor(math.log10(lo))
        ax.yaxis.set_major_locator(ticker.FixedLocator([10.0 ** k for k in range(first, 3) if lo <= 10.0 ** k <= 100]))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_locator(ticker.FixedLocator(
            [m * 10.0 ** k for k in range(first, 2) for m in range(2, 10) if lo <= m * 10.0 ** k <= 100]))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        ax.yaxis.set_major_locator(ticker.FixedLocator(range(0, 101, 20)))
        ax.yaxis.set_minor_locator(ticker.FixedLocator([v for v in range(0, 101, 5) if v % 20]))


def plot_cutflows(era_set: EraSet, entries: Sequence[tuple[str, CutflowEfficiency]], *, topology: str, channel: str,
                  m_wr: int | None = None):
    """Cumulative efficiency after each cut for several points (base weight), drawn for a note: the cut names
    under the axis, rotated, with the bottom margin made to fit them; the efficiency axis labelled up to
    100 % only (the space above holds the CMS label and the legend), linear unless the efficiencies span
    more than a factor 20.  *m_wr* names the plot's fixed W_R mass when every entry has it."""
    cuts = list(entries[0][1].cuts)
    for _, cf in entries[1:]:
        cuts += [c for c in cf.cuts if c not in cuts]
    fig, ax = _figure((12, 9))
    shown = []
    for i, (label, cf) in enumerate(entries):
        pos = np.array([cuts.index(c) for c in cf.cuts])
        st = _style(i)
        ax.errorbar(pos, cf.values * 100, yerr=cf.errors * 100, color=st["color"], marker=st["marker"], markersize=8,
                    lw=2, capsize=3, label=label)
        shown.append(cf.values * 100)
    shown = np.concatenate(shown)
    shown = shown[np.isfinite(shown) & (shown > 0)]
    low = float(shown.min()) if shown.size else 1.0
    log = bool(shown.size) and float(shown.max()) / low > 20
    if log:
        ax.set_yscale("log")
        ax.set_ylim(10 ** math.floor(math.log10(low / 2)), 100)
    else:
        ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(len(cuts)))
    ax.set_xticklabels([cut_label(c, "plot") for c in cuts], rotation=40, ha="right", rotation_mode="anchor",
                       fontsize=18)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_xlim(-0.5, len(cuts) - 0.5)
    ax.set_ylabel("Cumulative efficiency [%]")
    _fit_bottom(fig, ax)
    short = max(len(label) for label, _ in entries) <= 24
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND, ncols=2 if short and len(entries) > 3 else 1)
    _header(ax, era_set, label_text(channel, topology=topology, m_wr=m_wr) + "\nWithout lepton SFs")
    fit_header(ax, margin=0.07)           # the clearance is measured to the marker centres
    _efficiency_ticks(ax, log)
    return fig


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _fmt(v: Value | None, scale: float = 100.0, digits: int = 3) -> str:
    return "" if v is None or not math.isfinite(v.value) else f"{scale * v.value:.{digits}f}"


def _fmt_err(v: Value | None, scale: float = 100.0, digits: int = 3) -> str:
    return "" if v is None or not math.isfinite(v.value) else f"{scale * v.error:.{digits}f}"


def efficiency_rows(era_set: EraSet, summaries: Sequence[PointSummary]) -> list[dict]:
    rows = []
    for s in summaries:
        for ch in CHANNELS:
            if s.produced.channels[ch] <= 0:
                continue
            row = {"era": era_set.era, "dir": era_set.dir_name, "point": s.point, "m_wr": s.m_wr, "m_n": s.m_n,
                   "channel": ch, "produced_events": f"{s.produced.channels[ch]:.4g}"}
            for t in (*TOPOLOGIES, COMBINED):
                row[f"eff_{t}_pct"] = _fmt(s.efficiency[(t, ch)])
                row[f"eff_{t}_err_pct"] = _fmt_err(s.efficiency[(t, ch)])
            for topology, regions in CONTROL_REGIONS.items():
                for pattern in regions:
                    row[f"cr_{pattern.replace('{ch}_', '')}_pct"] = _fmt(s.cr_fraction[(topology, pattern, ch)])
            for t in TOPOLOGIES:
                sh = s.shape[(t, ch)]
                row[f"{t}_median"] = "" if sh is None else f"{sh.median.value:.1f}"
                row[f"{t}_sigma_eff"] = "" if sh is None else f"{sh.sigma_eff.value:.1f}"
            rows.append(row)
    return rows


def write_csv(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def efficiency_latex(era_set: EraSet, summaries: Sequence[PointSummary], m_wr: int, channel: str) -> str:
    """CMS-style table of A x epsilon (%, stat.) per m_N at one m_WR: resolved, boosted, combined."""
    info = load_lumi(era_set.era)
    lines = [r"\begin{table}[htbp]", r"  \centering",
             rf"  \topcaption{{Signal acceptance times efficiency (\%) in the {CHANNEL_LABELS[channel].replace('$', '')} channel "
             rf"for $m_{{W_R}} = {m_wr}$\GeV, {era_set.era} ({float(info['lumi']):.1f}\fbinv): the resolved and boosted "
             r"signal regions and their sum, over $\sigma \times L$ of the lepton flavour. Statistical errors.}",
             rf"  \label{{tab:signal_eff_{channel}_WR{m_wr}}}", r"  \cmsTable{", r"    \begin{tabular}{r|ccc}",
             r"      $m_N$ [GeV] & Resolved SR & Boosted SR & Combined \\", r"      \hline"]
    for s in summaries:
        if s.produced.channels[channel] <= 0:
            continue
        cells = []
        for t in (*TOPOLOGIES, COMBINED):
            v = s.efficiency[(t, channel)]
            cells.append("--" if v is None else rf"${100 * v.value:.2f} \pm {100 * v.error:.2f}$")
        lines.append(f"      {s.m_n} & " + " & ".join(cells) + r" \\")
    lines += [r"    \end{tabular}", r"  }", r"\end{table}", ""]
    return "\n".join(lines)
