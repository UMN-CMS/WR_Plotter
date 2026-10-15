"""Cutflow and efficiency plots (one cut per x-axis bin).

- :func:`plot_yields`: weighted cutflows as a background stack with data and a
  Data/Bkg. panel; unweighted ones as one line per group (raw MC counts do not
  add up across groups, so they are not stacked).
- :func:`plot_efficiency`: running, relative or single-cut efficiency per
  background group, total background, signal and data.

Rows whose data the analyzer withheld are shaded and marked "Blinded".
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
from matplotlib.figure import Figure

from .config import era_label
from .cutflow import CHANNEL_LATEX, CutflowTable, cut_label, efficiency
from .plotting_helpers import (FONT_SIZE_LEGEND, FONT_SIZE_MIN, SIGNAL_COLORS, SIGNAL_STYLES, cms_label,
                               custom_log_formatter, fit_header, log_ylim, poisson_errors,
                               reorder_legend, signal_label)

_BAND = {"hatch": "////", "facecolor": "none", "lw": 0, "edgecolor": "k", "alpha": 0.5}
_FIGSIZE = (10, 12)  # 10 inches wide like every figure (FONT_SIZE_* print as intended); tall for the 9-entry legend

EFFICIENCY_YLABEL = {
    "running": "Efficiency w.r.t. no cuts",
    "relative": "Efficiency w.r.t. previous cut",
    "single": "Single-cut efficiency",
}


def _title(table: CutflowTable, era: str, what: str) -> str:
    # One short item per line, so the region text leaves room for the legend beside it.
    return (f"{table.topology.capitalize()} {CHANNEL_LATEX.get(table.channel, table.channel)}\n"
            f"Cutflow ({table.kind})\n{what}")


def _cut_axis(ax, cuts, topology: str | None = None) -> None:
    ax.set_xticks(np.arange(len(cuts)))
    ax.set_xticklabels([cut_label(c, "plot", topology) for c in cuts], rotation=45, ha="right", rotation_mode="anchor",
                       fontsize=FONT_SIZE_MIN)
    ax.set_xlim(-0.5, len(cuts) - 0.5)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)


def _legend(ax) -> None:
    """Data first; two columns past six entries (backgrounds, total, three signals), so the
    legend stays short enough to clear the curves at 20 pt."""
    n = len(ax.get_legend_handles_labels()[1])
    reorder_legend(ax, fontsize=FONT_SIZE_LEGEND, ncol=2 if n > 6 else 1)


def _shade_blinded(axes, blinded: np.ndarray, *, label_ax=None) -> None:
    """Grey hatched band over the blinded rows, labelled once."""
    idx = np.flatnonzero(blinded)
    if idx.size == 0:
        return
    lo, hi = idx.min() - 0.5, idx.max() + 0.5
    for ax in axes:
        ax.axvspan(lo, hi, color="0.85", hatch="xx", alpha=0.5, zorder=0, lw=0)
    if label_ax is not None:
        label_ax.text(0.5 * (lo + hi), 0.3, "Blinded", transform=label_ax.get_xaxis_transform(),   # under the legend
                      ha="center", va="center", fontsize=FONT_SIZE_MIN, fontweight="bold", color="red",
                      rotation=90 if hi - lo < 2.5 else 0)


def _draw_signals(ax, table: CutflowTable, edges, values_of) -> list[np.ndarray]:
    drawn = []
    for i, col in enumerate(table.signals):
        vals = values_of(col.cutflow.values)
        hep.histplot(vals, edges, histtype="step", linewidth=2,
                     color=SIGNAL_COLORS[i % len(SIGNAL_COLORS)],
                     linestyle=SIGNAL_STYLES[i % len(SIGNAL_STYLES)],
                     label=signal_label(col.key), ax=ax)
        drawn.append(vals)
    return drawn


def plot_yields(table: CutflowTable, *, era: str, lumi: float, com: float = 13.6) -> Figure:
    """Cutflow yields: stacked backgrounds + data (weighted) or per-group lines (unweighted)."""
    hep.style.use("CMS")
    n = len(table.cuts)
    edges = np.arange(n + 1) - 0.5
    x = np.arange(n)
    show_data = table.data is not None
    drawn = []

    if table.unweighted:
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        rax = None
        for col in table.backgrounds:
            hep.histplot(col.cutflow.values, edges, histtype="step", linewidth=2, color=col.color,
                         label=col.label, ax=ax)
            drawn.append(col.cutflow.values)
    else:
        fig, (ax, rax) = plt.subplots(2, 1, figsize=(_FIGSIZE[0], _FIGSIZE[1] + 2), sharex=True,
                                      gridspec_kw=dict(height_ratios=[3, 1], hspace=0.08))
        stack = list(reversed(table.backgrounds))  # largest group on top
        if stack:
            hep.histplot([c.cutflow.values for c in stack], edges, stack=True, histtype="fill",
                         color=[c.color for c in stack], label=[c.label for c in stack],
                         ax=ax)
        total, total_err = table.total, np.sqrt(table.total_variance)
        ax.stairs(total + total_err, edges, baseline=np.clip(total - total_err, 0, None),
                  label="Bkg. stat. unc.", **_BAND)
        drawn.append(total)

    drawn += _draw_signals(ax, table, edges, lambda v: v)

    if show_data:
        ok = ~table.blinded
        ax.errorbar(x[ok], table.data[ok], yerr=poisson_errors(table.data[ok]), fmt="o",
                    color="k", markersize=7, label="Data", zorder=5)
        drawn.append(table.data[ok])

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylim(*log_ylim(drawn, pad_high=1e4))
    ax.set_ylabel("Simulated events (unweighted)" if table.unweighted else "Events")
    cms_label(ax, lumi, com, data=show_data, year=era_label(era),
              text=_title(table, era, "Unweighted (raw counts)" if table.unweighted else "Weighted"))
    _legend(ax)

    if rax is not None:
        total = table.total
        rel = np.divide(np.sqrt(table.total_variance), total, out=np.zeros_like(total),
                        where=total > 0)
        rax.stairs(1 - rel, edges, baseline=1 + rel, **_BAND)
        if show_data:
            ok = ~table.blinded & (total > 0)
            rax.errorbar(x[ok], table.data[ok] / total[ok],
                         yerr=poisson_errors(table.data[ok]) / total[ok],
                         fmt="o", color="k", markersize=6)
        else:
            rax.text(0.5, 0.78, "No data shown", transform=rax.transAxes, ha="center",
                     va="center", fontsize=FONT_SIZE_MIN, color="0.4")
        rax.axhline(1.0, ls="--", color="k")
        rax.set_ylim(0.0, 2.0)
        rax.set_ylabel("Data / Bkg.")
        _cut_axis(rax, table.cuts, table.topology)
        ax.set_xlabel("")
        _shade_blinded([ax, rax], table.blinded, label_ax=rax)
    else:
        _cut_axis(ax, table.cuts, table.topology)
        _shade_blinded([ax], table.blinded, label_ax=ax)
    fit_header(ax)
    return fig


def plot_efficiency(table: CutflowTable, mode: str, *, era: str, lumi: float,
                    com: float = 13.6) -> Figure:
    """Efficiency of every column vs cut; *mode* is "running" or "relative"."""
    hep.style.use("CMS")
    label_key = "single" if (table.kind == "onecut" and mode == "running") else mode
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    x = np.arange(len(table.cuts))
    drawn = []

    def line(values, **kw):
        eff = efficiency(values, mode)
        ax.plot(x, eff, marker="o", markersize=6, linewidth=2, **kw)
        drawn.append(eff)

    for col in table.backgrounds:
        line(col.cutflow.values, color=col.color, label=col.label)
    if len(table.backgrounds) > 1:
        line(table.total, color="0.3", linestyle="--", label="Total bkg.")
    for i, col in enumerate(table.signals):
        line(col.cutflow.values, color=SIGNAL_COLORS[i % len(SIGNAL_COLORS)],
             linestyle=SIGNAL_STYLES[i % len(SIGNAL_STYLES)], label=signal_label(col.key))
    if table.data is not None:
        eff = efficiency(table.data, mode)
        ax.plot(x, eff, "o", color="k", markersize=8, label="Data", zorder=5)
        drawn.append(eff)

    if mode == "relative":
        ax.set_ylim(0.0, 1.6)
    else:
        ax.set_yscale("log")
        ax.set_ylim(log_ylim(drawn, pad_high=1e3, floor=1e-7)[0], 1e3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylabel(EFFICIENCY_YLABEL[label_key])
    _cut_axis(ax, table.cuts, table.topology)
    _shade_blinded([ax], table.blinded, label_ax=ax)
    weighting = "Unweighted" if table.unweighted else "Weighted"
    cms_label(ax, lumi, com, data=table.data is not None, year=era_label(era),
              text=_title(table, era, f"{weighting}\n{EFFICIENCY_YLABEL[label_key].capitalize()}"))
    _legend(ax)
    fit_header(ax)
    return fig
