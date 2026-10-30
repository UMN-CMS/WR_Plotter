"""Plots comparing Drell-Yan samples (see bin/compare_dy.py).

- :func:`plot_dy_overlay`: the DYJets histogram of each DY sample, with the
  ratio to the first sample.
- :func:`plot_total_comparison`: data against the total background built with
  each DY sample (the other backgrounds are shared), with Data/Bkg. per
  sample, optionally with each total scaled to the data yield; where data is
  blinded or absent, the totals are compared to the first sample's instead.
- :func:`plot_dy_stat`: the statistical power of each DY sample per bin, its
  relative statistical uncertainty and its effective number of events.
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import mplhep as hep
import numpy as np
from hist import Hist
from matplotlib.figure import Figure

from .inputs import DYSample
from .plotting_helpers import (FONT_SIZE_LEGEND, cms_label, custom_log_formatter, data_errors,
                               fit_header, log_ylim, reorder_legend, set_y_label)
from .regions import Region
from .stack import Binning, hist_yield
from .variables import Variable

# Up to this many samples the legend also gives each one's yield (abs, shape) or N_eff (stat);
# beyond it (the six-sample `all` comparison) the names alone, so the legend fits at 20 pt.
MAX_LEGEND_DETAIL = 4
_BAND = {"hatch": "////", "facecolor": "none", "lw": 0, "edgecolor": "k", "alpha": 0.5}


def _xlabel(variable: Variable) -> str:
    return f"{variable.tlatex_alias} [{variable.unit}]" if variable.unit else variable.tlatex_alias


def _fmt_events(n: float) -> str:
    return f"{n:.0f}" if abs(n) >= 1e3 else f"{n:.4g}"


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)


def _finish(fig, ax, rax, region: Region, variable: Variable, ref: Hist, *, xlim, ylim,
            ratio_ylim, rlabel, lumi, com, text, data: bool, per_unit=False) -> Figure:
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    cms_label(ax, lumi, com, data=data, text=text, year=region.year)
    set_y_label(ax, ref, variable, per_unit=per_unit)
    ax.set_xlabel("")
    reorder_legend(ax, fontsize=FONT_SIZE_LEGEND)
    fit_header(ax)
    rax.axhline(1.0, ls="--", color="k")
    rax.set_ylim(*ratio_ylim)
    rax.set_ylabel(rlabel)
    rax.set_xlabel(_xlabel(variable))
    return fig


def plot_dy_overlay(
    region: Region,
    variable: Variable,
    entries: Sequence[tuple[DYSample, Hist]],
    *,
    binning: Binning,
    xlim: tuple[float, float],
    lumi: float,
    com: float = 13.6,
    normalize: bool = False,
    ratio_ylim: tuple[float, float] = (0.5, 1.5),
    extra_label: str | None = None,
) -> Figure:
    """DYJets of each DY sample overlaid (step + stat. errors); ratio to the first.

    With *normalize*, each sample is scaled to unit yield so only the shapes
    are compared; the legend keeps the absolute yields.
    """
    hep.style.use("CMS")
    fig, (ax, rax) = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1),
                                  sharex=True)
    edges = entries[0][1].axes[0].edges
    scaled = []
    for i, (dy, h) in enumerate(entries):
        total = hist_yield(h, binning)
        scale = 1.0 / total if (normalize and total > 0) else 1.0
        vals = h.values() * scale
        errs = np.sqrt(h.variances()) * scale
        scaled.append((dy, vals, errs))
        ref = ", ref." if i == 0 and len(entries) > 1 else ""
        # the yield on a second line: the legend stays narrow enough to sit beside the label
        label = (f"{dy.legend}\n({_fmt_events(total)} ev.{ref})" if len(entries) <= MAX_LEGEND_DETAIL
                 else f"{dy.legend}{' (ref.)' if ref else ''}")
        hep.histplot(vals, edges, yerr=errs, histtype="step", color=dy.color, linewidth=2,
                     label=label, ax=ax)

    _, ref_vals, ref_errs = scaled[0]
    rel = _ratio(ref_errs, ref_vals)
    rax.stairs(1 - rel, edges, baseline=1 + rel, **_BAND)
    for dy, vals, errs in scaled[1:]:
        hep.histplot(_ratio(vals, ref_vals), edges, yerr=_ratio(errs, ref_vals),
                     histtype="errorbar", color=dy.color, ax=rax, xerr=True)

    shown = [v[(edges[:-1] < xlim[1]) & (edges[1:] > xlim[0])] for _, v, _ in scaled]
    text = f"{region.tlatex_alias}\nDrell-Yan only" + ("\nUnit yield" if normalize else "")
    if extra_label:
        text += f"\n{extra_label}"
    fig = _finish(fig, ax, rax, region, variable, entries[0][1], xlim=xlim,
                  ylim=log_ylim(shown, pad_high=1e3, floor=1e-7 if normalize else 1e-3),
                  ratio_ylim=ratio_ylim,
                  rlabel="Ratio to ref.",
                  lumi=lumi, com=com, text=text, data=False, per_unit=binning.per_gev)
    if normalize:
        ax.set_ylabel(ax.get_ylabel().replace("Events", "Fraction of events", 1))
    return fig


def plot_total_comparison(
    region: Region,
    variable: Variable,
    entries: Sequence[tuple[DYSample, Hist]],
    *,
    data: Hist | None,
    data_blinded: bool,
    xlim: tuple[float, float],
    ylim: tuple[float, float] | None,
    lumi: float,
    com: float = 13.6,
    ratio_ylim: tuple[float, float] = (0.5, 2.0),
    extra_label: str | None = None,
    scale_to_data: bool = False,
    binning: Binning | None = None,
) -> Figure:
    """Data against the total background made with each DY sample.

    With *scale_to_data* (only where data is shown), each total is scaled to
    the data yield in *xlim*, so only the shapes are compared; the factor is
    shown in the legend.  *binning* tells whether the bins are per GeV.
    """
    hep.style.use("CMS")
    fig, (ax, rax) = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1),
                                  sharex=True)
    edges = entries[0][1].axes[0].edges
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    per_gev = binning is not None and binning.per_gev
    show_data = data is not None and not data_blinded
    scale_to_data = scale_to_data and show_data

    factors = [1.0] * len(entries)
    if scale_to_data:
        in_range = (edges[:-1] < xlim[1]) & (edges[1:] > xlim[0])
        count = lambda h: float(((h.values() * widths) if per_gev else h.values())[in_range].sum())
        n_data = count(data)
        factors = [n_data / n if (n := count(total)) > 0 else 1.0 for _, total in entries]

    for i, (dy, total) in enumerate(entries):
        ref = " (ref.)" if i == 0 and len(entries) > 1 and not show_data else ""
        # the factor on a second line (a narrow legend) when the legend is short enough to take it
        sep = "\n" if len(entries) <= MAX_LEGEND_DETAIL else " "
        scaled = sep + rf"($\times${factors[i]:.2f})" if scale_to_data else ""
        hep.histplot(total.values() * factors[i], edges, histtype="step", color=dy.color,
                     linewidth=2, label=f"{dy.legend}{ref}{scaled}", ax=ax)   # the text says "Total bkg. by DY"
    if show_data:
        # Garwood errors; horizontal bars only where the bin width varies (PubGuidelines)
        derrs = data_errors(data, per_unit=per_gev)
        hep.histplot(data, yerr=derrs, label="Data", xerr=not np.allclose(widths, widths[0]),
                     color="k", histtype="errorbar", ax=ax)

    n = len(entries)
    ref_vals = entries[0][1].values()
    if show_data:
        dvals = data.values()
        # Side by side at each bin centre, a fixed distance apart on the page: a shift in
        # GeV put them TeV apart in the wide last bin, where they read as other masses.
        shown = (edges[:-1] < xlim[1]) & (edges[1:] > xlim[0])
        axis_pts = rax.get_position().width * fig.get_size_inches()[0] * 72
        narrowest = np.min(widths[shown]) / (xlim[1] - xlim[0]) * axis_pts
        step = min(6.0, 0.7 * narrowest / max(n - 1, 1))
        for i, (dy, total) in enumerate(entries):
            tvals = total.values() * factors[i]
            shift = mtransforms.offset_copy(rax.transData, fig=fig, units="points",
                                            x=(i - (n - 1) / 2) * step)
            rax.errorbar(centers, _ratio(dvals, tvals),
                         yerr=np.nan_to_num(np.array([_ratio(e, tvals) for e in derrs])),
                         fmt="o", color=dy.color, markersize=4, capsize=2, transform=shift)
        rlabel = "Data / Bkg."
    else:
        rel = _ratio(np.sqrt(entries[0][1].variances()), ref_vals)
        rax.stairs(1 - rel, edges, baseline=1 + rel, **_BAND)
        for dy, total in entries[1:]:
            rax.stairs(_ratio(total.values(), ref_vals), edges, color=dy.color, linewidth=2)
        rlabel = "Ratio to ref."
        if data_blinded:  # corner label: the panel holds the ratios to the reference
            rax.text(0.02, 0.92, "Data blinded", transform=rax.transAxes, ha="left", va="top",
                     fontsize=FONT_SIZE_LEGEND, fontweight="bold", color="red")

    if ylim is None:
        arrays = ([t.values() * f for (_, t), f in zip(entries, factors)]
                  + ([data.values()] if show_data else []))
        ylim = log_ylim(arrays)
    # "Total bkg." once here, not in every legend entry: seven samples' entries
    # were too wide to sit next to this text.
    text = region.tlatex_alias + "\nTotal bkg. by DY"
    text += "\n(scaled to data)" if scale_to_data else ""
    text += f"\n{extra_label}" if extra_label else ""
    return _finish(fig, ax, rax, region, variable, entries[0][1], xlim=xlim, ylim=ylim,
                   ratio_ylim=ratio_ylim, rlabel=rlabel, lumi=lumi, com=com, text=text,
                   data=show_data, per_unit=per_gev)


def effective_events(h: Hist) -> np.ndarray:
    """N_eff = (sum w)^2 / sum w^2 per bin: the number of unweighted events with the same
    relative statistical uncertainty, 1/sqrt(N_eff).  NaN where the sum of weights is not
    positive (negative-weight NLO bins) or the bin is empty."""
    sw = np.asarray(h.values(), dtype=float)
    sw2 = np.asarray(h.variances(), dtype=float)
    ok = (sw > 0) & (sw2 > 0)
    return np.divide(sw * sw, sw2, out=np.full_like(sw, np.nan), where=ok)


def plot_dy_stat(
    region: Region,
    variable: Variable,
    entries: Sequence[tuple[DYSample, Hist]],
    *,
    xlim: tuple[float, float],
    lumi: float,
    com: float = 13.6,
    extra_label: str | None = None,
) -> Figure:
    """The statistical power of each DY sample per bin: its relative statistical
    uncertainty, sqrt(sum w^2) / sum w (left axis, log), which is 1/sqrt(N_eff) with the
    effective number of simulated events N_eff on the right axis.  The legend gives each
    sample's N_eff over the plotted range.  A bin where a sample has no events (or a
    non-positive sum of weights) is left as a gap in its line: drawn at the bottom it would
    read as perfectly precise."""
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 10))
    edges = np.asarray(entries[0][1].axes[0].edges)
    shown = (edges[:-1] < xlim[1]) & (edges[1:] > xlim[0])
    drawn = []
    for dy, h in entries:
        rel = 100.0 / np.sqrt(effective_events(h))
        drawn.append(rel[shown])
        sw = np.asarray(h.values(), dtype=float)[shown].sum()
        sw2 = np.asarray(h.variances(), dtype=float)[shown].sum()
        n_tot = sw * sw / sw2 if sw > 0 and sw2 > 0 else 0.0
        label = (f"{dy.legend}\n($N_{{eff}}$ = {n_tot:.0f})" if len(entries) <= MAX_LEGEND_DETAIL
                 else dy.legend)
        ok = np.isfinite(rel)
        # one open step line per run of filled bins: gaps where the sample has no events
        starts = np.flatnonzero(ok & ~np.r_[False, ok[:-1]])
        stops = np.flatnonzero(ok & ~np.r_[ok[1:], False]) + 1
        for a, b in zip(starts, stops):
            ax.stairs(rel[a:b], edges[a:b + 1], baseline=None, color=dy.color, lw=2, label=label)
            label = None
    finite = np.concatenate(drawn)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    lo, hi = (finite.min(), finite.max()) if finite.size else (1.0, 100.0)
    ax.set_yscale("log")
    ax.set_ylim(lo / 2, max(hi * 3, 150.0))
    ax.set_xlim(*xlim)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_xlabel(_xlabel(variable))
    ax.set_ylabel("Relative stat. unc. per bin [%]")
    # N_eff = (100 / rel)^2 on the right axis; the CMS style's right-hand ticks would sit on it.
    ax.tick_params(axis="y", which="both", right=False)
    sec = ax.secondary_yaxis("right", functions=(lambda r: (100.0 / np.maximum(r, 1e-9)) ** 2,
                                                  lambda n: 100.0 / np.sqrt(np.maximum(n, 1e-18))))
    sec.set_ylabel(r"Effective number of simulated events $N_{eff}$")
    sec.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    text = (f"{region.tlatex_alias}\nDrell-Yan only\nGaps: no simulated events"
            + (f"\n{extra_label}" if extra_label else ""))
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    cms_label(ax, lumi, com, data=False, text=text, year=region.year)
    fit_header(ax)
    return fig
