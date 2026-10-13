from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
import mplhep as hep
from mplhep.label import ExpSuffix, ExpText
from scipy.stats import chi2

from .histogram_utils import for_drawing, sum_hists
from .systematics import band_error2

if TYPE_CHECKING:
    from hist import Hist
    from .regions import Region
    from .variables import Variable

# ----------------------------- constants -------------------------------------- #

# CMS figure guidelines (FigGuidelines twiki, CAT plotting pages): the text after "CMS".
# "Work in progress" is the thesis-endorsement label; a PAS or CMS note takes "Preliminary".
CMS_EXTRA_TEXT = "Work in progress"

FONT_SIZE_TITLE  = 18   # region text: no larger than the legend (FigGuidelines)
FONT_SIZE_LABEL  = 20
FONT_SIZE_LEGEND = 18

# Colours and linestyles cycled over multiple signal overlays so they remain
# visually distinguishable when more than one signal is shown on the same plot.
# CMS (Petroff) palette colours the background fills do not use (those are the
# 6-colour scheme's blue, orange, red and purple, data/sample_groups.yaml):
# the 6-colour violet, then the 10-colour brown, grey, dark orange and dark red.
SIGNAL_COLORS = ["#7a21dd", "#a96b59", "#717581", "#e76300", "#bd1f01"]
SIGNAL_STYLES = ["--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

# ----------------------------- formatters/labels ----------------------------- #

def _safe_sum_hists(hists):
    """Sum of *hists* (None when empty), keeping the overflow hint (histogram_utils.sum_hists)."""
    return sum_hists(hists)

def custom_log_formatter(y: float, pos) -> str:
    """
    Format major ticks on a log y-axis:
      1 -> '1', 10 -> '10', others -> '10^{n}'
    Gracefully handle non-positive y (matplotlib may ask).
    """
    y = float(y)  # ensure plain Python float (ROOT/numpy scalars can confuse np.isclose)
    if y <= 0:
        return ""
    # exact 1 or 10 shown plainly
    if np.isclose(y, 1.0):
        return "1"
    if np.isclose(y, 10.0):
        return "10"
    n = int(np.round(np.log10(y)))
    # show only for powers of 10; empty for others to avoid clutter
    if np.isclose(y, 10.0**n):
        return rf"$10^{{{n}}}$"
    return ""

def _format_bin_width(w: float) -> str:
    return f"{int(w)}" if float(w).is_integer() else f"{w:.1f}"

def set_y_label(ax, tot, variable, *, per_unit: bool = False) -> None:
    """
    Set y-axis label as 'Events / <binwidth> [unit]' for uniform binning.
    For non-uniform binning, 'Events / bin', or '< Events / [unit] >' when the contents
    were divided by the bin width (*per_unit*); the brackets mark the average over
    the bin (CMS PubGuidelines, figures).
    """
    edges = np.asarray(tot.axes[0].edges)
    widths = np.diff(edges)
    unit = getattr(variable, "unit", "") or ""
    has_uniform = np.allclose(widths, widths[0])

    if has_uniform:
        bw = _format_bin_width(float(widths[0]))
        if unit:
            ax.set_ylabel(f"Events / {bw} {unit}")
        else:
            ax.set_ylabel(f"Events / {bw}")
    elif per_unit:
        ax.set_ylabel(rf"$\langle$Events / {unit or 'unit'}$\rangle$")
    else:
        ax.set_ylabel("Events / bin")

# --------------------------------- legends ---------------------------------- #

def reorder_legend(ax, priority=("Data", "Sim. stat."), fontsize=18, **legend_kw) -> None:
    """
    Put 'Data' first (CMS FigGuidelines: data at the top of a stack plot's legend),
    the MC uncertainty band second, and keep the rest in reverse stack order.
    The band is matched as a prefix (handles both 'Sim. stat. unc.' and
    'Sim. stat. + syst. unc.').  Missing labels are skipped gracefully.
    The legend goes top right: the CMS label and region text hold the top left.
    """
    handles, labels = ax.get_legend_handles_labels()
    idx = {label: i for i, label in enumerate(labels)}
    first = idx.get(priority[0])
    second = next((idx[l] for l in labels if l.startswith(priority[1])), None)
    rest = [i for i in range(len(labels)) if i not in {first, second}]
    order = ([first] if first is not None else []) + \
            ([second] if second is not None else []) + rest
    if not order:
        return
    ax.legend([handles[i] for i in order],
              [labels[i] for i in order],
              loc="upper right", fontsize=fontsize, **legend_kw)

# --------------------------------- drawing ---------------------------------- #

def poisson_errors(counts: np.ndarray) -> np.ndarray:
    """(down, up) offsets of the Garwood 68 % central interval on event *counts*.

    For zero counts the upper limit (1.84) and no downward error, as the CMS
    Statistics Committee recommends for data points (PubGuidelines, figures).
    """
    n = np.asarray(counts, dtype=float)
    alpha = 1 - 0.682689492137
    lo = np.where(n > 0, chi2.ppf(alpha / 2, 2 * n) / 2, 0.0)
    hi = chi2.ppf(1 - alpha / 2, 2 * (n + 1)) / 2
    return np.array([n - lo, hi - n])


def data_counts(data: Hist, *, per_unit: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(event counts, contents per count) of a data histogram; with *per_unit*
    the contents were divided by the bin width, so one event is 1/width."""
    widths = np.diff(np.asarray(data.axes[0].edges))
    scale = 1.0 / widths if per_unit else np.ones_like(widths)
    return np.rint(data.values() / scale), scale


def data_errors(data: Hist, *, per_unit: bool = False) -> np.ndarray:
    """(down, up) Garwood errors of a data histogram in its own units."""
    counts, scale = data_counts(data, per_unit=per_unit)
    return poisson_errors(counts) * scale


def data_ratio(data: Hist, tot_vals: np.ndarray, *,
               per_unit: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Data/MC ratio per bin and its (down, up) statistical errors, shape (2, N);
    all 0 where the MC total is not positive.

    The errors are the Garwood interval on the event count over the MC total,
    so they stay right when the contents are per GeV (*per_unit*).
    """
    data_vals = data.values()
    errs = data_errors(data, per_unit=per_unit)
    ok = tot_vals > 0
    ratio = np.divide(data_vals, tot_vals, out=np.zeros_like(data_vals, dtype=float), where=ok)
    ratio_err = np.divide(errs, tot_vals, out=np.zeros_like(errs, dtype=float), where=ok)
    return ratio, ratio_err


def plot_stack(
    region: Region,
    variable: Variable,
    *,
    stack_list: list[Hist],
    stack_colors: list[str],
    stack_labels: list[str],
    data_hist: list[Hist],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    lumi: float,
    com: float = 13.6,
    ratio_ylim: tuple[float, float] = (0.5, 2.0),
    syst_variations: dict[str, dict[str, Hist]] | None = None,
    fontsize_title: int = FONT_SIZE_TITLE,
    fontsize_label: int = FONT_SIZE_LABEL,
    fontsize_legend: int = FONT_SIZE_LEGEND,
    show_data: bool = True,
    signal_hists: dict[str, Hist] | None = None,
    extra_label: str | None = None,
    per_unit: bool = False,
) -> Figure:
    """
    Draw stacked MC + data with ratio panel.

    Args:
      syst_variations: {syst: {"up": Hist, "down": Hist}} totals of the whole MC
        stack (see wrplotter.systematics.total_variations); each source adds its
        largest shift in quadrature to the simulation's statistical band.
      show_data: If False, data points are not shown and ratio panel displays "Blinded"
      signal_hists: Dict of signal sample name -> histogram to overlay (not stacked)
    """
    if signal_hists is None:
        signal_hists = {}
    bkg_stack  = list(stack_list or [])
    bkg_labels = stack_labels
    bkg_colors = stack_colors
    tot = _safe_sum_hists(bkg_stack)
    data = _safe_sum_hists(list(data_hist or []))

    if tot is not None:
        edges = tot.axes[0].edges
    elif data is not None:
        logging.warning(
            "No MC histograms found for region '%s' variable '%s' (plotting data-only).",
            region.name, variable.name,
        )
        edges = data.axes[0].edges
    else:
        raise ValueError(f"No histograms to plot for region='{region.name}' variable='{variable.name}'.")

    hep.style.use("CMS")
    fig, (ax, rax) = plt.subplots(
        2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), sharex=True
    )
    # Horizontal bars on data points only where the bin width varies (PubGuidelines);
    # the same bars in both panels and in the legend.
    widths = np.diff(edges)
    data_xerr = not np.allclose(widths, widths[0])

    # main panel: the palette colours as given, no transparency (CAT colour scheme)
    if bkg_stack:
        hep.histplot([for_drawing(h) for h in bkg_stack], stack=True, label=bkg_labels,
                     color=bkg_colors, histtype="fill", ax=ax,
                     flow="hint")

    if show_data and data is not None:
        hep.histplot(for_drawing(data), yerr=data_errors(data, per_unit=per_unit),
                     label="Data", xerr=data_xerr, color="k",
                     histtype="errorbar", ax=ax, flow="hint")

    # --- Signal overlays ---
    for idx, (sig_name, sig_hist) in enumerate(signal_hists.items()):
        hep.histplot(for_drawing(sig_hist), label=signal_label(sig_name),
                     color=SIGNAL_COLORS[idx % len(SIGNAL_COLORS)],
                     linestyle=SIGNAL_STYLES[idx % len(SIGNAL_STYLES)],
                     histtype="step", linewidth=2, ax=ax,
                     flow="hint")

    # --- stat + syst uncertainties ---
    errps = {"hatch": "////", "facecolor": "none", "lw": 0, "edgecolor": "k", "alpha": 0.5}

    if tot is not None:
        tot_vals = tot.values()
        tot_vars = tot.variances()
        syst_err2, has_syst = band_error2(tot_vals, syst_variations)
        mc_total_errs = np.sqrt(tot_vars + syst_err2)

        # "Sim.", as the ratio panel's "Data/Sim.": the PubGuidelines keep "MC" for an
        # adjective ("Monte Carlo program"), never a noun.
        unc_label = "Sim. stat. + syst. unc." if has_syst else "Sim. stat. unc."
        hep.histplot(tot, histtype="band", yerr=mc_total_errs, ax=ax, **errps, label=unc_label)

        # ratio
        rel_err = np.divide(mc_total_errs, tot_vals,
                            out=np.zeros_like(tot_vals, dtype=float),
                            where=(tot_vals > 0))

        if show_data and data is not None:
            ratio, ratio_err = data_ratio(data, tot_vals, per_unit=per_unit)
            hep.histplot(ratio, edges, yerr=ratio_err, xerr=data_xerr, ax=rax,
                         histtype="errorbar", color="k", label="Data")

        band_low, band_high = 1 - rel_err, 1 + rel_err
        bad = tot_vals <= 0
        band_low[bad] = np.nan
        band_high[bad] = np.nan
        rax.stairs(band_low, edges, baseline=band_high, **errps)
    else:
        # No MC: keep the ratio panel but indicate missing denominator.
        rax.text(0.5, 0.5, "No simulation", transform=rax.transAxes,
                 ha="center", va="center", fontsize=18, fontweight="bold", color="red")

    # cosmetics
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylim(*ylim)
    alias = region.tlatex_alias.replace("\\n", "\n")
    if extra_label:
        alias = f"{alias}\n{extra_label}"
    # Blinded or data-less plots show simulation only: "CMS Simulation".
    cms_label(ax, lumi, com, data=show_data and data is not None, text=alias,
              fontsize=fontsize_label, text_fontsize=fontsize_title)

    xlabel = f"{variable.tlatex_alias} [{variable.unit}]" if getattr(variable, "unit", "") else variable.tlatex_alias
    rax.set_xlabel(xlabel)
    ax.set_xlabel("")

    if show_data:
        rax.set_ylabel("Data/Sim.")
    else:
        rax.set_ylabel("")
        ymid = (ratio_ylim[0] + ratio_ylim[1]) / 2
        xmid = (xlim[0] + xlim[1]) / 2
        rax.text(xmid, ymid, "Blinded", ha="center", va="center",
                 fontsize=24, fontweight="bold", color="red")

    rax.set_ylim(*ratio_ylim)
    y_range = ratio_ylim[1] - ratio_ylim[0]
    if y_range <= 2:
        step = 0.5
    elif y_range <= 4:
        step = 1.0
    else:
        step = 2.0
    yticks = np.arange(ratio_ylim[0], ratio_ylim[1] + step/2, step)
    yticks = yticks[yticks <= ratio_ylim[1]]
    rax.set_yticks(yticks)
    rax.axhline(1.0, ls="--", color="k")
    ax.set_xlim(*xlim)
    # y label (per-bin width)
    set_y_label(ax, tot if tot is not None else data, variable, per_unit=per_unit)

    reorder_legend(ax, fontsize=fontsize_legend)
    fit_header(ax)
    return fig


# ------------------------------ shared cosmetics ----------------------------- #

def cms_label(ax, lumi: float | None, com: float = 13.6, *, data: bool = True,
              text: str | None = None, rlabel: str | None = None,
              fontsize: int = FONT_SIZE_LABEL, text_fontsize: int = FONT_SIZE_TITLE):
    """The CMS label as the CMS figure guidelines ask: "CMS" in bold inside the frame,
    top left, the same distance from both axes, with the italic secondary label
    (``Simulation`` first when *data* is False) under it; "<lumi> fb^-1 (<com> TeV)",
    or *rlabel*, above the frame on the right.  *text* (the region description) goes
    under the label.  Returns the lowest text artist of the block.
    """
    pos = ax.get_position()
    fig_w, fig_h = ax.figure.get_size_inches()
    aspect = (pos.width * fig_w) / (pos.height * fig_h)
    right = {"rlabel": rlabel} if rlabel is not None else {"lumi": f"{lumi:.2f}", "com": com}
    _, suffix, _ = hep.cms.label(CMS_EXTRA_TEXT, loc=2, ax=ax, data=data, fontsize=fontsize,
                                 pad=0.05 * (aspect - 1), **right)
    if text:
        return region_text(ax, text, fontsize=text_fontsize, below=suffix)
    return suffix


def _content_vertices(ax) -> np.ndarray:
    """(x, y) data points of everything drawn in *ax* (lines, bars, fills, steps)."""
    pts = []
    for line in ax.lines:
        if line.get_transform().contains_branch(ax.transData):
            pts.append(line.get_path().interpolated(50).vertices)
    for coll in ax.collections:
        if isinstance(coll, PathCollection):
            pts.append(np.asarray(coll.get_offsets()))
        elif coll.get_transform().contains_branch(ax.transData):
            pts += [p.interpolated(50).vertices for p in coll.get_paths() if len(p.vertices)]
    for patch in ax.patches:
        if len(patch.get_path().vertices) and patch.get_data_transform().contains_branch(ax.transData):
            pts.append(patch.get_patch_transform().transform(
                patch.get_path().interpolated(50).vertices))
    pts = [np.asarray(p, dtype=float).reshape(-1, 2) for p in pts if len(p)]
    return np.concatenate(pts) if pts else np.empty((0, 2))


def fit_header(ax, *, margin: float = 0.02) -> None:
    """Keep the CMS label, the region text and the legend clear of the plot content.

    Puts "Simulation" on its own line when the secondary label runs into the
    legend, then raises the top of the y axis until nothing drawn lies under any
    of them.  The CMS label and the region text always count; a legend counts only
    when it hangs from the top of the frame: raising the axis cannot clear a legend
    placed low (limit plots put theirs lower left).
    Call it last, once the legend and the x range are set.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    legend = ax.get_legend()
    if legend is not None:
        leg_box = legend.get_window_extent(renderer)
        for t in ax.texts:
            if (isinstance(t, ExpSuffix) and t.get_text().startswith("Simulation ")
                    and t.get_window_extent(renderer).overlaps(leg_box)):
                t.set_text(t.get_text().replace("Simulation ", "Simulation\n", 1))
                # the region text hangs from this label: lay the figure out again so its extent below
                # is measured where it now sits, one line lower
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()

    header = [t for t in ax.texts
              if isinstance(t, (ExpText, ExpSuffix)) or getattr(t, "is_region_text", False)]
    blockers = [t.get_window_extent(renderer) for t in header if t.get_text().strip()]
    n_header = len(blockers)
    if legend is not None:
        legend = _fit_legend(ax, legend, blockers, renderer, margin)
        blockers.append(legend.get_window_extent(renderer))
    pts = _content_vertices(ax)
    log = ax.get_yscale() == "log"
    if log:
        pts = pts[pts[:, 1] > 0]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if not len(pts):
        return

    lo, hi = ax.get_ylim()
    f = np.log10 if log else (lambda v: v)
    to_axes = ax.transAxes.inverted()
    to_data = ax.transData.inverted()
    # The top moves up at most to twice the axis span: a header that needs more
    # (a legend pushed low in the frame) is left as it is, with the warning above.
    need, cap = f(hi), f(lo) + 2 * (f(hi) - f(lo))
    for i, box in enumerate(blockers):
        bottom = to_axes.transform((box.x0, box.y0))[1]
        top = to_axes.transform((box.x1, box.y1))[1]
        if not 0 < bottom < 1:                       # out of the frame (luminosity)
            continue
        if i >= n_header and top < 0.75:             # a legend not hanging from the top of the frame
            continue
        x0 = to_data.transform((box.x0, box.y0))[0]
        x1 = to_data.transform((box.x1, box.y0))[0]
        under = pts[(pts[:, 0] >= min(x0, x1)) & (pts[:, 0] <= max(x0, x1)), 1]
        if not len(under) or (room := bottom - margin) <= 0:
            continue
        top = f(lo) + (f(under.max()) - f(lo)) / room
        if np.isfinite(top):
            need = max(need, min(top, cap))
    if need > f(hi):
        ax.set_ylim(lo, 10 ** need if log else need)


LEGEND_MIN_FONTSIZE = 12


def _fit_legend(ax, legend, blockers, renderer, margin):
    """The legend clear of the label block *blockers*: its font shrinks (to
    LEGEND_MIN_FONTSIZE) until it fits beside them; failing that it goes under
    them, with a warning.  Returns the legend, re-made when the font changed."""
    def overlaps(lg):
        box = lg.get_window_extent(renderer)
        return any(b.overlaps(box) for b in blockers)

    if not overlaps(legend):
        return legend
    handles, labels = legend.legend_handles, [t.get_text() for t in legend.texts]
    ncols = getattr(legend, "_ncols", 1)
    size = legend.texts[0].get_fontsize() if legend.texts else LEGEND_MIN_FONTSIZE
    while overlaps(legend) and size > LEGEND_MIN_FONTSIZE:
        size = max(size - 1, LEGEND_MIN_FONTSIZE)
        legend.remove()
        legend = ax.legend(handles, labels, loc="upper right", fontsize=size, ncols=ncols)
    if overlaps(legend):
        lowest = min(ax.transAxes.inverted().transform((0, b.y0))[1] for b in blockers)
        legend.set_bbox_to_anchor((1, lowest - margin), transform=ax.transAxes)
        logging.warning("Legend of %d entries does not fit beside the CMS label even at %d pt; "
                        "placed under it. Shorten its labels.", len(labels), LEGEND_MIN_FONTSIZE)
    return legend


def region_text(ax, text: str, *, fontsize: int = FONT_SIZE_TITLE, x: float = 0.05,
                y: float = 0.96, below=None):
    """Multi-line region description (literal '\\n' allowed): under the *below* text
    artist when given (the CMS label), else at (*x*, *y*) in axes fraction."""
    text = text.replace("\\n", "\n")
    if below is not None:
        ann = ax.annotate(text, xy=(0, 0), xycoords=below, xytext=(0, -0.5 * fontsize),
                          textcoords="offset points", fontsize=fontsize, ha="left", va="top")
        ann.is_region_text = True        # fit_header keeps the content clear of it
        return ann
    return ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize, va="top")


def signal_label(name: str) -> str:
    """'signal_WR4000_N2100' / 'WR4000_N2100' -> '(m_WR, m_N) = (4000, 2100) GeV'."""
    match = re.search(r"WR(\d+)_N(\d+)", name)
    if not match:
        return name
    m_wr, m_n = match.groups()
    return rf"$(m_{{W_R}}, m_N) = ({m_wr}, {m_n})$ GeV"


def log_ylim(arrays, *, pad_low: float = 0.5, pad_high: float = 50.0,
             floor: float = 1e-3) -> tuple[float, float]:
    """Log-scale y range covering the positive values of *arrays*."""
    pos = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays]) if arrays else np.array([])
    pos = pos[np.isfinite(pos) & (pos > 0)]
    if pos.size == 0:
        return floor, 1.0
    return max(pos.min() * pad_low, floor), pos.max() * pad_high
