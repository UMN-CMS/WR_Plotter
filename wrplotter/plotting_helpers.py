from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
import mplhep as hep

from .histogram_utils import for_drawing, sum_hists
from .systematics import band_error2

if TYPE_CHECKING:
    from hist import Hist
    from .regions import Region
    from .variables import Variable

# ----------------------------- constants -------------------------------------- #

FONT_SIZE_TITLE  = 20
FONT_SIZE_LABEL  = 20
FONT_SIZE_LEGEND = 18

# Colours and linestyles cycled over multiple signal overlays so they remain
# visually distinguishable when more than one signal is shown on the same plot.
SIGNAL_COLORS = ["black", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
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
    For non-uniform binning, 'Events / bin', or 'Events / [unit]' when the contents
    were divided by the bin width (*per_unit*).
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
        ax.set_ylabel(f"Events / {unit}" if unit else "Events / unit")
    else:
        ax.set_ylabel("Events / bin")

# --------------------------------- legends ---------------------------------- #

def reorder_legend(ax, priority=("MC stat.", "Data"), fontsize=18) -> None:
    """
    Put MC uncertainty band first, 'Data' second, keep the rest in reverse stack order.
    The first priority element is matched as a prefix (handles both
    'MC stat. unc.' and 'MC stat. + syst. unc.').
    Missing labels are skipped gracefully.
    """
    handles, labels = ax.get_legend_handles_labels()
    idx = {label: i for i, label in enumerate(labels)}
    # Prefix match for the MC uncertainty label
    first = next((idx[l] for l in labels if l.startswith(priority[0])), None)
    second = idx.get(priority[1])
    rest = [i for i in range(len(labels)) if i not in {first, second}]
    order = ([first] if first is not None else []) + \
            ([second] if second is not None else []) + rest
    if not order:
        return
    ax.legend([handles[i] for i in order],
              [labels[i] for i in order],
              loc="best", fontsize=fontsize)

# --------------------------------- drawing ---------------------------------- #

def data_ratio(data: Hist, tot_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Data/MC ratio per bin and its statistical error (both 0 where the MC total is not positive).

    The error uses the data's own variances, sqrt(N) per bin, so it stays right
    when the contents are per GeV (``--per-gev`` divides the variances by the
    bin width squared); sqrt of the per-GeV contents was too large by sqrt(width).
    """
    data_vals = data.values()
    data_vars = data.variances()
    data_err = np.sqrt(data_vars if data_vars is not None else data_vals)
    ok = tot_vals > 0
    ratio = np.divide(data_vals, tot_vals, out=np.zeros_like(data_vals, dtype=float), where=ok)
    ratio_err = np.divide(data_err, tot_vals, out=np.zeros_like(data_vals, dtype=float), where=ok)
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
        largest shift in quadrature to the MC stat. band.
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

    # main panel
    if bkg_stack:
        hep.histplot([for_drawing(h) for h in bkg_stack], stack=True, label=bkg_labels,
                     color=bkg_colors, histtype="fill", alpha=0.7, ax=ax,
                     flow="hint")

    if show_data and data is not None:
        hep.histplot(for_drawing(data), label="Data", xerr=True, color="k",
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

        unc_label = "MC stat. + syst. unc." if has_syst else "MC stat. unc."
        hep.histplot(tot, histtype="band", yerr=mc_total_errs, ax=ax, **errps, label=unc_label)

        # ratio
        rel_err = np.divide(mc_total_errs, tot_vals,
                            out=np.zeros_like(tot_vals, dtype=float),
                            where=(tot_vals > 0))

        if show_data and data is not None:
            ratio, ratio_err = data_ratio(data, tot_vals)
            hep.histplot(ratio, edges, yerr=ratio_err, xerr=False, ax=rax,
                         histtype="errorbar", color="k", capsize=4, label="Data")

        band_low, band_high = 1 - rel_err, 1 + rel_err
        bad = tot_vals <= 0
        band_low[bad] = np.nan
        band_high[bad] = np.nan
        rax.stairs(band_low, edges, baseline=band_high, **errps)
    else:
        # No MC: keep the ratio panel but indicate missing denominator.
        rax.text(0.5, 0.5, "No MC", transform=rax.transAxes,
                 ha="center", va="center", fontsize=18, fontweight="bold", color="red")

    # cosmetics
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylim(*ylim)
    alias = region.tlatex_alias.replace("\\n", "\n")
    if extra_label:
        alias = f"{alias}\n{extra_label}"
    ax.text(
        0.05, 0.96, alias,
        transform=ax.transAxes,
        fontsize=fontsize_title,
        va="top"
    )

    hep.cms.label(loc=0, ax=ax, data=True,
                  label="Work in Progress", lumi=f"{lumi:.2f}",
                  com=com, fontsize=fontsize_label)

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
    return fig


# ------------------------------ shared cosmetics ----------------------------- #

def cms_label(ax, lumi: float, com: float = 13.6, *, fontsize: int = FONT_SIZE_LABEL) -> None:
    """CMS 'Work in Progress' header with luminosity and centre-of-mass energy."""
    hep.cms.label(loc=0, ax=ax, data=True, label="Work in Progress",
                  lumi=f"{lumi:.2f}", com=com, fontsize=fontsize)


def region_text(ax, text: str, *, fontsize: int = FONT_SIZE_TITLE, x: float = 0.05,
                y: float = 0.96) -> None:
    """Multi-line region description in the top-left corner (literal '\\n' allowed)."""
    ax.text(x, y, text.replace("\\n", "\n"), transform=ax.transAxes,
            fontsize=fontsize, va="top")


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
