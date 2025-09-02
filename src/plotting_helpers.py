# src/plotting_helpers.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep

# ----------------------------- formatters/labels ----------------------------- #

def custom_log_formatter(y: float, pos) -> str:
    """
    Format major ticks on a log y-axis:
      1 -> '1', 10 -> '10', others -> '10^{n}'
    Gracefully handle non-positive y (matplotlib may ask).
    """
    if y <= 0:
        return ""
    # exact 1 or 10 shown plainly
    if np.isclose(y, 1.0):
        return "1"
    if np.isclose(y, 10.0):
        return "10"
    n = int(np.round(np.log10(y)))
    # show only for powers of 10; empty for others to avoid clutter
    if np.isclose(y, 10**n):
        return rf"$10^{{{n}}}$"
    return ""

def _format_bin_width(w: float) -> str:
    return f"{int(w)}" if float(w).is_integer() else f"{w:.1f}"

def set_y_label(ax, tot, variable) -> None:
    """
    Set y-axis label as 'Events / <binwidth> [unit]' for uniform binning.
    For non-uniform binning, fall back to 'Events / bin'.
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
    else:
        if unit:
            ax.set_ylabel(f"Events / bin ({unit})")
        else:
            ax.set_ylabel("Events / bin")

# --------------------------------- legends ---------------------------------- #

def reorder_legend(ax, priority=("MC stat. unc.", "Data"), fontsize=18) -> None:
    """
    Put 'MC stat. unc.' first, 'Data' second, keep the rest in reverse stack order.
    Missing labels are skipped gracefully.
    """
    handles, labels = ax.get_legend_handles_labels()
    idx = {label: i for i, label in enumerate(labels)}
    first  = idx.get(priority[0])
    second = idx.get(priority[1])
    rest = [i for i in range(len(labels)) if i not in {first, second}]
    rest.reverse()  # stacks read from bottom to top
    order = ([first] if first is not None else []) + \
            ([second] if second is not None else []) + rest
    if not order:
        return
    ax.legend([handles[i] for i in order],
              [labels[i] for i in order],
              loc="best", fontsize=fontsize)

# --------------------------------- drawing ---------------------------------- #

def plot_stack(plotter, region, variable,
               fontsize_title=20, fontsize_label=20, fontsize_legend=18):
    """
    Draw stacked MC + data with ratio panel.
    Expects Plotter to provide:
      - stack_list, stack_labels, stack_colors, data_hist
      - xlim, ylim, lumi
      - a y-label helper: set_y_label(ax, tot, variable)  (we call the version above)
    """
    bkg_stack  = plotter.stack_list
    bkg_labels = plotter.stack_labels
    bkg_colors = plotter.stack_colors
    tot  = sum(plotter.stack_list)
    data = sum(plotter.data_hist)
    edges = tot.axes[0].edges

    hep.style.use("CMS")
    fig, (ax, rax) = plt.subplots(
        2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), sharex=True
    )

    # main panel
    hep.histplot(bkg_stack, stack=True, label=bkg_labels,
                 color=bkg_colors, histtype="fill", ax=ax)
    hep.histplot(data, label="Data", xerr=True, color="k",
                 histtype="errorbar", ax=ax)

    # MC stat uncertainty band
    errps = {"hatch": "////", "facecolor": "none", "lw": 0, "edgecolor": "k", "alpha": 0.5}
    hep.histplot(tot, histtype="band", ax=ax, **errps, label="MC stat. unc.")

    # ratio
    data_vals = data.values()
    tot_vals  = tot.values()
    ratio = np.divide(data_vals, tot_vals,
                      out=np.zeros_like(data_vals, dtype=float),
                      where=(tot_vals > 0))
    ratio_err = np.divide(np.sqrt(data_vals), tot_vals,
                          out=np.zeros_like(data_vals, dtype=float),
                          where=(tot_vals > 0))

    tot_vars = tot.variances()
    mc_err = np.sqrt(tot_vars)
    rel_err = np.divide(mc_err, tot_vals,
                        out=np.zeros_like(tot_vals, dtype=float),
                        where=(tot_vals > 0))

    hep.histplot(ratio, edges, yerr=ratio_err, xerr=True, ax=rax,
                 histtype="errorbar", color="k", capsize=4, label="Data")
    band_low, band_high = 1 - rel_err, 1 + rel_err
    bad = tot_vals <= 0
    band_low[bad] = np.nan
    band_high[bad] = np.nan
    rax.stairs(band_low, edges, baseline=band_high, **errps)

    # cosmetics
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylim(*plotter.ylim)
    alias = region.tlatex_alias.replace("\\n", "\n")
    ax.text(
        0.05, 0.96, alias,
        transform=ax.transAxes,
        fontsize=fontsize_title,
        va="top"
    )

    hep.cms.label(loc=0, ax=ax, data=region.unblind_data,
                  label="Work in Progress", lumi=f"{plotter.lumi:.1f}",
                  com=13.6, fontsize=fontsize_label)

    xlabel = f"{variable.tlatex_alias} [{variable.unit}]" if getattr(variable, "unit", "") else variable.tlatex_alias
    rax.set_xlabel(xlabel)
    ax.set_xlabel("")
    rax.set_ylabel("Data/Sim.")
    rax.set_ylim(0.7, 1.3)
    rax.set_yticks([0.8, 1.0, 1.2])
    rax.axhline(1.0, ls="--", color="k")
    ax.set_xlim(*plotter.xlim)

    # y label (per-bin width)
    set_y_label(ax, tot, variable)

    reorder_legend(ax, fontsize=fontsize_legend)
    return fig
