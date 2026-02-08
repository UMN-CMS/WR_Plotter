from __future__ import annotations

import logging
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep

# ----------------------------- constants -------------------------------------- #

FONT_SIZE_TITLE  = 20
FONT_SIZE_LABEL  = 20
FONT_SIZE_LEGEND = 18

# ----------------------------- formatters/labels ----------------------------- #

def _safe_sum_hists(hists):
    """
    Sum a list of hist-like objects without letting Python's `sum()` default start=0
    turn empty inputs into an int.

    Returns:
      - None for empty input
      - the sole element for length-1 input
      - the pairwise sum for length>=2
    """
    if not hists:
        return None
    total = hists[0]
    for h in hists[1:]:
        total = total + h
    return total

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
    For non-uniform binning, use 'Events / X [unit]'.
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
            ax.set_ylabel(f"Events / X {unit}")
        else:
            ax.set_ylabel("Events / X")

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

def plot_stack(region, variable, *,
               stack_list, stack_colors, stack_labels, data_hist,
               xlim, ylim, lumi, com=13.6,
               ratio_ylim=(0.5, 2.0), syst_hists=None,
               fontsize_title=FONT_SIZE_TITLE, fontsize_label=FONT_SIZE_LABEL, fontsize_legend=FONT_SIZE_LEGEND,
               show_data=True, signal_hists=None):
    """
    Draw stacked MC + data with ratio panel.

    Args:
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
        hep.histplot(bkg_stack, stack=True, label=bkg_labels,
                     color=bkg_colors, histtype="fill", alpha=0.7, ax=ax)

    if show_data and data is not None:
        hep.histplot(data, label="Data", xerr=True, color="k",
                     histtype="errorbar", ax=ax)

    # --- Signal overlays ---
    for idx, (sig_name, sig_hist) in enumerate(signal_hists.items()):
        match = re.search(r'WR(\d+)_N(\d+)', sig_name)
        if match:
            m_wr, m_n = match.groups()
            label = rf"$(m_{{W_R}}, m_N) = ({m_wr}, {m_n})$ GeV"
        else:
            label = sig_name

        hep.histplot(sig_hist, label=label, color="black",
                     histtype="step", linewidth=2, linestyle="--", ax=ax)

    # --- stat + syst uncertainties ---
    errps = {"hatch": "////", "facecolor": "none", "lw": 0, "edgecolor": "k", "alpha": 0.5}

    if tot is not None:
        tot_vals = tot.values()
        tot_vars = tot.variances()
        syst_err2 = np.zeros_like(tot_vals)
        has_syst = False

        if syst_hists:
            systs_for_region = syst_hists.get(region.name, {}).get(variable.name, {})
            for syst, dirs in systs_for_region.items():
                if "up" in dirs and "down" in dirs:
                    up_total = sum((h.values(flow=False) for h in dirs["up"].values()), start=0)
                    down_total = sum((h.values(flow=False) for h in dirs["down"].values()), start=0)

                    delta_up = np.abs(up_total - tot_vals)
                    delta_down = np.abs(down_total - tot_vals)
                    delta = np.maximum(delta_up, delta_down)

                    if np.any(delta > 0):
                        has_syst = True
                    syst_err2 += delta**2

        mc_total_errs = np.sqrt(tot_vars + syst_err2)

        unc_label = "MC stat. + syst. unc." if has_syst else "MC stat. unc."
        hep.histplot(tot, histtype="band", yerr=mc_total_errs, ax=ax, **errps, label=unc_label)

        # ratio
        rel_err = np.divide(mc_total_errs, tot_vals,
                            out=np.zeros_like(tot_vals, dtype=float),
                            where=(tot_vals > 0))

        if show_data and data is not None:
            data_vals = data.values()
            ratio = np.divide(data_vals, tot_vals,
                              out=np.zeros_like(data_vals, dtype=float),
                              where=(tot_vals > 0))
            ratio_err = np.divide(np.sqrt(data_vals), tot_vals,
                                  out=np.zeros_like(data_vals, dtype=float),
                                  where=(tot_vals > 0))

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
    ax.text(
        0.05, 0.96, alias,
        transform=ax.transAxes,
        fontsize=fontsize_title,
        va="top"
    )

    hep.cms.label(loc=0, ax=ax, data=region.unblind_data,
                  label="Work in Progress", lumi=f"{lumi:.1f}",
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
    rax.set_yticks(yticks)
    rax.axhline(1.0, ls="--", color="k")
    ax.set_xlim(*xlim)
    # y label (per-bin width)
    set_y_label(ax, tot if tot is not None else data, variable)

    reorder_legend(ax, fontsize=fontsize_legend)
    return fig
