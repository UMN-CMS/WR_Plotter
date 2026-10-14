"""Event-weight distributions (bin/plot_event_weights.py).

The analyzer writes, for MC, ``event_weights/<region>/positive`` and
``event_weights/<region>/negative``: the number of events per bin of
log10|w|, where w is the per-region nominal event weight (the weight that
fills the region's histograms), split by sign.  Entries outside the axis range
sit in the flow bins and are counted.

The effective number of events, N_eff = (sum w)^2 / sum w^2, comes from a
nominal histogram of the region (every event fills each 1D histogram once).

``event_weights/<region>/components/`` holds one distribution per factor of
that weight, again as event counts: the generator weight
(``genweight_positive`` / ``genweight_negative``, log10|genWeight| split by
sign) and the xsec * lumi / sum(genWeight) normalization (``normalization``,
log10) on log axes; every multiplicative correction (pileup, lepton SFs, DY
reshape, ...) and their product (``corrections``) on a linear axis.  The
total weight is genweight * normalization * corrections.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
from hist import Hist
from matplotlib.figure import Figure

from .histogram_cache import _open_root, has_key, read_hist
from .plotting_helpers import FONT_SIZE_LEGEND, cms_label, custom_log_formatter, fit_header, log_ylim

SIGNS = ("positive", "negative")

COMPONENTS_DIR = "components"
GENWEIGHT = "genweight"            # stored as genweight_positive / genweight_negative
NORMALIZATION = "normalization"
CORRECTIONS = "corrections"
LOG_COMPONENTS = (GENWEIGHT, NORMALIZATION)

# Legend/axis names of the components; anything else is spelled out from its
# CamelCase analyzer label (e.g. "LooseMuonIdSf" -> "Loose muon ID SF").
COMPONENT_LABELS = {
    GENWEIGHT: "Generator weight",
    NORMALIZATION: r"$\sigma \cdot L\,/\,\sum w_{\mathrm{gen}}$",
    CORRECTIONS: "Product of all corrections",
    "pileup": "Pileup weight",
    "dy_zpt": r"DY Z $p_T$ reweighting",
    "dy_reshape": "DY reshape",
    "signal_kfactor": "Signal NLO/LO k-factor (shape)",
    "ElectronIdSf": "Electron HEEP ID SF",
    "LsfSf": "LSF SF",
}

# A region histogram every event fills once, for sum(w) and sum(w^2).
SUMS_VARIABLE = "mass_dilepton"


def weights_key(region: str, sign: str) -> str:
    return f"event_weights/{region}/{sign}"


def has_event_weights(path: str | Path) -> bool:
    """True if the file has event-weight histograms (a file that cannot be read raises)."""
    return has_key(path, "event_weights")


def _read(path, key) -> Hist | None:
    return read_hist(path, key)


@dataclass
class WeightDistribution:
    """Event-weight distribution of one sample (or group) in one region."""

    positive: Hist | None = None
    negative: Hist | None = None
    sumw: float = 0.0
    sumw2: float = 0.0

    def __add__(self, other: "WeightDistribution") -> "WeightDistribution":
        def add(a, b):
            return b if a is None else (a if b is None else a + b)
        return WeightDistribution(add(self.positive, other.positive),
                                  add(self.negative, other.negative),
                                  self.sumw + other.sumw, self.sumw2 + other.sumw2)

    def count(self, sign: str) -> float:
        h = getattr(self, sign)
        return 0.0 if h is None else float(h.values(flow=True).sum())

    @property
    def n_events(self) -> float:
        return self.count("positive") + self.count("negative")

    @property
    def negative_fraction(self) -> float:
        n = self.n_events
        return self.count("negative") / n if n > 0 else 0.0

    @property
    def n_eff(self) -> float:
        return self.sumw ** 2 / self.sumw2 if self.sumw2 > 0 else 0.0

    @property
    def empty(self) -> bool:
        return self.positive is None and self.negative is None


def read_distribution(files: Sequence[str | Path], region: str) -> WeightDistribution:
    """Weight distribution of one sample in *region*, summed over its files (sub-eras)."""
    total = WeightDistribution()
    for f in files:
        pos, neg = (_read(f, weights_key(region, s)) for s in SIGNS)
        sumw = sumw2 = 0.0
        nominal = _read(f, f"{region}/{SUMS_VARIABLE}_{region}")
        if nominal is not None:
            sumw = float(nominal.values(flow=True).sum())
            sumw2 = float(nominal.variances(flow=True).sum())
        total = total + WeightDistribution(pos, neg, sumw, sumw2)
    return total


def plot_weight_distributions(
    entries: Sequence[tuple[str, str, WeightDistribution]],
    *,
    text: str,
    lumi: float,
    com: float = 13.6,
) -> Figure:
    """log10|w| per entry (label, color, distribution): solid w > 0, dashed w < 0."""
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 10))
    drawn = []
    edges = None
    for label, color, dist in entries:
        stats = (f"N={dist.n_events:.3g}, $N_{{eff}}$={dist.n_eff:.3g}"
                 + (f", {100 * dist.negative_fraction:.1f}% w<0" if dist.count("negative") else ""))
        for sign, style in (("positive", "-"), ("negative", "--")):
            h = getattr(dist, sign)
            if h is None or h.values(flow=True).sum() == 0:
                continue
            edges = h.axes[0].edges
            hep.histplot(h.values(), edges, histtype="step", color=color, linestyle=style,
                         linewidth=2, ax=ax, flow=None,
                         label=f"{label}\n{stats}" if sign == "positive" else None)
            drawn.append(h.values())
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylim(*log_ylim(drawn, pad_high=1e3, floor=0.5))
    if edges is not None:
        nonzero = np.flatnonzero(np.sum([d for d in drawn], axis=0) > 0) if drawn else []
        if len(nonzero):
            ax.set_xlim(edges[max(nonzero[0] - 5, 0)], edges[min(nonzero[-1] + 6, len(edges) - 1)])
    ax.set_xlabel(r"$\log_{10}|w|$ (per-region nominal event weight)")
    ax.set_ylabel("Simulated events")
    cms_label(ax, lumi, com, data=False, text=text + "\nsolid: $w > 0$, dashed: $w < 0$")
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    fit_header(ax)
    return fig


# ---------------------------------------------------------------------------
# Weight components
# ---------------------------------------------------------------------------

def components_key(region: str) -> str:
    return f"event_weights/{region}/{COMPONENTS_DIR}"


def component_label(name: str) -> str:
    """Display name of a weight component."""
    if name in COMPONENT_LABELS:
        return COMPONENT_LABELS[name]
    words = re.findall(r"[A-Z][a-z]*|[a-z]+|\d+", name)
    if words and words[-1] == "Sf":
        words[-1] = "SF"
    words = [{"Id": "ID", "Trig": "trigger"}.get(w, w) for w in words]
    return " ".join([words[0]] + [w if w.isupper() else w.lower() for w in words[1:]]) if words else name


def component_names(stored_keys: Sequence[str]) -> list[str]:
    """Component names from stored keys (``genweight_positive`` -> ``genweight``),
    in first-seen order with ``corrections`` last."""
    names = []
    for key in stored_keys:
        name = GENWEIGHT if key.startswith(f"{GENWEIGHT}_") else key
        if name not in names:
            names.append(name)
    if CORRECTIONS in names:
        names.remove(CORRECTIONS)
        names.append(CORRECTIONS)
    return names


def read_components(files: Sequence[str | Path], region: str) -> dict[str, Hist]:
    """``{stored key: Hist}`` of *region*'s weight components, summed over *files*.

    Empty when the files predate the component histograms.
    """
    total: dict[str, Hist] = {}
    for f in files:
        if not has_key(f, components_key(region)):
            continue
        directory = _open_root(str(f))[components_key(region)]
        for key in directory.keys(recursive=False, cycle=False):
            h = directory[key].to_hist()
            total[key] = h if key not in total else total[key] + h
    return total


@dataclass
class ComponentStats:
    """Event-count summary of one component distribution (flow bins included in N)."""

    n: float
    mean: float
    rms: float
    outside: float        # fraction of events in the flow bins
    negative: float = 0.0  # fraction with a negative value (generator weight only)


def component_stats(hists: dict[str, Hist], name: str) -> ComponentStats | None:
    """Mean and RMS over the in-range bins (log10 for the log components)."""
    if name == GENWEIGHT:
        parts = [hists[k] for k in (f"{GENWEIGHT}_positive", f"{GENWEIGHT}_negative") if k in hists]
    else:
        parts = [hists[name]] if name in hists else []
    if not parts:
        return None
    n = sum(float(h.values(flow=True).sum()) for h in parts)
    if n <= 0:
        return ComponentStats(0.0, float("nan"), float("nan"), 0.0)
    centers = parts[0].axes[0].centers
    counts = np.sum([h.values() for h in parts], axis=0)
    in_range = counts.sum()
    mean = float(np.sum(counts * centers) / in_range) if in_range > 0 else float("nan")
    rms = (float(np.sqrt(np.sum(counts * (centers - mean) ** 2) / in_range))
           if in_range > 0 else float("nan"))
    negative = 0.0
    if name == GENWEIGHT and f"{GENWEIGHT}_negative" in hists:
        negative = float(hists[f"{GENWEIGHT}_negative"].values(flow=True).sum()) / n
    return ComponentStats(n, mean, rms, 1.0 - in_range / n, negative)


def plot_component(
    name: str,
    entries: Sequence[tuple[str, str, dict[str, Hist]]],
    *,
    text: str,
    lumi: float,
    com: float = 13.6,
) -> Figure | None:
    """One weight component for each entry (label, color, read_components result).

    Each entry is normalized to unit area so the shapes compare across samples
    of very different size; the legend gives N and the mean and RMS (in log10
    for the log components) on a line under each sample.  The generator weight is drawn solid for w > 0
    and dashed for w < 0.  Returns None when no entry has the component.
    """
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 10))
    drawn, edges = [], None
    log_axis = name in LOG_COMPONENTS
    for label, color, hists in entries:
        stats = component_stats(hists, name)
        if stats is None or stats.n <= 0:
            continue
        unit = "log10" if log_axis else ""
        legend = (f"{label}\nN={stats.n:.3g}, mean{'' if not unit else ' ' + unit}={stats.mean:.3g}, "
                  f"RMS={stats.rms:.2g}")
        if name == GENWEIGHT and stats.negative:
            legend += f", {100 * stats.negative:.1f}% w<0"
        if stats.outside > 0:
            legend += f", {100 * stats.outside:.2g}% off axis"
        keys = ((f"{GENWEIGHT}_positive", "-"), (f"{GENWEIGHT}_negative", "--")) if name == GENWEIGHT \
            else ((name, "-"),)
        first = True
        for key, style in keys:
            h = hists.get(key)
            if h is None or h.values().sum() == 0:
                continue
            values = h.values() / stats.n
            edges = h.axes[0].edges
            hep.histplot(values, edges, histtype="step", color=color, linestyle=style,
                         linewidth=2, ax=ax, flow=None, label=legend if first else None)
            first = False
            drawn.append(values)
    if not drawn:
        plt.close(fig)
        return None

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylim(*log_ylim(drawn, pad_high=1e3, floor=1e-7))
    nonzero = np.flatnonzero(np.sum([np.pad(d, (0, len(edges) - 1 - len(d))) for d in drawn], axis=0) > 0)
    if len(nonzero):
        pad = max(3, (nonzero[-1] - nonzero[0]) // 10)
        ax.set_xlim(edges[max(nonzero[0] - pad, 0)], edges[min(nonzero[-1] + 1 + pad, len(edges) - 1)])
    if log_axis:
        ax.set_xlabel(rf"$\log_{{10}}|${component_label(name)}$|$" if name == GENWEIGHT
                      else rf"$\log_{{10}}$ {component_label(name)}")
    else:
        ax.set_xlabel(component_label(name))
    ax.set_ylabel("Fraction of simulated events")
    note = "\nsolid: $w > 0$, dashed: $w < 0$" if name == GENWEIGHT else ""
    cms_label(ax, lumi, com, data=False, text=text + note)
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    fit_header(ax)
    return fig


def format_summary(
    region_label: str,
    totals: Sequence[tuple[str, WeightDistribution]],
    components: Sequence[tuple[str, dict[str, Hist]]],
) -> str:
    """Plain-text summary of one region: the total weight per sample, then
    every component per sample."""
    lines = [f"Event weights: {region_label}", "",
             f"{'Sample':<28s} {'N':>12s} {'N_eff':>12s} {'sum w':>12s} {'w<0':>7s}"]
    for label, dist in totals:
        if dist.empty:
            continue
        lines.append(f"{label:<28s} {dist.n_events:12.6g} {dist.n_eff:12.6g} {dist.sumw:12.6g} "
                     f"{100 * dist.negative_fraction:6.2f}%")
    names = component_names([k for _, hists in components for k in hists])
    for name in names:
        unit = " (log10)" if name in LOG_COMPONENTS else ""
        lines += ["", f"{component_label(name)} [{name}]{unit}",
                  f"  {'Sample':<26s} {'N':>12s} {'mean':>10s} {'RMS':>10s} {'off axis':>9s}"
                  + ("  w<0" if name == GENWEIGHT else "")]
        for label, hists in components:
            stats = component_stats(hists, name)
            if stats is None:
                continue
            row = (f"  {label:<26s} {stats.n:12.6g} {stats.mean:10.4g} {stats.rms:10.3g} "
                   f"{100 * stats.outside:8.3g}%")
            if name == GENWEIGHT:
                row += f"  {100 * stats.negative:.2f}%"
            lines.append(row)
    return "\n".join(lines) + "\n"
