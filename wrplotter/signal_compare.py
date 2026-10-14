"""Signal samples of several eras side by side: efficiencies and shapes (see bin/compare_signal_eras.py).

An *era set* is an era or a combination of eras (data/lumi.yaml) read from one
analysis directory, e.g. ``2022`` from 20261008_signals; the histograms of its
eras are added, as everywhere in the plotter.  A mass point counts for a set
only when every era of the set has its file.

Efficiency
    The acceptance x efficiency of a channel is its expected signal-region
    yield over the signal events produced in the channel's lepton flavour,
    sigma x L.  The yield is the integral of the region's mass histogram, with
    every weight the analyzer applies (lepton scale factors included).  sigma
    is the cross section of the datasets the file's manifest lists, taken from
    the analyzer's dataset configs (WrCoffea data/configs/<Run>/<Year>/<Era>/
    <Era>_signal.json), or from the fileset the analyzer ran with when one is
    given for the era, and L comes from data/lumi.yaml.  A WRtoNEltoElElJJ
    dataset counts for ee, a WRtoNMutoMuMuJJ one for mumu, and a flavour-mixed
    WRtoNLtoLLJJ one half for each.

    The cutflow rows carry the base weight only (no lepton scale factors).
    They are efficiencies of the channel's flavour: the leading rows that the
    ee and mumu cutflows share (no_cuts and the flavour-blind jet cuts) hold
    events of every flavour and are divided by sigma x L of all of them, the
    rows from the first lepton-flavour cut on by sigma x L of the channel's
    flavour.  The last row is then the channel's efficiency before scale
    factors, whatever mix of flavours the files hold.  Row 0 (no_cuts) is what
    the skim and the analyzer's event filters (MET filters, jet veto map) keep;
    the analyzer's skimmed signal filesets (WrCoffea data/filesets) separate
    the two: skim = nevts / genEventCount, filters = unweighted no_cuts / nevts.
    The efficiency is the product no_cuts x selection (last row / row 0) x
    scale factors (region yield / last row, the mean lepton scale-factor
    weight), which tells where an efficiency difference between eras arises;
    the analyzer's event_weights/<region>/components histograms split the last
    factor into its scale factors (reco, ID, isolation, trigger, ...).

Shape
    The signal-region mass (m_lljj resolved, m_lJ boosted) of each set, scaled
    to unit area, and the position and width of its peak: the median, the 16 and
    84 % quantiles, and sigma_eff, half the width of the narrowest interval that
    holds 68.3 % of the events.  A quantile or interval reaching into the
    overflow is undefined (NaN).  Statistical errors use the Gaussian
    approximations 1.2533 sigma_eff / sqrt(N_eff) for the median and
    sigma_eff / sqrt(2 N_eff) for sigma_eff, with N_eff = (sum w)^2 / sum w^2.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker
import mplhep as hep
import numpy as np
from hist import Hist
from matplotlib.figure import Figure

from .config import era_label, load_lumi
from .cutflow import cut_label, read_files
from .histogram_cache import _open_root
from .histogram_utils import rebin_histogram
from .inputs import Inputs, make_inputs, mass_point, read_manifest, root_file
from .paths import repo_root
from .plotting_helpers import (FONT_SIZE_LABEL, FONT_SIZE_LEGEND, FONT_SIZE_MIN, FONT_SIZE_TITLE, cms_label,
                               fit_header)
from .stack import hist_key

logger = logging.getLogger(__name__)

CHANNELS = ("ee", "mumu")
TOPOLOGIES = ("resolved", "boosted")
COMBINED = "combined"                     # resolved + boosted signal regions
MASS_VARIABLES = {"resolved": "mass_fourobject", "boosted": "mass_twoobject"}
MASS_LABELS = {"resolved": r"$m_{\ell\ell jj}$", "boosted": r"$m_{\ell J}$"}
CHANNEL_LABELS = {"ee": "ee", "mumu": r"$\mu\mu$"}
TOPOLOGY_LABELS = {"resolved": "Resolved SR", "boosted": "Boosted SR", COMBINED: "Resolved + boosted SR"}

# Share of a signal dataset's cross section in each channel, by process name.
FLAVOUR_SHARES = {
    "WRtoNEltoElElJJ": {"ee": 1.0, "mumu": 0.0},
    "WRtoNMutoMuMuJJ": {"ee": 0.0, "mumu": 1.0},
    "WRtoNLtoLLJJ": {"ee": 0.5, "mumu": 0.5},     # e and mu mixed (Run 2 private grid)
}

COVERAGE = math.erf(1 / math.sqrt(2))     # 68.3 %: the fraction inside sigma_eff
# CMS (Petroff) 6-colour palette; no black, which CMS keeps for data
COLORS = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
MARKERS = ["o", "s", "^", "v", "D", "P"]


def sr_region(topology: str, channel: str) -> str:
    return f"wr_{channel}_{topology}_sr"


def flavour_shares(dataset: str) -> dict[str, float]:
    """Share of a signal dataset's cross section in each channel ({"ee": ..., "mumu": ...})."""
    name = dataset.lstrip("/")
    for process, shares in FLAVOUR_SHARES.items():
        if name.startswith(process):
            return shares
    raise ValueError(f"Unknown signal process in dataset {dataset!r}; known: {', '.join(FLAVOUR_SHARES)}")


# ---------------------------------------------------------------------------
# Numbers with statistical errors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Value:
    value: float
    error: float

    def __truediv__(self, other: "Value") -> "Value":
        """Ratio of two independent values."""
        if not other.value or not math.isfinite(other.value) or not math.isfinite(self.value):
            return Value(math.nan, math.nan)
        ratio = self.value / other.value
        return Value(ratio, math.hypot(self.error / other.value, ratio * other.error / other.value))


NAN = Value(math.nan, math.nan)


# ---------------------------------------------------------------------------
# Era sets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EraSet:
    era: str                      # era or combination of data/lumi.yaml
    dir_name: str
    inputs: Inputs                # signal files, one directory per era of the set
    eras: tuple[str, ...]         # the set's single eras, aligned with inputs.dirs
    lumis: tuple[float, ...]      # fb^-1, aligned with eras
    com: float                    # TeV
    label: str                    # legend label


def make_era_sets(specs: Sequence[tuple[str, str]], working_dir: str | Path | None = None) -> list[EraSet]:
    """One :class:`EraSet` per (era, directory); sets of the same era are labelled with their directory.

    Raises FileNotFoundError for a missing directory, KeyError for an unknown era.
    """
    eras = [era for era, _ in specs]
    out = []
    for era, dir_name in specs:
        info = load_lumi(era)
        sub_eras = tuple(info.get("sub_eras") or [era])
        lumis = tuple(float(x) for x in (info.get("sub_lumis") or [info["lumi"]]))
        (inputs,) = make_inputs(era, dir_name, with_dy=False, working_dir=working_dir)
        com = float(info.get("com", 13.6))
        label = f"{era_label(era)}, {com:g} TeV"
        if eras.count(era) > 1:
            label += f" ({dir_name})"
        out.append(EraSet(era, dir_name, inputs, sub_eras, lumis, com, label))
    return out


def point_files(era_set: EraSet, point: str) -> list[Path] | None:
    """The point's file in each era of the set, or None when an era lacks it."""
    files = [root_file(d, f"signal_{point}") for d in era_set.inputs.dirs]
    return files if all(f.is_file() for f in files) else None


def set_points(era_set: EraSet) -> list[str]:
    """Mass points with a file in every era of the set, by WR then N mass."""
    points = era_set.inputs.signal_points()
    partial = [p for p in points if point_files(era_set, p) is None]
    if partial:
        logger.warning("%s: %d point(s) missing in some of its eras are left out: %s", era_set.label,
                       len(partial), ", ".join(partial))
    return [p for p in points if p not in partial]


def common_points(sets: Sequence[EraSet]) -> list[str]:
    """Mass points every set has, by WR then N mass."""
    common = set.intersection(*(set(set_points(s)) for s in sets)) if sets else set()
    return sorted(common, key=mass_point)


# ---------------------------------------------------------------------------
# Produced signal events: sigma x L
# ---------------------------------------------------------------------------

def signal_config_path(era: str, analyzer_root: str | Path | None = None) -> Path:
    """The analyzer's dataset config of an era's signal (WrCoffea data/configs)."""
    info = load_lumi(era)
    root = Path(analyzer_root) if analyzer_root else repo_root().parent
    return root / "data" / "configs" / str(info["run"]) / str(info["year"]) / era / f"{era}_signal.json"


@lru_cache(maxsize=None)
def _cross_sections(path: str) -> dict[str, float | None]:
    """Cross section [pb] of each dataset of a dataset config or a fileset (its entries'
    ``metadata``), under its key, dataset, sample and DAS names."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No signal dataset config {p} (the WrCoffea analyzer's data/configs).")
    out: dict[str, float | None] = {}
    for key, entry in json.loads(p.read_text()).items():
        if not isinstance(entry, dict):
            continue
        fields = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else entry
        names = {key} | {fields.get(k) for k in ("dataset", "sample", "das_name")}
        for name in names - {None, ""}:
            out[str(name)] = fields.get("xsec")
    return out


def clear_config_cache() -> None:
    _cross_sections.cache_clear()
    _skim_counts.cache_clear()


@dataclass(frozen=True)
class Produced:
    """Signal events produced (sigma x L) of one mass point in one era set."""
    channels: dict[str, float]    # in each channel's lepton flavour
    total: float                  # both flavours
    datasets: tuple[str, ...]


def produced_events(era_set: EraSet, point: str, *, analyzer_root: str | Path | None = None,
                    filesets: Mapping[str, Path] | None = None) -> Produced:
    """sigma x L of the datasets in the point's files, per channel and in total.

    The cross sections come from the era's dataset config, or from
    ``filesets[era]``, the fileset the analyzer ran with, when given.
    Raises ValueError when a file has no manifest listing its datasets, a
    dataset has no cross section, or its process is not a known signal.
    """
    files = point_files(era_set, point)
    if files is None:
        raise ValueError(f"{era_set.label}: {point} is missing in some of its eras.")
    channels = dict.fromkeys(CHANNELS, 0.0)
    total = 0.0
    names: list[str] = []
    for era, lumi, path in zip(era_set.eras, era_set.lumis, files):
        datasets = (read_manifest(path) or {}).get("datasets")
        if not datasets:
            raise ValueError(f"{path} has no analyzer manifest listing its datasets "
                             f"({path.stem}_systematics.json), so its cross section is unknown.")
        config = (filesets or {}).get(era) or signal_config_path(era, analyzer_root)
        xsecs = _cross_sections(str(config))
        for dataset in datasets:
            xsec = xsecs.get(dataset)
            if xsec is None:
                raise ValueError(f"{dataset} ({path}) has no cross section in {config}.")
            events = float(xsec) * lumi * 1000.0          # pb x fb^-1
            total += events
            for channel, share in flavour_shares(dataset).items():
                channels[channel] += share * events
            names.append(dataset)
    return Produced(channels, total, tuple(names))


# ---------------------------------------------------------------------------
# Histograms and yields
# ---------------------------------------------------------------------------

def load_point_hist(era_set: EraSet, point: str, key: str) -> Hist | None:
    """Histogram *key* of a mass point summed over the set's eras (flow bins kept); None if absent."""
    files = point_files(era_set, point)
    if files is None:
        return None
    total = None
    for path in files:
        try:
            h = _open_root(str(path))[key].to_hist()
        except (FileNotFoundError, KeyError, OSError):
            return None
        total = h if total is None else total + h
    return total


def region_variables(era_set: EraSet, point: str, region: str) -> list[str]:
    """Names of the 1D histograms of a region in the point's file of the set's first era."""
    files = point_files(era_set, point)
    if not files:
        return []
    f = _open_root(str(files[0]))
    suffix = f"_{region}"
    return sorted(key.split("/", 1)[1].removesuffix(suffix)
                  for key, cls in f.classnames(cycle=False).items()
                  if key.startswith(f"{region}/") and key.endswith(suffix) and cls.startswith("TH1"))


def flow_arrays(h: Hist) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(edges, contents, variances) of a 1D histogram; contents[0] is the underflow, contents[-1]
    the overflow (zero when the axis has none)."""
    axis = h.axes[0]
    values = np.asarray(h.values(flow=True), dtype=float)
    variances = h.variances(flow=True)
    variances = values.copy() if variances is None else np.asarray(variances, dtype=float)
    pad = (0 if axis.traits.underflow else 1, 0 if axis.traits.overflow else 1)
    values = np.pad(values, pad)
    variances = np.pad(variances, pad)
    return np.asarray(axis.edges, dtype=float), values, variances


def region_yield(era_set: EraSet, point: str, topology: str, channel: str) -> Value | None:
    """Expected yield of a signal region (integral of its mass histogram, flow bins included)."""
    h = load_point_hist(era_set, point, hist_key(sr_region(topology, channel), MASS_VARIABLES[topology]))
    if h is None:
        return None
    _, values, variances = flow_arrays(h)
    return Value(float(values.sum()), math.sqrt(max(float(variances.sum()), 0.0)))


def efficiency(era_set: EraSet, point: str, topology: str, channel: str, produced: Produced) -> Value | None:
    """Acceptance x efficiency of a channel in one topology or both (COMBINED); None without the
    histograms or without signal of the channel's flavour."""
    produced_n = produced.channels[channel]
    if produced_n <= 0:
        return None
    yields = [region_yield(era_set, point, t, channel)
              for t in (TOPOLOGIES if topology == COMBINED else (topology,))]
    if any(y is None for y in yields):
        return None
    value = sum(y.value for y in yields)
    error = math.sqrt(sum(y.error ** 2 for y in yields))
    return Value(value / produced_n, error / produced_n)


@dataclass(frozen=True)
class Factors:
    """A channel's efficiency in one topology as the product no_cuts x selection x scale factors."""
    no_cuts: float            # cutflow row 0 (skim x event filters)
    selection: float          # last cutflow row (the signal region, base weight) / row 0
    scale_factors: float      # signal-region yield (all weights) / last cutflow row: mean lepton-SF weight


def efficiency_factors(era_set: EraSet, point: str, topology: str, channel: str,
                       produced: Produced) -> Factors | None:
    """A channel's efficiency split into no_cuts x selection x scale factors (see cutflow_efficiency)."""
    cf = cutflow_efficiency(era_set, point, topology, channel, produced)
    sr = region_yield(era_set, point, topology, channel)
    if cf is None or sr is None or cf.values[0] <= 0 or cf.values[-1] <= 0:
        return None
    last = float(cf.values[-1])
    return Factors(float(cf.values[0]), last / float(cf.values[0]),
                   sr.value / produced.channels[channel] / last)


def weight_factor_means(era_set: EraSet, point: str, topology: str, channel: str) -> dict[str, float]:
    """Mean of each per-event weight factor (lepton scale factors, pileup, and ``corrections``, their
    product) over a channel's signal-region events, from the analyzer's
    event_weights/<region>/components histograms (each event counts once).

    The means use the bin centres of those histograms (0.005 wide), so they
    are good to about 0.003.  A factor that is 1 for every event (all entries
    in the bin starting at 1, e.g. a DY weight on signal) is the identity and
    is left out.  Over the set's eras the means are averaged with the region
    yield of each era as weight.  Factors missing in some era are left out;
    {} for files without these histograms (made before the analyzer wrote them).
    """
    files = point_files(era_set, point)
    if not files:
        return {}
    region = sr_region(topology, channel)
    per_file = []
    for path in files:
        f = _open_root(str(path))
        base = f"event_weights/{region}/components"
        try:
            names = [k for k, cls in f[base].classnames(recursive=False, cycle=False).items()
                     if cls.startswith("TH1") and k != "normalization" and not k.startswith("genweight")]
            mass = f[hist_key(region, MASS_VARIABLES[topology])]
        except KeyError:
            return {}
        means, identity = {}, set()
        for name in names:
            h = f[f"{base}/{name}"]
            counts = np.asarray(h.values(), dtype=float)
            if counts.sum() <= 0:
                continue
            edges = h.axis().edges()
            filled = np.flatnonzero(counts)
            if len(filled) == 1 and np.isclose(edges[filled[0]], 1.0):
                identity.add(name)
                continue
            means[name] = float((h.axis().centers() * counts).sum() / counts.sum())
        per_file.append((float(np.sum(mass.values(flow=True))), means, identity))
    identity = set.intersection(*(i for _, _, i in per_file))
    names = set.intersection(*(set(m) | i for _, m, i in per_file)) - identity
    if any(n not in m for _, m, _ in per_file for n in names):     # the identity in some eras only
        per_file = [(y, {**dict.fromkeys(i, 1.0), **m}, i) for y, m, i in per_file]
    per_file = [(y, m) for y, m, _ in per_file]
    total = sum(y for y, _ in per_file)
    if total <= 0:
        return {}
    return {name: sum(y * m[name] for y, m in per_file) / total for name in sorted(names)}


def signal_fileset_path(era: str, analyzer_root: str | Path | None = None, *, skimmed: bool = True) -> Path:
    """The analyzer's signal fileset of an era (WrCoffea data/filesets)."""
    info = load_lumi(era)
    root = Path(analyzer_root) if analyzer_root else repo_root().parent
    return (root / "data" / "filesets" / str(info["run"]) / str(info["year"]) / era
            / ("skimmed" if skimmed else "unskimmed") / f"{era}_signal_fileset.json")


@lru_cache(maxsize=None)
def _skim_counts(path: str) -> dict[str, tuple[int, int]]:
    """(events in the skim, generated events) of each dataset of a skimmed fileset."""
    p = Path(path)
    if not p.is_file():
        return {}
    out = {}
    for key, entry in json.loads(p.read_text()).items():
        md = (entry or {}).get("metadata") or {}
        if md.get("nevts") and md.get("genEventCount"):
            out[str(md.get("sample", key))] = (int(md["nevts"]), int(md["genEventCount"]))
    return out


def no_cuts_breakdown(era_set: EraSet, point: str, produced: Produced, *,
                      analyzer_root: str | Path | None = None,
                      filesets: Mapping[str, Path] | None = None) -> tuple[float, float] | None:
    """(skim, filters): the share of generated events in the skim (fileset nevts / genEventCount),
    and the share of those the analyzer's event filters (MET filters, jet veto map) keep
    (unweighted no_cuts / nevts), over the set's eras.  None without the skimmed fileset's counts.

    Read from ``filesets[era]`` when given, else from the era's installed skimmed signal fileset,
    which must still describe the skims the files were made from.
    """
    files = point_files(era_set, point)
    if not files:
        return None
    in_skim = generated = 0
    for era in era_set.eras:
        path = (filesets or {}).get(era) or signal_fileset_path(era, analyzer_root)
        counts = _skim_counts(str(path))
        for dataset in set(produced.datasets):
            if dataset not in counts:
                return None
            in_skim += counts[dataset][0]
            generated += counts[dataset][1]
    no_cuts = read_files(files, "resolved", "ee", "cumulative", unweighted=True)
    if no_cuts is None or not in_skim or not generated:
        return None
    return in_skim / generated, float(no_cuts.values[0]) / in_skim


@dataclass(frozen=True)
class CutflowEfficiency:
    cuts: tuple[str, ...]
    values: np.ndarray            # efficiency after each cut (see cutflow_efficiency)
    errors: np.ndarray
    n_shared: int                 # leading rows every channel shares (divided by sigma x L of all flavours)


def cutflow_efficiency(era_set: EraSet, point: str, topology: str, channel: str,
                       produced: Produced) -> CutflowEfficiency | None:
    """Cumulative cutflow of a channel (base weight) as efficiencies of the channel's flavour.

    The leading rows that the ee and mumu cutflows share (same cut, same
    value: no_cuts and the flavour-blind jet cuts) count events of every
    flavour and are divided by sigma x L of all flavours; the rows from the
    first lepton-flavour cut on are divided by sigma x L of the channel's
    flavour.  The last row is then the channel's efficiency before lepton
    scale factors, whatever mix of flavours the files hold.  The shared rows
    average over the flavours in the files (the skim keeps fewer electron than
    muon events), so between sets with different mixes (a muon-only file
    against one with both) the split of the efficiency between those rows and
    the rest is approximate; the last row is not.  None without the cutflow or
    without signal of the channel's flavour.
    """
    files = point_files(era_set, point)
    cf = read_files(files, topology, channel, "cumulative") if files else None
    if cf is None or produced.total <= 0 or produced.channels[channel] <= 0:
        return None
    other = read_files(files, topology, "mumu" if channel == "ee" else "ee", "cumulative")
    n_shared = 0
    if other is not None:
        for cut, value, other_cut, other_value in zip(cf.cuts, cf.values, other.cuts, other.values):
            if cut != other_cut or not math.isclose(value, other_value, rel_tol=1e-9, abs_tol=1e-12):
                break
            n_shared += 1
    denominators = np.where(np.arange(len(cf.cuts)) < n_shared, produced.total, produced.channels[channel])
    return CutflowEfficiency(cf.cuts, cf.values / denominators,
                             np.sqrt(np.clip(cf.variances, 0, None)) / denominators, n_shared)


# ---------------------------------------------------------------------------
# Shape of the mass peak
# ---------------------------------------------------------------------------

def histogram_quantile(edges: np.ndarray, contents: np.ndarray, p: float) -> float:
    """The p-quantile of a histogram with flow bins (contents[0] underflow, contents[-1]
    overflow), linear inside each bin; NaN when it falls in a flow bin or the histogram is empty.

    Negative bins count as empty.
    """
    c = np.clip(np.asarray(contents, dtype=float), 0, None)
    total = c.sum()
    if total <= 0:
        return math.nan
    cdf = np.cumsum(c)
    k = int(np.searchsorted(cdf, p * total, side="left"))
    if k == 0 or k >= len(c) - 1:
        return math.nan
    frac = (p * total - cdf[k - 1]) / c[k] if c[k] > 0 else 0.0
    return float(edges[k - 1] + frac * (edges[k] - edges[k - 1]))


def sigma_eff(edges: np.ndarray, contents: np.ndarray, fraction: float = COVERAGE) -> float:
    """Half the width of the narrowest interval holding *fraction* of all events (flow bins
    count in the total); NaN when every such interval reaches into a flow bin.

    The cumulative distribution is linear inside each bin, so the narrowest
    interval starts or ends on a bin edge: both families are scanned.
    """
    c = np.clip(np.asarray(contents, dtype=float), 0, None)
    total = c.sum()
    if total <= 0:
        return math.nan
    # CDF at each edge; a tiny ramp makes it strictly increasing for the inversion
    cdf = c[0] + np.concatenate([[0.0], np.cumsum(c[1:-1])])
    cdf = cdf + np.arange(len(cdf)) * total * 1e-12
    need = fraction * total
    widths = []
    up = cdf + need                     # intervals starting on an edge
    ok = up <= cdf[-1]
    if ok.any():
        widths.append(np.interp(up[ok], cdf, edges) - edges[ok])
    down = cdf - need                   # intervals ending on an edge
    ok = down >= cdf[0]
    if ok.any():
        widths.append(edges[ok] - np.interp(down[ok], cdf, edges))
    if not widths:
        return math.nan
    return float(0.5 * np.concatenate(widths).min())


@dataclass(frozen=True)
class Shape:
    n_eff: float
    median: Value
    q16: float
    q84: float
    sigma_eff: Value


def shape_summary(h: Hist) -> Shape:
    """Median, 16/84 % quantiles and sigma_eff of a histogram (see the module docstring)."""
    edges, values, variances = flow_arrays(h)
    sum_w2 = float(np.clip(variances, 0, None).sum())
    n_eff = float(values.sum()) ** 2 / sum_w2 if sum_w2 > 0 else 0.0
    width = sigma_eff(edges, values)
    root_n = math.sqrt(n_eff) if n_eff > 0 else math.nan
    return Shape(
        n_eff=n_eff,
        median=Value(histogram_quantile(edges, values, 0.5), 1.2533 * width / root_n),
        q16=histogram_quantile(edges, values, 0.5 - COVERAGE / 2),
        q84=histogram_quantile(edges, values, 0.5 + COVERAGE / 2),
        sigma_eff=Value(width, width / math.sqrt(2) / root_n),
    )


def interpolate_in_mn(points: Sequence[tuple[int, Value]], m_n: int) -> Value | None:
    """A quantity known at the m_N values of *points* (one m_WR), at *m_n*.

    The value itself at a grid point; else a monotone cubic (PCHIP) interpolation in m_N, with the
    error interpolated linearly.  None in the grid's first and last interval, where the efficiency
    curves are too steep to interpolate (light N, and N near m_WR), and for fewer than four points.
    For comparing samples made on different m_N grids (UL18 official vs the pre-legacy scan).
    """
    pts = sorted(points, key=lambda x: x[0])
    xs = [m for m, _ in pts]
    if m_n in xs:
        return pts[xs.index(m_n)][1]
    if len(pts) < 4 or not xs[1] < m_n < xs[-2]:
        return None
    from scipy.interpolate import PchipInterpolator
    value = float(PchipInterpolator(xs, [v.value for _, v in pts])(m_n))
    return Value(value, float(np.interp(m_n, xs, [v.error for _, v in pts])))


def display_range(hists: Sequence[Hist], *, target_bins: int = 40,
                  coverage: tuple[float, float] = (0.002, 0.998)) -> tuple[int, tuple[float, float]]:
    """(rebin factor, x range) that shows the bulk of every histogram in about *target_bins* bins.

    The range runs from the lowest *coverage[0]* to the highest *coverage[1]*
    quantile of the histograms (the axis ends where a quantile is in a flow
    bin), snapped outward to the rebinned edges.  Non-uniform axes are not rebinned.
    """
    edges = np.asarray(hists[0].axes[0].edges, dtype=float)
    lows, highs = [], []
    for h in hists:
        e, values, _ = flow_arrays(h)
        lo, hi = (histogram_quantile(e, values, p) for p in coverage)
        lows.append(e[0] if math.isnan(lo) else lo)
        highs.append(e[-1] if math.isnan(hi) else hi)
    lo, hi = min(lows), max(highs)
    widths = np.diff(edges)
    if not np.allclose(widths, widths[0]):
        return 1, (float(max(lo, edges[0])), float(min(hi, edges[-1])))
    width = float(widths[0])
    factor = max(1, int(round((hi - lo) / width / target_bins)))
    step = factor * width
    lo = edges[0] + math.floor((lo - edges[0]) / step + 1e-9) * step
    hi = edges[0] + math.ceil((hi - edges[0]) / step - 1e-9) * step
    if hi <= lo:
        hi = lo + step
    return factor, (float(max(lo, edges[0])), float(min(hi, edges[-1])))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def style(index: int) -> dict:
    return {"color": COLORS[index % len(COLORS)], "marker": MARKERS[index % len(MARKERS)]}


def _header(ax, sets: Sequence[EraSet], text: str, fontsize: int = FONT_SIZE_LABEL,
            text_fontsize: int = FONT_SIZE_TITLE) -> None:
    """CMS simulation label with *text* under it; the energies of the sets on the right."""
    energies = " / ".join(dict.fromkeys(f"{s.com:g}" for s in sets))
    cms_label(ax, None, data=False, rlabel=f"({energies} TeV)", text=text, fontsize=fontsize,
              text_fontsize=text_fontsize)


def _ratio_panel(rax, ref_label: str, ylim: tuple[float, float]) -> None:
    rax.axhline(1.0, ls="--", color="k", lw=1)
    rax.set_ylim(*ylim)
    rax.set_ylabel("Ratio to\n" + ref_label, fontsize=FONT_SIZE_MIN)


def _figure():
    hep.style.use("CMS")
    return plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), sharex=True, figsize=(10, 10))


def overflow_fraction(h: Hist, xmax: float) -> float:
    """Share of a histogram's events at or above *xmax*, overflow included."""
    edges, values, _ = flow_arrays(h)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    beyond = float(values[1:-1][edges[:-1] >= xmax - 1e-9].sum() + values[-1])
    return beyond / total


def _normalized(h: Hist, factor: int, xlim: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(edges, fractions, errors) of *h* rebinned for display; the overflow sits in the last bin and
    the fractions are of all events, flow bins included."""
    _, values, _ = flow_arrays(h)
    total = float(values.sum())
    shown = rebin_histogram(h, factor, density=False, xmax=xlim[1])
    scale = 1.0 / total if total > 0 else 0.0
    return (np.asarray(shown.axes[0].edges), shown.values() * scale,
            np.sqrt(np.clip(shown.variances(), 0, None)) * scale)


def plot_shape_overlay(entries: Sequence[tuple[EraSet, Hist]], *, xlabel: str, text: str,
                       ratio_ylim: tuple[float, float] = (0.5, 1.5), target_bins: int = 40) -> Figure:
    """Each set's histogram scaled to unit area, with the ratio to the first set.

    Events beyond the drawn range sit in the last bin; the label says so when
    they are more than 1 % of a set.
    """
    factor, xlim = display_range([h for _, h in entries], target_bins=target_bins)
    if max(overflow_fraction(h, xlim[1]) for _, h in entries) > 0.01:
        text += "\nLast bin: overflow"
    fig, (ax, rax) = _figure()
    drawn = []
    for i, (era_set, h) in enumerate(entries):
        edges, vals, errs = _normalized(h, factor, xlim)
        drawn.append((edges, vals, errs))
        hep.histplot(vals, edges, yerr=errs, histtype="step", color=style(i)["color"], linewidth=2,
                     label=era_set.label, ax=ax)
    edges, ref, ref_err = drawn[0]
    rel = np.divide(ref_err, ref, out=np.zeros_like(ref), where=ref > 0)
    rax.stairs(1 + rel, edges, baseline=1 - rel, fill=True, color="grey", alpha=0.3, lw=0)
    for i, (edges, vals, errs) in enumerate(drawn[1:], 1):
        ratio = np.divide(vals, ref, out=np.full_like(vals, np.nan), where=ref > 0)
        rerr = np.divide(errs, ref, out=np.full_like(vals, np.nan), where=ref > 0)
        hep.histplot(ratio, edges, yerr=rerr, histtype="errorbar", color=style(i)["color"], ax=rax,
                     xerr=True, marker=style(i)["marker"], markersize=5)
    top = max(float(np.nanmax(v[(e[:-1] < xlim[1]) & (e[1:] > xlim[0])], initial=0)) for e, v, _ in drawn)
    ax.set_ylim(0, 1.45 * top if top > 0 else 1)
    ax.set_xlim(*xlim)
    ax.set_ylabel("Fraction of events / bin")
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    _header(ax, [s for s, _ in entries], text)
    fit_header(ax)
    _ratio_panel(rax, "ref.", ratio_ylim)
    rax.set_xlabel(xlabel)
    return fig


def plot_vs_mn(series: Sequence[tuple[EraSet, list[tuple[int, Value]]]], *, ylabel: str, text: str,
               percent: bool = False, ratio_ylim: tuple[float, float] = (0.7, 1.3),
               ylim: tuple[float, float] | None = None) -> Figure:
    """A quantity vs m_N for each set (points with errors, joined), with the ratio to the first set:
    filled at the m_N values both have, open where the first set is interpolated in m_N
    (:func:`interpolate_in_mn`, sets on another m_N grid)."""
    fig, (ax, rax) = _figure()
    scale = 100.0 if percent else 1.0
    ref = dict(series[0][1])
    shift = 0.0
    interpolated = False
    for i, (era_set, points) in enumerate(series):
        if not points:
            continue
        mn = np.array([m for m, _ in points], dtype=float)
        val = np.array([v.value for _, v in points]) * scale
        err = np.array([v.error for _, v in points]) * scale
        st = style(i)
        ax.errorbar(mn, val, yerr=err, color=st["color"], marker=st["marker"], markersize=6, lw=1.5,
                    capsize=2, label=era_set.label)
        if i:
            spread = (max(ref) - min(ref)) if len(ref) > 1 else 1.0
            shift = 0.004 * spread * (i - (len(series) - 1) / 2)
            pairs = [(m, v / ref[m]) for m, v in points if m in ref]
            if pairs:
                rax.errorbar(np.array([m for m, _ in pairs]) + shift, [r.value for _, r in pairs],
                             yerr=[r.error for _, r in pairs], color=st["color"], marker=st["marker"],
                             markersize=5, lw=0, elinewidth=1.5, capsize=2)
            between = [(m, v / r) for m, v in points if m not in ref
                       and (r := interpolate_in_mn(series[0][1], m)) is not None]
            if between:
                interpolated = True
                rax.errorbar(np.array([m for m, _ in between]) + shift, [r.value for _, r in between],
                             yerr=[r.error for _, r in between], color=st["color"], marker=st["marker"],
                             mfc="white", markersize=5, lw=0, elinewidth=1.0, capsize=2)
    if interpolated:
        rax.text(0.98, 0.06, "open: reference interpolated in $m_N$", transform=rax.transAxes, ha="right",
                 fontsize=FONT_SIZE_MIN)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 0) if lo >= 0 else lo, hi + 0.35 * (hi - min(lo, 0)))
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    _header(ax, [s for s, _ in series], text)
    fit_header(ax)
    _ratio_panel(rax, series[0][0].label.split(",")[0], ratio_ylim)
    rax.set_xlabel(r"$m_N$ [GeV]")
    return fig


def plot_cutflow_efficiency(series: Sequence[tuple[EraSet, CutflowEfficiency]], *, text: str,
                            ratio_ylim: tuple[float, float] = (0.8, 1.2)) -> Figure:
    """Cumulative efficiency of each cut for each set, with the ratio to the first set.

    Cuts are matched by name; a cut a set lacks is left out of its line.
    """
    cuts = list(series[0][1].cuts)
    for _, cf in series[1:]:
        cuts += [c for c in cf.cuts if c not in cuts]
    x = np.arange(len(cuts))
    fig, (ax, rax) = _figure()
    fig.set_size_inches(14, 10)
    ref = dict(zip(series[0][1].cuts, zip(series[0][1].values, series[0][1].errors)))
    for i, (era_set, cf) in enumerate(series):
        st = style(i)
        pos = np.array([cuts.index(c) for c in cf.cuts])
        ax.errorbar(pos, cf.values * 100, yerr=cf.errors * 100, color=st["color"], marker=st["marker"],
                    markersize=6, lw=1.5, capsize=2, label=era_set.label)
        if i:
            pairs = [(cuts.index(c), Value(v, e) / Value(*ref[c])) for c, v, e in
                     zip(cf.cuts, cf.values, cf.errors) if c in ref]
            rax.errorbar([p + 0.08 * (i - (len(series) - 1) / 2) for p, _ in pairs],
                         [r.value for _, r in pairs], yerr=[r.error for _, r in pairs],
                         color=st["color"], marker=st["marker"], markersize=5, lw=0, elinewidth=1.5,
                         capsize=2)
    shown = np.concatenate([cf.values for _, cf in series]) * 100
    shown = shown[np.isfinite(shown) & (shown > 0)]
    top = float(shown.max()) if shown.size else 1.0
    if shown.size and top / float(shown.min()) > 20:     # boosted chains fall by orders of magnitude
        ax.set_yscale("log")
        ax.set_ylim(float(shown.min()) / 3, top * 30)
    else:
        ax.set_ylim(0, 1.5 * top)
    ax.set_ylabel("Cumulative efficiency [%]")
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    _header(ax, [s for s, _ in series], text + "\nBase weight (no lepton SFs)")
    fit_header(ax)
    _ratio_panel(rax, series[0][0].label.split(",")[0], ratio_ylim)
    rax.set_xticks(x)
    rax.set_xticklabels([cut_label(c, "plot") for c in cuts], rotation=45, ha="right", rotation_mode="anchor",
                        fontsize=FONT_SIZE_MIN)
    rax.set_xlim(-0.5, len(cuts) - 0.5)
    return fig


def plot_resolution(series: Sequence[tuple[EraSet, list[tuple[int, Shape]]]], *, m_wr: int,
                    text: str, ratio_ylim: tuple[float, float] = (0.8, 1.2)) -> Figure:
    """Peak position (median / m_WR) and relative width (sigma_eff / median) vs m_N, with the ratio
    of the widths to the first set."""
    hep.style.use("CMS")
    fig, (ax_pos, ax_wid, rax) = plt.subplots(3, 1, gridspec_kw=dict(height_ratios=[2, 2, 1], hspace=0.1),
                                              sharex=True, figsize=(10, 12))
    ref = {m: s.sigma_eff / s.median for m, s in series[0][1]}
    for i, (era_set, points) in enumerate(series):
        if not points:
            continue
        st = style(i)
        masses = [m for m, _ in points]
        pos = [s.median / Value(m_wr, 0.0) for _, s in points]
        wid = [s.sigma_eff / s.median for _, s in points]
        kw = dict(color=st["color"], marker=st["marker"], markersize=6, lw=1.5, capsize=2)
        ax_pos.errorbar(masses, [v.value for v in pos], yerr=[v.error for v in pos], label=era_set.label, **kw)
        ax_wid.errorbar(masses, [v.value * 100 for v in wid], yerr=[v.error * 100 for v in wid], **kw)
        if i:
            pairs = [(m, w / ref[m]) for m, w in zip(masses, wid) if m in ref]
            rax.errorbar([m for m, _ in pairs], [r.value for _, r in pairs], yerr=[r.error for _, r in pairs],
                         color=st["color"], marker=st["marker"], markersize=5, lw=0, elinewidth=1.5, capsize=2)
    lo, hi = ax_pos.get_ylim()
    ax_pos.set_ylim(lo, hi + 0.6 * (hi - lo))
    ax_pos.set_ylabel(r"Median / $m_{W_R}$", fontsize=22)
    ax_wid.set_yscale("log")            # light N: tens of %, the peak region: a few %
    ax_wid.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax_wid.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax_wid.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_wid.set_ylabel(r"$\sigma_{\mathrm{eff}}$ / median [%]", fontsize=22)
    ax_pos.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    _header(ax_pos, [s for s, _ in series], text)
    fit_header(ax_pos)
    _ratio_panel(rax, "ref. width", ratio_ylim)
    rax.set_xlabel(r"$m_N$ [GeV]")
    return fig
