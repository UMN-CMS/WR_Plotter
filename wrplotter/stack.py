"""Assemble the histograms of one stack plot (one region, one variable).

Signal-region data is read only from data files the analyzer made unblinded
(see :func:`wrplotter.inputs.sr_data_unblinded`); otherwise it is never loaded
and the plot is marked blinded.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from hist import Hist

from .histogram_cache import load_file, load_per_file, sum_files
from .histogram_utils import add_hists, rebin_histogram
from .inputs import Inputs, sr_data_unblinded
from .regions import Region
from .sample_groups import SampleGroup
from .systematics import sample_variations, sources_in_files, total_variations

logger = logging.getLogger(__name__)

# (era, dir, sample, what) already reported: a sample missing from the stack or lacking variations
_REPORTED: set[tuple[str, str, str, str]] = set()


@dataclass(frozen=True)
class Binning:
    rebin: int | list[float] = 1
    xmax: float | None = None
    density: bool = False   # divide variable-width bins by their width (Run 2 drew events per bin)

    @property
    def per_gev(self) -> bool:
        """Bin contents are divided by the bin width (variable-width rebinning)."""
        return self.density and not isinstance(self.rebin, int)


@dataclass(frozen=True)
class AxisConfig:
    binning: Binning
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    ratio_ylim: tuple[float, float]


def axis_config(vcfg: dict, *, variable_rebin: bool = True, density: bool = False) -> AxisConfig:
    """Binning and axis ranges of one (region, variable) plot-settings entry.

    The variable-width edges (``rebin_variable``, the Run 2 m(W_R) fit binning) are used
    where given and *variable_rebin* is True; otherwise the uniform ``rebin``.  The
    variable edges keep their own last edge as the axis end, so ``xlim`` (set for the
    uniform bins) never cuts the Run 2 bins short.
    """
    xlim = tuple(map(float, vcfg.get("xlim", (0.0, 1.0))))
    if variable_rebin and "rebin_variable" in vcfg:
        rebin = list(vcfg["rebin_variable"])
        xlim = (xlim[0], float(rebin[-1]))
    else:
        rebin = vcfg.get("rebin", 1)
    return AxisConfig(
        binning=Binning(rebin=rebin, xmax=xlim[1], density=density),
        xlim=xlim,
        ylim=tuple(map(float, vcfg.get("ylim", (1.0, 1e6)))),
        ratio_ylim=tuple(map(float, vcfg.get("ratio_ylim", (0.5, 2.0)))),
    )


def hist_yield(h: Hist, binning: Binning) -> float:
    """Number of (weighted) events in *h*, undoing the bin-width division if any."""
    values = h.values()
    if binning.per_gev:
        values = values * h.axes[0].widths
    return float(values.sum())


class HistLoader:
    """Load one histogram key of one sample: summed over sub-eras, rebinned."""

    def __init__(self, inputs: Inputs, binning: Binning):
        self.inputs = inputs
        self.binning = binning

    def __call__(self, sample: str, hist_key: str) -> Hist | None:
        # Read the files Inputs resolved, not WRAnalyzer_<sample>.root rebuilt in their folders: since
        # 2026-10-12 every DY variant is WRAnalyzer_DYJets_<tag>.root next to the default WRAnalyzer_DYJets.root,
        # and the rebuilt name made every variant read the default sample.
        return sum_files(self.per_file(sample, hist_key), hist_key)

    def per_file(self, sample: str, hist_key: str) -> dict[Path, Hist | None]:
        """``{file: hist_key of that file, rebinned, or None}``, one file per sub-era of *sample*."""
        return load_per_file(self.inputs.files_for(sample), hist_key, self.binning.rebin,
                             density=self.binning.density, xmax=self.binning.xmax)

    def file(self, path: Path, hist_key: str) -> Hist | None:
        """*hist_key* of one file, rebinned like every other histogram of the plot."""
        return load_file(path, hist_key, self.binning.rebin,
                         density=self.binning.density, xmax=self.binning.xmax)

    def rebin(self, h: Hist) -> Hist:
        """*h* (raw, analysis binning) rebinned like every other histogram of the plot."""
        return rebin_histogram(h, self.binning.rebin, density=self.binning.density,
                               xmax=self.binning.xmax)


@dataclass
class StackHists:
    mc: list[tuple[SampleGroup, Hist]] = field(default_factory=list)  # bottom of the stack first
    data: Hist | None = None
    data_blinded: bool = False       # signal-region data withheld
    signals: dict[str, Hist] = field(default_factory=dict)
    syst: dict[str, dict[str, Hist]] = field(default_factory=dict)

    @property
    def total(self) -> Hist | None:
        total = None
        for _, h in self.mc:
            total = add_hists(total, h)
        return total

    @property
    def empty(self) -> bool:
        return not self.mc and self.data is None


def order_for_stack(mc: list[tuple[SampleGroup, Hist]], region: Region
                    ) -> list[tuple[SampleGroup, Hist]]:
    """Move groups with ``stack_position: top`` to the visual top of the stack.

    In flavor CRs, where those groups (DY) are a small fraction, they go to the
    visual bottom instead.
    """
    top = [item for item in mc if item[0].stack_position == "top"]
    rest = [item for item in mc if item[0].stack_position != "top"]
    return top + rest if "flavor_cr" in region.name else rest + top


def hist_key(region: str, variable: str) -> str:
    return f"{region}/{variable}_{region}"


def _report_once(inputs: Inputs, sample: str, what: str, message: str, *args) -> None:
    key = (inputs.era, inputs.dir_name, sample, what)
    if key not in _REPORTED:
        _REPORTED.add(key)
        logger.warning(message, *args)


def _report_samples_without_variations(inputs: Inputs, per_sample_syst: list) -> None:
    """Report MC samples with no variation in any file while other samples of the stack have some."""
    if not any(variations for _, _, variations in per_sample_syst):
        return
    for sample, _, variations in per_sample_syst:
        files = inputs.files_for(sample)
        if variations or sources_in_files(files):
            continue
        _report_once(inputs, sample, "no_variations",
                     "%s has no systematic variations (%s) while other samples of the stack have "
                     "them: its nominal stands in, so the band leaves out its systematic "
                     "uncertainties.", sample, ", ".join(str(f.parent) for f in files))


def build_stack(
    region: Region,
    variable: str,
    *,
    inputs: Inputs,
    groups: Sequence[SampleGroup],
    binning: Binning,
    signals: Sequence[str] = (),
    systematics: Sequence[str] = (),
) -> StackHists:
    """Histograms for one (region, variable) stack plot.

    *groups* are in YAML order (MC and data); data comes from the group of the
    region's data stream.  *signals* are sample names overlaid as lines.
    Systematic variations are collected for every MC sample that has them,
    era by era (see :mod:`wrplotter.systematics`).  An MC sample with no file,
    or without variations while others have them, is reported once.
    """
    load = HistLoader(inputs, binning)
    key = hist_key(region.name, variable)
    out = StackHists()
    per_sample_syst = []

    for group in groups:
        if group.kind == "data":
            if group.key != region.primary_dataset:
                continue
            for sample in group.samples:
                files = inputs.files_for(sample)
                if not files:
                    continue
                if region.is_signal_region and not all(sr_data_unblinded(f) for f in files):
                    out.data_blinded = True
                    continue
                h = load(sample, key)
                if h is not None:
                    out.data = add_hists(out.data, h)
            continue

        combined = None
        for sample in group.samples:
            per_file = load.per_file(sample, key)
            if not per_file:
                _report_once(inputs, sample, "no_file",
                             "No %s file in %s: %s is missing from the stack.",
                             sample, ", ".join(str(d) for d in inputs.dirs_for(sample)), sample)
                continue
            h = sum_files(per_file, key)
            if h is None:
                continue
            combined = add_hists(combined, h)
            if systematics:
                variations = sample_variations(per_file, region.name, variable, systematics,
                                               load=load.file, rebin=load.rebin)
                per_sample_syst.append((sample, h, variations))
        if combined is not None:
            out.mc.append((group, combined))

    out.mc = order_for_stack(out.mc, region)
    if per_sample_syst:
        _report_samples_without_variations(inputs, per_sample_syst)
        out.syst = total_variations([(h, variations) for _, h, variations in per_sample_syst])

    for sample in signals:
        h = load(sample, key)
        if h is not None:
            out.signals[sample] = h
        else:
            logger.debug("No %s for signal %s", key, sample)
    return out
