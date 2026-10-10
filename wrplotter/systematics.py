"""Systematic up/down variations for the uncertainty band of the stack plots.

The analyzer writes a variation as
``syst_<label>_<region>/<variable>_syst_<label>_<region>``, where <label> is
the source label lower-cased (``jesup``, ``scalemurfdown``, ...), and only for
the (region, label) pairs it produced: a source that cannot affect a region
gets no folder there (the manifest lists it as "unaffected").

A sample has one file per sub-era, and each variation is summed over them,
every file adding either its own variation or its own nominal, never nothing:

- a variation missing in a region of a file that has the source elsewhere
  means the source does not affect that region, and the file's nominal is used;
- a file without some (or all) of the sources that the sample's other files
  have also uses its nominal for them, and is reported: the band then leaves
  out that era's share of those uncertainties;
- a sample with no variation in any of its files adds no band.

Composite sources (``SYST_ENVELOPES``) become one up/down pair from the
per-bin envelope of their member variations.

``pdf`` and ``alphas`` have no variation folders: they are formed from the
analyzer's PDF members at the plot binning (``wrplotter/pdf.py``).
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from hist import Hist

from .histogram_cache import keys_in_file
from .histogram_utils import add_hists
from .pdf import PDF_SOURCES, pdf_variations

logger = logging.getLogger(__name__)

# composite systematic sources: one band from the envelope of several
# correlated analyzer variations (kept in step with signal_fitting/shared/syst_band.ENVELOPES)
SYST_ENVELOPES = {"scale": ("scalemur", "scalemuf", "scalemurf")}

DIRECTIONS = ("up", "down")

_SYST_FOLDER = re.compile(r"^syst_([a-z0-9]+?)(?:up|down)_wr_")

# files already reported as lacking sources that other eras' files of the sample have
_ERA_GAPS_REPORTED: set[str] = set()


@lru_cache(maxsize=256)
def _sources_in_file(path: str) -> frozenset[str]:
    """Sources with a variation folder in the file; empty for a missing file.

    A file that exists but cannot be read raises (it would otherwise turn the
    band into a statistical one without notice).
    """
    if not Path(path).is_file():
        return frozenset()
    folders = {k.split("/", 1)[0] for k in keys_in_file(path)}
    return frozenset(m.group(1) for k in folders if (m := _SYST_FOLDER.match(k)))


def sources_in_files(paths: Iterable[str | Path]) -> frozenset[str]:
    """Source names (``jes``, ``scalemur``, ...) with at least one variation folder in the files."""
    out: set[str] = set()
    for p in paths:
        out |= _sources_in_file(str(p))
    return frozenset(out)


def clear_cache() -> None:
    _sources_in_file.cache_clear()
    _ERA_GAPS_REPORTED.clear()


def variation_key(source: str, direction: str, region: str, variable: str) -> str:
    folder = f"syst_{source}{direction}_{region}"
    return f"{folder}/{variable}_{folder}"


def _with_values(template: Hist, values: np.ndarray) -> Hist:
    """Copy of *template* with its bin values replaced (variances kept)."""
    h = template.copy()
    view = h.view()
    if hasattr(view, "value"):
        view.value[...] = values
    else:
        view[...] = values
    return h


def _report_era_gaps(sources: Mapping[Path, frozenset[str]], wanted: set[str]) -> None:
    """Report, once per file, the wanted sources a file lacks while other files of the sample have them."""
    present = frozenset().union(*sources.values())
    for path, have in sources.items():
        missing = sorted((wanted & present) - have)
        if not missing or str(path) in _ERA_GAPS_REPORTED:
            continue
        _ERA_GAPS_REPORTED.add(str(path))
        if not have:
            logger.warning("%s has no systematic variations while the sample's files of other eras "
                           "have them: its nominal stands in for every source, so the band leaves "
                           "out this era's uncertainties.", path)
        else:
            logger.warning("%s has no %s variations while the sample's files of other eras have "
                           "them: its nominal stands in, so the band leaves out this era's share "
                           "of those sources.", path, ", ".join(missing))


def sample_variations(
    nominals: Mapping[Path, Hist | None],
    region: str,
    variable: str,
    systematics: Sequence[str],
    *,
    load: Callable[[Path, str], Hist | None],
    rebin: Callable[[Hist], Hist] | None = None,
) -> dict[str, dict[str, Hist]]:
    """``{syst: {"up": Hist, "down": Hist}}`` of one sample in one region.

    *nominals* maps each ROOT file of the sample (one per sub-era) to its
    nominal histogram, rebinned (None where the file lacks it); *load(file,
    key)* reads one histogram of one file, rebinned like the nominal.  Each
    variation is summed over the files, a file without it adding its own
    nominal (see the module docstring).  Systematics with no variation in any
    file are left out.  ``pdf`` / ``alphas`` come from the PDF members, each
    member rebinned with *rebin* (the plot's rebinning) before the quadrature.
    """
    files = [f for f, h in nominals.items() if h is not None]
    nominal = None
    for f in files:
        nominal = add_hists(nominal, nominals[f])
    out: dict[str, dict[str, Hist]] = {}
    pdf_wanted = [s for s in systematics if s in PDF_SOURCES]
    if pdf_wanted and rebin is not None and nominal is not None:
        out.update(pdf_variations(files, region, variable, nominal, rebin, pdf_wanted))

    sources = {f: _sources_in_file(str(f)) for f in files}
    present = frozenset().union(*sources.values())
    if not present:
        return out
    _report_era_gaps(sources, {m for s in systematics for m in SYST_ENVELOPES.get(s, (s,))})

    for syst in systematics:
        if syst in PDF_SOURCES:
            continue
        members = [m for m in SYST_ENVELOPES.get(syst, (syst,)) if m in present]
        if not members:
            continue
        pairs = {}
        for member in members:
            for direction in DIRECTIONS:
                key = variation_key(member, direction, region, variable)
                total = None
                for f in files:
                    h = load(f, key)
                    if h is None:
                        logger.debug("%s has no %s: its nominal is used", f, key)
                        h = nominals[f]
                    total = add_hists(total, h)
                pairs[(member, direction)] = total
        if syst in SYST_ENVELOPES:
            stack = np.array([h.values() for h in pairs.values()])
            out[syst] = {"up": _with_values(nominal, stack.max(axis=0)),
                         "down": _with_values(nominal, stack.min(axis=0))}
        else:
            out[syst] = {d: pairs[(syst, d)] for d in DIRECTIONS}
    return out


def total_variations(per_sample: Sequence[tuple[Hist, dict[str, dict[str, Hist]]]]
                     ) -> dict[str, dict[str, Hist]]:
    """Sum per-sample variations into stack totals.

    *per_sample* holds ``(nominal, variations)`` for every MC sample in the
    stack; a sample without a given systematic contributes its nominal.
    """
    systs = sorted({s for _, var in per_sample for s in var})
    out: dict[str, dict[str, Hist]] = {}
    for syst in systs:
        out[syst] = {}
        for direction in DIRECTIONS:
            total = None
            for nominal, var in per_sample:
                h = var.get(syst, {}).get(direction, nominal)
                total = h if total is None else total + h
            out[syst][direction] = total
    return out


def band_error2(nominal_values: np.ndarray, variations: dict[str, dict[str, Hist]] | None
                ) -> tuple[np.ndarray, bool]:
    """Per-bin squared systematic error (largest shift of each source, in quadrature).

    Returns ``(err2, has_syst)``; *has_syst* is False when no source moves any bin.
    """
    err2 = np.zeros_like(nominal_values, dtype=float)
    has_syst = False
    for dirs in (variations or {}).values():
        if "up" not in dirs or "down" not in dirs:
            continue
        delta = np.maximum(np.abs(dirs["up"].values() - nominal_values),
                           np.abs(dirs["down"].values() - nominal_values))
        # relative tolerance: summing the same histograms in a different order
        # leaves floating-point noise
        if np.any(delta > np.maximum(np.abs(nominal_values), 1.0) * 1e-10):
            has_syst = True
        err2 += delta ** 2
    return err2, has_syst
