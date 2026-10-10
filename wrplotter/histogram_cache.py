"""Histograms read from the analyzer's ROOT files, with cached file handles.

Read through :func:`read_hist` (or :func:`load_file` / :func:`load_and_rebin`):
it checks the key against the file's key list, listed once per file.  Asking
uproot directly for a key that does not exist is slow on files with systematic
variations: for a missing folder it lists the whole file (hundreds of folders)
to build its error message, about 50 ms per lookup against 0.06 ms for a read.
"""
from __future__ import annotations
import logging
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import uproot
from hist import Hist

from .histogram_utils import add_hists, rebin_histogram

logger = logging.getLogger(__name__)

# (file, top-level folder) already reported as missing where other eras have it
_PARTIAL_REPORTED: set[tuple[str, str]] = set()


@lru_cache(maxsize=64)
def _open_root(path: str):
    """Cache open ROOT file handles for reuse across histogram reads."""
    return uproot.open(path)


@lru_cache(maxsize=64)
def keys_in_file(path: str) -> frozenset[str]:
    """Every key of the ROOT file: ``folder/name`` and the folders themselves, no cycles."""
    return frozenset(_open_root(path).keys(recursive=True, cycle=False))


def clear_cache():
    """Release all cached ROOT file handles and key lists."""
    _open_root.cache_clear()
    keys_in_file.cache_clear()
    _PARTIAL_REPORTED.clear()


def has_key(path: str | Path, key: str) -> bool:
    """True if the file exists and holds *key*.  A file that exists but cannot be read raises."""
    return Path(path).is_file() and key in keys_in_file(str(path))


def read_hist(path: str | Path, key: str) -> Hist | None:
    """Histogram *key* of the ROOT file *path*; None if the file or the key does not exist.

    A file that exists but cannot be read raises: skipping it would silently drop
    its contents from sums over files.
    """
    if not has_key(path, key):
        return None
    return _open_root(str(path))[key].to_hist()


def load_file(
    path: str | Path,
    hist_key: str,
    n_rebin: int | list[float],
    *,
    density: bool = True,
    xmax: float | None = None,
) -> Hist | None:
    """*hist_key* of one file, rebinned (:func:`rebin_histogram`); None if absent."""
    raw = read_hist(path, hist_key)
    if raw is None:
        return None
    return rebin_histogram(raw, n_rebin, density=density, xmax=xmax)


def load_per_file(
    files: Sequence[str | Path],
    hist_key: str,
    n_rebin: int | list[float],
    *,
    density: bool = True,
    xmax: float | None = None,
) -> dict[Path, Hist | None]:
    """``{file: hist_key of that file, rebinned, or None}`` for the *files* that exist."""
    return {Path(f): load_file(f, hist_key, n_rebin, density=density, xmax=xmax)
            for f in files if Path(f).is_file()}


def report_partial(path: str | Path, key: str) -> None:
    """Report, once per file and folder, that *path* lacks *key* while other eras' files have it."""
    folder = key.split("/", 1)[0]
    if (str(path), folder) in _PARTIAL_REPORTED:
        return
    _PARTIAL_REPORTED.add((str(path), folder))
    logger.warning("%s has no %s while the other eras' files have it: the sum over eras leaves "
                   "this file out (other keys of %s/ missing there are not reported again).",
                   path, key, folder)


def sum_files(per_file: Mapping[Path, Hist | None], hist_key: str) -> Hist | None:
    """Sum of the histograms of *per_file* (one file per sub-era); None if none has it.

    A key that some files have and others lack is reported (:func:`report_partial`):
    the sum leaves those files out.
    """
    total = None
    for h in per_file.values():
        if h is not None:
            total = add_hists(total, h)
    if total is not None:
        for path, h in per_file.items():
            if h is None:
                report_partial(path, hist_key)
    return total


def load_and_rebin(
    input_dirs: Sequence[Path],
    sample: str,
    hist_key: str,
    n_rebin: int | list[float],
    *,
    density: bool = True,
    xmax: float | None = None,
) -> Hist | None:
    """Sum *hist_key* of *sample* over *input_dirs* (one per sub-era), rebinned.

    MC histograms are already scaled to luminosity by the analyzer.  Missing
    files are skipped (``Inputs.files_for`` reports them); a key missing from
    some of the files is reported (:func:`sum_files`).
    """
    files = [Path(d) / f"WRAnalyzer_{sample}.root" for d in input_dirs]
    per_file = load_per_file(files, hist_key, n_rebin, density=density, xmax=xmax)
    return sum_files(per_file, hist_key)
