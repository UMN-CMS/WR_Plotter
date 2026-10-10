"""Where each sample's ROOT file lives, and what the analyzer recorded about it.

An analysis set is one ``--dir`` under ``rootfiles/<Run>/<Year>/<Era>`` (one
directory per sub-era for combined eras such as 2022).  Every sample is read
from there except DYJets, which comes from the directory of the chosen
Drell-Yan sample: the analyzer always names the file WRAnalyzer_DYJets.root, so
the DY sample is told apart by directory only (see data/dy_samples.yaml).

Next to each ROOT file the analyzer writes a manifest,
``WRAnalyzer_<sample>_systematics.json``.  It lists the datasets the file was
made from and, for data, whether signal-region data was withheld
(``blinding.applied``).  Blinding happens in the analyzer; the plotter only
reads its record.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from .config import load_dy_samples, load_lumi
from .paths import input_dirs_for_era, repo_root

logger = logging.getLogger(__name__)

DY_SAMPLE = "DYJets"

# (era, dir, sample) already reported as missing in some of a combination's eras
_PARTIAL_REPORTED: set[tuple[str, str, str]] = set()


def root_file(directory: str | Path, sample: str) -> Path:
    return Path(directory) / f"WRAnalyzer_{sample}.root"


def mass_point(name: str) -> tuple[int, int] | None:
    """``WR2000_N1100`` or ``signal_WR2000_N1100`` -> (2000, 1100); None for other names."""
    m = re.fullmatch(r"(?:signal_)?WR(\d+)_N(\d+)", name)
    return (int(m[1]), int(m[2])) if m else None


def manifest_path(root_path: str | Path) -> Path:
    root_path = Path(root_path)
    return root_path.with_name(root_path.stem + "_systematics.json")


@lru_cache(maxsize=256)
def _read_json(path: str) -> dict | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unreadable manifest %s: %s", p, exc)
        return None


def read_manifest(root_path: str | Path) -> dict | None:
    """The analyzer's manifest for a ROOT file, or None (no manifest written)."""
    return _read_json(str(manifest_path(root_path)))


def clear_manifest_cache() -> None:
    _read_json.cache_clear()


def blinding_recorded(root_path: str | Path) -> bool:
    """Whether the analyzer recorded its blinding decision for this file.

    Files without the record predate analyzer-side blinding: their data still
    contains the signal regions and the full cutflows.
    """
    return isinstance((read_manifest(root_path) or {}).get("blinding"), dict)


def sr_data_unblinded(root_path: str | Path) -> bool:
    """Whether a data file may show signal-region data.

    Only a file whose manifest says blinding was *not* applied (a Run 2
    analyzer run made with ``--unblind``) qualifies.  Blinded files carry no
    signal-region data anyway, and files without the record are treated as
    blinded.
    """
    blinding = (read_manifest(root_path) or {}).get("blinding")
    return isinstance(blinding, dict) and blinding.get("applied") is False


# ---------------------------------------------------------------------------
# Drell-Yan samples
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DYSample:
    tag: str                 # analyzer --dy tag, e.g. "lo_ht"
    label: str               # legend label
    color: str
    dirs: tuple[Path, ...]   # directory of WRAnalyzer_DYJets.root, one per sub-era


def dy_dir_name(dir_name: str, tag: str) -> str:
    """Directory holding the DY sample *tag* for the analysis set *dir_name*."""
    if tag == load_dy_samples()["default"]:
        return dir_name
    return f"{dir_name}_{tag}_dy" if dir_name else f"{tag}_dy"


def _dy_datasets(manifest: dict) -> list[str]:
    datasets = manifest.get("datasets") or {}
    return [name for name, meta in datasets.items()
            if (meta or {}).get("physics_group", DY_SAMPLE) == DY_SAMPLE]


def detect_dy_tag(root_path: str | Path) -> str | None:
    """The DY sample whose dataset pattern matches every dataset in the file, if unique."""
    datasets = _dy_datasets(read_manifest(root_path) or {})
    if not datasets:
        return None
    samples = load_dy_samples()["samples"]
    matches = [tag for tag, info in samples.items()
               if info.get("datasets")
               and all(re.search(info["datasets"], d) for d in datasets)]
    return matches[0] if len(matches) == 1 else None


def resolve_dy(era: str, dir_name: str, spec: str | None = None,
               working_dir: str | Path | None = None) -> DYSample:
    """Resolve ``--dy TAG`` or ``--dy TAG=DIR`` (None = the default sample).

    DIR is a directory name under each sub-era, like ``--dir``.  A directory
    whose DYJets file was made from another DY sample is refused.
    """
    cfg = load_dy_samples()
    samples = cfg["samples"]
    tag, _, explicit = (spec or cfg["default"]).partition("=")
    tag, explicit = tag.strip(), explicit.strip()
    if tag not in samples:
        raise ValueError(f"Unknown DY sample '{tag}'. Known: {', '.join(sorted(samples))} "
                         "(data/dy_samples.yaml).")

    sub_dir = explicit or dy_dir_name(dir_name, tag)
    dirs, _ = input_dirs_for_era(era, working_dir or repo_root(), sub_dir)
    missing = [str(d) for d in dirs if not root_file(d, DY_SAMPLE).is_file()]
    if missing:
        raise FileNotFoundError(
            f"No WRAnalyzer_{DY_SAMPLE}.root for DY sample '{tag}' in {missing}. "
            f"Run the analyzer with --dy {tag} --dir {sub_dir}, or point to it with "
            f"--dy {tag}=<dir>."
        )
    for d in dirs:
        found = detect_dy_tag(root_file(d, DY_SAMPLE))
        if found is not None and found != tag:
            raise ValueError(
                f"{root_file(d, DY_SAMPLE)} holds the '{found}' DY sample, not '{tag}' "
                "(dataset names in its manifest)."
            )

    info = samples[tag]
    return DYSample(tag=tag, label=str(info.get("label", tag)),
                    color=str(info.get("color", "#000000")), dirs=tuple(dirs))


# ---------------------------------------------------------------------------
# Analysis set
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Inputs:
    """ROOT-file locations of one analysis set with one Drell-Yan sample."""

    era: str
    dir_name: str
    dirs: tuple[Path, ...]   # base directories, one per sub-era
    dy: DYSample | None      # None: made with make_inputs(..., with_dy=False)

    def dirs_for(self, sample: str) -> list[Path]:
        if sample != DY_SAMPLE:
            return list(self.dirs)
        if self.dy is None:
            raise ValueError(f"No Drell-Yan sample chosen for {self.era}/{self.dir_name} "
                             "(inputs made with with_dy=False).")
        return list(self.dy.dirs)

    def files_for(self, sample: str) -> list[Path]:
        """Existing files of *sample*, one per sub-era directory that has it.

        A sample present in only some eras of a combination is reported once:
        summing what exists would undercount it.
        """
        dirs = self.dirs_for(sample)
        files = [p for d in dirs if (p := root_file(d, sample)).is_file()]
        if files and len(files) < len(dirs):
            key = (self.era, self.dir_name, sample)
            if key not in _PARTIAL_REPORTED:
                _PARTIAL_REPORTED.add(key)
                missing = [str(d) for d in dirs if not root_file(d, sample).is_file()]
                logger.warning("%s exists in only %d of the %d eras of %s; missing in %s. "
                               "The combined %s is undercounted.", root_file("", sample).name,
                               len(files), len(dirs), self.era, ", ".join(missing), sample)
        return files

    def signal_points(self) -> list[str]:
        """Signal mass points with a file in any sub-era (``WR2000_N1100``), by WR then N mass.

        A point missing in some eras of a combination is reported by
        :meth:`files_for` when it is read.
        """
        names = {p.name.removeprefix("WRAnalyzer_signal_").removesuffix(".root")
                 for d in self.dirs for p in d.glob("WRAnalyzer_signal_*.root")}
        return sorted((n for n in names if mass_point(n)), key=mass_point)

    def with_dy(self, dy: DYSample) -> "Inputs":
        return replace(self, dy=dy)


def make_inputs(era: str, dir_name: str = "", dy_specs: Sequence[str] | None = None,
                working_dir: str | Path | None = None, *, with_dy: bool = True) -> list[Inputs]:
    """One :class:`Inputs` per requested DY sample (the default one when none).

    ``with_dy=False`` gives a single :class:`Inputs` without a DY sample, for
    readers of other samples only (e.g. signal-only analysis sets, which have
    no DYJets file).

    Raises FileNotFoundError when an era directory or a DY file is missing, and
    ValueError for an unknown or mislabelled DY sample.
    """
    load_lumi(era)  # validates the era
    dirs, _ = input_dirs_for_era(era, working_dir or repo_root(), dir_name)
    absent = [str(d) for d in dirs if not d.is_dir()]
    if absent:
        raise FileNotFoundError(f"Input directory not found: {', '.join(absent)}")
    if not with_dy:
        if dy_specs:
            raise ValueError("DY samples requested with with_dy=False.")
        return [Inputs(era=era, dir_name=dir_name, dirs=tuple(dirs), dy=None)]
    specs = list(dy_specs) if dy_specs else [None]
    out = []
    for spec in specs:
        dy = resolve_dy(era, dir_name, spec, working_dir)
        if any(i.dy.tag == dy.tag for i in out):
            raise ValueError(f"DY sample '{dy.tag}' requested twice.")
        out.append(Inputs(era=era, dir_name=dir_name, dirs=tuple(dirs), dy=dy))
    return out
