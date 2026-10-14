"""Where each sample's ROOT file lives, and what the analyzer recorded about it.

An analysis set is one ``--dir`` under ``rootfiles/<Run>/<Year>/<Era>`` (one
directory per sub-era for combined eras such as 2022).  Every sample is read
from there, DYJets from the file of the chosen Drell-Yan sample:
WRAnalyzer_DYJets.root for the default sample, WRAnalyzer_DYJets_<tag>.root for
any other.  Sets made before 2026-10-12 kept every DY sample as
WRAnalyzer_DYJets.root in its own directory, <dir>_<tag>_dy; a tag whose file is
not in ``--dir`` is read from there (see data/dy_samples.yaml).

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
    tag: str                 # data/dy_samples.yaml tag, e.g. "lo_ht" (never an alias)
    label: str               # full name (the stack plots' "DY: ..." line)
    color: str
    files: tuple[Path, ...]  # its DYJets file, one per sub-era
    short: str = ""          # legend name in DY comparisons, where it shares the legend; default: label

    @property
    def legend(self) -> str:
        return self.short or self.label


def canonical_dy_tag(name: str) -> str:
    """The data/dy_samples.yaml tag *name* stands for: itself, or the tag listing it in ``aliases``."""
    samples = load_dy_samples()["samples"]
    if name in samples:
        return name
    for tag, info in samples.items():
        if name in (info.get("aliases") or ()):
            return tag
    raise ValueError(f"Unknown DY sample '{name}'. Known: {', '.join(sorted(samples))} "
                     "(data/dy_samples.yaml).")


def dy_file_name(tag: str) -> str:
    """The analyzer's file for the DY sample *tag*: WRAnalyzer_DYJets.root for the default
    sample, WRAnalyzer_DYJets_<tag>.root for any other (WrCoffea cli_utils.dy_file_tag)."""
    if tag == load_dy_samples()["default"]:
        return root_file("", DY_SAMPLE).name
    return root_file("", f"{DY_SAMPLE}_{tag}").name


def legacy_dy_dir_names(dir_name: str, tag: str) -> list[str]:
    """The old-layout directories (sets made before 2026-10-12) that held *tag* as
    WRAnalyzer_DYJets.root: <dir>_<name>_dy for the tag and its aliases, then its legacy_dirs."""
    info = load_dy_samples()["samples"][tag]
    names = [f"{{dir}}_{n}_dy" for n in (tag, *(info.get("aliases") or ()))]
    names += list(info.get("legacy_dirs") or ())
    return [n.format(dir=dir_name).lstrip("_") for n in names]


def _dy_datasets(manifest: dict) -> list[str]:
    datasets = manifest.get("datasets") or {}
    return [name for name, meta in datasets.items()
            if (meta or {}).get("physics_group", DY_SAMPLE) == DY_SAMPLE]


def detect_dy_tag(root_path: str | Path) -> str | None:
    """The DY sample whose dataset pattern matches every dataset in the file, if unique.

    Samples sharing a pattern (lo_ht and its lo_ht_* variants) count as
    one; the first listed is returned.
    """
    datasets = _dy_datasets(read_manifest(root_path) or {})
    if not datasets:
        return None
    samples = load_dy_samples()["samples"]
    matches = [tag for tag, info in samples.items()
               if info.get("datasets")
               and all(re.search(info["datasets"], d) for d in datasets)]
    patterns = {samples[tag]["datasets"] for tag in matches}
    return matches[0] if len(patterns) == 1 else None


# The DY reweighting factors (--dy-reweight) and the per-factor event-weight histograms
# the analyzer writes for them, event_weights/<region>/components/<factor>, MC only.
_DY_REWEIGHT_FACTORS = (("zpt", "dy_zpt"), ("reshape", "dy_reshape"))


def expected_dy_reweight(tag: str) -> str | None:
    """The ``--dy-reweight`` mode the DY sample *tag* stands for: its ``analyzer_args``,
    "none" for the default sample (not reweighted since 2026-10-12), None for samples
    the reweighting does not target."""
    cfg = load_dy_samples()
    tag = canonical_dy_tag(tag)
    m = re.search(r"--dy-reweight\s+(\S+)", cfg["samples"][tag].get("analyzer_args") or "")
    if m:
        return m[1]
    return "none" if tag == cfg["default"] else None


def dy_reweight_in_file(root_path: str | Path) -> str | None:
    """The ``--dy-reweight`` mode a DYJets file was made with ("none", "zpt" or
    "zpt+reshape"), read from its per-factor event-weight histograms: a factor counts as
    applied when some events have a value other than 1.  None for files without them."""
    from .histogram_cache import keys_in_file, read_hist   # lazy: histogram_cache is heavier

    keys = keys_in_file(str(root_path))
    regions = sorted({k.split("/")[1] for k in keys
                      if k.startswith("event_weights/") and "/components/" in k})
    if not regions:
        return None
    applied = []
    for mode, factor in _DY_REWEIGHT_FACTORS:
        for region in regions:
            h = read_hist(root_path, f"event_weights/{region}/components/{factor}")
            if h is None or h.values().sum() <= 0:
                continue
            at_one = h.axes[0].index(1.0)
            values = h.values()
            if values.sum() - values[at_one] > 0:
                applied.append(mode)
            break
    return "+".join(applied) or "none"


def resolve_dy(era: str, dir_name: str, spec: str | None = None,
               working_dir: str | Path | None = None) -> DYSample:
    """Resolve ``--dy TAG`` or ``--dy TAG=DIR`` (None = the default sample).

    The DY file is the tag's file in ``--dir`` or, for sets made before 2026-10-12,
    WRAnalyzer_DYJets.root in one of its old-layout directories.  DIR is a directory
    name under each sub-era, like ``--dir``, holding the tag's file or (old layout)
    WRAnalyzer_DYJets.root.  A file made from another DY sample is refused, and so is one
    whose DY reweighting (read from the file) is not the one *TAG* stands for.
    For the implicit default (*spec* None) that is only a warning: analysis sets
    made before 2026-10-12 hold the reweighted DY in ``--dir``.
    """
    cfg = load_dy_samples()
    samples = cfg["samples"]
    name, _, explicit = (spec or cfg["default"]).partition("=")
    tag, explicit = canonical_dy_tag(name.strip()), explicit.strip()
    info = samples[tag]
    working_dir = working_dir or repo_root()
    fname, plain = dy_file_name(tag), root_file("", DY_SAMPLE).name
    if explicit:
        dirs, _ = input_dirs_for_era(era, working_dir, explicit)
        candidates = [[d / fname, d / plain] for d in dirs]
        old = []
    else:
        dirs, _ = input_dirs_for_era(era, working_dir, dir_name)
        old = legacy_dy_dir_names(dir_name, tag) if tag != cfg["default"] else []
        old_dirs = [input_dirs_for_era(era, working_dir, n)[0] for n in old]
        candidates = [[d / fname] + [od[i] / plain for od in old_dirs] for i, d in enumerate(dirs)]
    files = [next((p for p in cands if p.is_file()), None) for cands in candidates]
    missing = [str(cands[0]) for cands, f in zip(candidates, files) if f is None]
    if missing:
        run = info.get("analyzer_args") or f"--dy {tag}"
        raise FileNotFoundError(
            f"No {fname} for DY sample '{tag}': {', '.join(missing)}"
            + (f" (nor {plain} in the old-layout {', '.join(old)})" if old else "")
            + f". Run the analyzer with {run} --dir {explicit or dir_name}, or point to it with "
            f"--dy {tag}=<dir>."
        )
    for f in files:
        found = detect_dy_tag(f)
        if found is not None and samples[found]["datasets"] != info.get("datasets"):
            raise ValueError(
                f"{f} holds the '{found}' DY sample, not '{tag}' "
                "(dataset names in its manifest)."
            )
        expected = expected_dy_reweight(tag)
        made_with = dy_reweight_in_file(f) if expected else None
        if made_with is not None and made_with != expected:
            right = [t for t in samples if samples[t].get("datasets") == info.get("datasets")
                     and expected_dy_reweight(t) == made_with]
            msg = (f"{f} holds the DY made with --dy-reweight {made_with} "
                   f"(its event-weight histograms), but '{tag}' stands for --dy-reweight {expected}")
            msg += f": read it with --dy {right[0]}={f.parent.name}." if right else "."
            if spec is None:
                logger.warning(msg)
            else:
                raise ValueError(msg)

    return DYSample(tag=tag, label=str(info.get("label", tag)),
                    color=str(info.get("color", "#000000")), files=tuple(files),
                    short=str(info.get("short", "")))


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

    def _dy(self) -> DYSample:
        if self.dy is None:
            raise ValueError(f"No Drell-Yan sample chosen for {self.era}/{self.dir_name} "
                             "(inputs made with with_dy=False).")
        return self.dy

    def dirs_for(self, sample: str) -> list[Path]:
        if sample != DY_SAMPLE:
            return list(self.dirs)
        return [f.parent for f in self._dy().files]

    def files_for(self, sample: str) -> list[Path]:
        """Existing files of *sample*, one per sub-era directory that has it (DYJets: the
        chosen DY sample's file, resolved in every sub-era).

        A sample present in only some eras of a combination is reported once:
        summing what exists would undercount it.
        """
        if sample == DY_SAMPLE:
            return list(self._dy().files)
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
