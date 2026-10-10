"""Cutflow tables from the analyzer's cutflow histograms.

The analyzer writes ``cutflow_<topology>/<channel>/<kind>[_unweighted]`` with
topology resolved/boosted, channel ee/mumu/em and kind cumulative/onecut, each
a histogram whose bin labels are the cut names (read from the file, so the
sequence always matches the analyzer):

- ``cumulative[i]``: events passing cuts 0..i; the last ee/mumu row is the
  signal region, the last em row the flavor CR (resolved) or
  wr_emu_boosted_flavor_cr (boosted);
- ``onecut[i]``: events passing cut i alone (not N-1);
- row 0, ``no_cuts``: every event after the event filters (and the skim);
- weighted MC bins carry the base event weight, without the per-region lepton
  scale factors, so the last row sits a few percent above the region histogram;
  ``_unweighted`` holds raw event counts.

Blinding is done by the analyzer: data cutflows stop before the
m(ll) > 200 GeV cut in ee and mumu.  Rows a data cutflow lacks are shown as
blinded, and a data file without the analyzer's blinding record (made before
analyzer-side blinding) gives no data column at all.

The data column is the file of the channel's trigger stream
(data/cutflows.yaml); both data files fill every channel.
"""
from __future__ import annotations

import logging
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import load_cutflow_config
from .histogram_cache import _open_root, has_key, report_partial
from .inputs import Inputs, blinding_recorded, mass_point
from .sample_groups import load_sample_groups

logger = logging.getLogger(__name__)

TOPOLOGIES = ("resolved", "boosted")
CHANNELS = ("ee", "mumu", "em")
KINDS = ("cumulative", "onecut")

# Efficiency modes per cutflow kind.  For onecut, "running" (row / no_cuts) is
# the single-cut efficiency.
EFFICIENCIES = {"cumulative": ("running", "relative"), "onecut": ("running",)}

CHANNEL_LATEX = {"ee": "ee", "mumu": r"$\mu\mu$", "em": r"$e\mu$"}
CHANNEL_WORDS = {"ee": "dielectron", "mumu": "dimuon", "em": "electron--muon"}


@dataclass(frozen=True)
class Cutflow:
    cuts: tuple[str, ...]
    values: np.ndarray
    variances: np.ndarray

    def __add__(self, other: "Cutflow") -> "Cutflow":
        if other.cuts != self.cuts:
            raise ValueError(f"Cannot add cutflows with different cuts: {self.cuts} vs {other.cuts}")
        return Cutflow(self.cuts, self.values + other.values, self.variances + other.variances)


def cutflow_key(topology: str, channel: str, kind: str, unweighted: bool = False) -> str:
    return f"cutflow_{topology}/{channel}/{kind}{'_unweighted' if unweighted else ''}"


def read_cutflow(path: str | Path, topology: str, channel: str, kind: str,
                 unweighted: bool = False) -> Cutflow | None:
    """One cutflow histogram of one file, or None if the file or key is missing.

    A file that exists but cannot be read raises.
    """
    key = cutflow_key(topology, channel, kind, unweighted)
    if not has_key(path, key):
        return None
    h = _open_root(str(path))[key]
    values = np.asarray(h.values(flow=False), dtype=float)
    labels = h.axis().labels()
    if labels is None or len(labels) != len(values):
        logger.warning("%s:%s has no cut labels; skipped.", path, key)
        return None
    return Cutflow(tuple(str(x) for x in labels), values,
                   np.asarray(h.variances(flow=False), dtype=float))


def read_files(files: Sequence[str | Path], topology: str, channel: str, kind: str,
               unweighted: bool = False) -> Cutflow | None:
    """Cutflow summed over *files* (e.g. the sub-eras of a combined era).

    A file without the cutflow while others have it is reported: the sum leaves it out.
    """
    key = cutflow_key(topology, channel, kind, unweighted)
    total = None
    for f in files:
        cf = read_cutflow(f, topology, channel, kind, unweighted)
        if cf is None:
            continue
        if total is not None and cf.cuts != total.cuts:
            raise ValueError(f"{f}:{key} has the cuts "
                             f"{list(cf.cuts)}, the files before it {list(total.cuts)}.")
        total = cf if total is None else total + cf
    if total is not None:
        for f in files:
            if Path(f).is_file() and not has_key(f, key):
                report_partial(f, key)
    return total


def signal_cutflows(inputs: Inputs, points: Sequence[str], topology: str, channel: str,
                    kind: str, unweighted: bool = False) -> dict[str, Cutflow]:
    """Cutflow of each signal mass point (e.g. ``WR2000_N1100``), in the order given.

    Points without this cutflow are skipped with a warning.  Raises ValueError
    when the points do not all share one cut sequence.
    """
    key = cutflow_key(topology, channel, kind, unweighted)
    flows = {}
    for point in points:
        cf = read_files(inputs.files_for(f"signal_{point}"), topology, channel, kind, unweighted)
        if cf is None:
            logger.warning("No %s cutflow for signal %s.", key, point)
        else:
            flows[point] = cf
    by_cuts: dict[tuple[str, ...], list[str]] = {}
    for point, cf in flows.items():
        by_cuts.setdefault(cf.cuts, []).append(point)
    if len(by_cuts) > 1:
        detail = "\n".join(f"  {', '.join(pts)}: {', '.join(cuts)}" for cuts, pts in by_cuts.items())
        raise ValueError(f"The signal points do not share one {key} cut sequence:\n{detail}")
    return flows


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

@dataclass
class Column:
    key: str        # sample-group key or signal mass point (e.g. WR2000_N1100)
    label: str      # LaTeX / legend label
    color: str
    cutflow: Cutflow


@dataclass
class CutflowTable:
    topology: str
    channel: str
    kind: str
    unweighted: bool
    cuts: tuple[str, ...]
    backgrounds: list[Column]              # largest-yield groups first
    signals: list[Column] = field(default_factory=list)
    data: np.ndarray | None = None         # aligned to cuts; NaN where blinded
    data_note: str = ""                    # why data is absent or partly blinded

    @property
    def total(self) -> np.ndarray:
        return np.sum([c.cutflow.values for c in self.backgrounds], axis=0) \
            if self.backgrounds else np.zeros(len(self.cuts))

    @property
    def total_variance(self) -> np.ndarray:
        return np.sum([c.cutflow.variances for c in self.backgrounds], axis=0) \
            if self.backgrounds else np.zeros(len(self.cuts))

    @property
    def blinded(self) -> np.ndarray:
        """Rows whose data is withheld."""
        if self.data is None:
            return np.zeros(len(self.cuts), dtype=bool)
        return np.isnan(self.data)

    @property
    def name(self) -> str:
        return f"{self.kind}_{self.topology}_{self.channel}" + ("_unweighted" if self.unweighted else "")


def signal_column_label(point: str) -> str:
    """WR2000_N1100 -> (2000,1100)."""
    masses = mass_point(point)
    return f"({masses[0]},{masses[1]})" if masses else point


def _group_cutflow(inputs: Inputs, samples, topology, channel, kind, unweighted):
    total = None
    for sample in samples:
        cf = read_files(inputs.files_for(sample), topology, channel, kind, unweighted)
        if cf is not None:
            total = cf if total is None else total + cf
    return total


def _data_column(inputs: Inputs, groups, topology, channel, kind, unweighted, cuts):
    stream = load_cutflow_config()["data_stream"].get(topology, {}).get(channel)
    group = groups.get(stream)
    if group is None:
        return None, f"no data stream configured for {topology}/{channel} (data/cutflows.yaml)"
    files = [f for s in group.samples for f in inputs.files_for(s)]
    if not files:
        return None, "no data file"
    legacy = [f for f in files if not blinding_recorded(f)]
    if legacy:
        return None, (f"{legacy[0].name} has no analyzer blinding record (made before "
                      "analyzer-side blinding); rerun the data to show it")
    cf = read_files(files, topology, channel, kind, unweighted)
    if cf is None:
        return None, "no data cutflow"
    unknown = [c for c in cf.cuts if c not in cuts]
    if unknown:
        raise ValueError(f"Data cutflow {topology}/{channel}/{kind} has cuts {unknown} "
                         f"that the MC cutflow lacks: {list(cuts)}")
    aligned = np.full(len(cuts), np.nan)
    for cut, value in zip(cf.cuts, cf.values):
        aligned[cuts.index(cut)] = value
    n_blind = int(np.isnan(aligned).sum())
    note = f"{n_blind} row(s) blinded by the analyzer" if n_blind else ""
    return aligned, note


def build_table(inputs: Inputs, topology: str, channel: str, kind: str, *,
                unweighted: bool = False, signal_points: Sequence[str] = ()) -> CutflowTable | None:
    """Background, signal and data columns of one cutflow; None if no MC has it."""
    groups, order = load_sample_groups()
    backgrounds = []
    for gkey in reversed(order):  # the stack's top (largest) groups first
        group = groups[gkey]
        if group.kind != "mc":
            continue
        cf = _group_cutflow(inputs, group.samples, topology, channel, kind, unweighted)
        if cf is not None:
            backgrounds.append(Column(gkey, group.tlatex_alias, group.color, cf))
    signals = [Column(point, signal_column_label(point), "", cf) for point, cf in
               signal_cutflows(inputs, signal_points, topology, channel, kind, unweighted).items()]

    mc = backgrounds + signals
    if not mc:
        return None
    cuts = mc[0].cutflow.cuts
    for col in mc[1:]:
        if col.cutflow.cuts != cuts:
            raise ValueError(f"Cut sequence of {col.key} differs from {mc[0].key}: "
                             f"{col.cutflow.cuts} vs {cuts}")
    data, note = _data_column(inputs, groups, topology, channel, kind, unweighted, cuts)
    return CutflowTable(topology, channel, kind, unweighted, cuts, backgrounds, signals, data, note)


def table_variant(kind: str, unweighted: bool = False, efficiency_mode: str | None = None) -> str:
    """Name of a table/plot variant, used in file names and LaTeX labels.

    Yields: ``cumulative``, ``onecut`` (+ ``_unweighted``); efficiencies:
    ``running_eff``, ``relative_eff`` (cumulative) and ``single_eff`` (onecut).
    """
    if efficiency_mode is None:
        return kind + ("_unweighted" if unweighted else "")
    if kind == "onecut" and efficiency_mode == "running":
        return "single_eff"
    return f"{efficiency_mode}_eff"


def efficiency(values: np.ndarray, mode: str) -> np.ndarray:
    """Fractions: ``running`` = row / row 0, ``relative`` = row / previous row (first = 1)."""
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan)
    if mode == "running":
        if values[0] > 0:
            out = values / values[0]
    elif mode == "relative":
        out[0] = 1.0 if values[0] > 0 else np.nan
        prev = values[:-1]
        np.divide(values[1:], prev, out=out[1:], where=prev > 0)
    else:
        raise ValueError(f"Unknown efficiency mode {mode!r}")
    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def cut_label(cut: str, which: str = "table") -> str:
    """Display label of a cut (data/cutflows.yaml), falling back to the key."""
    entry = load_cutflow_config()["labels"].get(cut)
    if isinstance(entry, dict) and entry.get(which):
        return str(entry[which])
    return cut.replace("_", r"\_") if which == "table" else cut


def format_count(x: float, unweighted: bool = False) -> str:
    """A yield as plain text."""
    if unweighted:
        return f"{x:.0f}"
    if x == 0:
        return "0"
    if abs(x) >= 1e5:
        return f"{x:.3e}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    if abs(x) >= 0.01:
        return f"{x:.3f}"
    return f"{x:.2e}"


def format_percent(x: float) -> str:
    """A fraction as a percentage ("-" when undefined)."""
    if not np.isfinite(x):
        return "-"
    x *= 100
    if x >= 99.995:
        return "100"
    if x >= 10:
        return f"{x:.1f}"
    if x >= 0.1:
        return f"{x:.2f}"
    return f"{x:.2g}"


def latex_count(x: float, unweighted: bool = False) -> str:
    """A yield for a LaTeX table (scientific notation from 10 up and below 0.01)."""
    if unweighted:
        return f"{x:.0f}"
    if x == 0:
        return "0"
    if 0.01 <= abs(x) < 10:
        return f"{x:.2f}"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / 10 ** exp
    if round(mant, 2) >= 10:  # 9.997 -> 10.00
        exp += 1
        mant /= 10
    return f"${mant:.2f} \\times 10^{{{exp}}}$"


def _cells(values: np.ndarray, efficiency_mode: str | None) -> np.ndarray:
    return efficiency(values, efficiency_mode) if efficiency_mode else values


def format_text(table: CutflowTable, *, efficiency_mode: str | None = None) -> str:
    """Plain-text table: backgrounds, total, data, data/bkg., signals."""
    eff = efficiency_mode
    has_data = table.data is not None
    headers = ["Cut"] + [c.key for c in table.backgrounds] + ["Total bkg."]
    if has_data:
        headers += ["Data"] + ([] if eff else ["Data/Bkg."])
    headers += [c.key for c in table.signals]

    total = _cells(table.total, eff)
    data = _cells(table.data, eff) if has_data else None
    fmt = (lambda x: format_percent(x)) if eff else (lambda x: format_count(x, table.unweighted))
    rows = []
    for i, cut in enumerate(table.cuts):
        row = [cut] + [fmt(_cells(c.cutflow.values, eff)[i]) for c in table.backgrounds]
        row.append(fmt(total[i]))
        if has_data:
            if np.isnan(table.data[i]):
                row += ["blinded"] + ([] if eff else ["-"])
            else:
                row.append(fmt(data[i]))
                if not eff:
                    row.append(f"{table.data[i] / table.total[i]:.3f}" if table.total[i] > 0 else "-")
        row += [fmt(_cells(c.cutflow.values, eff)[i]) for c in table.signals]
        rows.append(row)

    widths = [max(len(str(r[j])) for r in [headers] + rows) for j in range(len(headers))]
    line = lambda r: "  ".join(str(r[j]).rjust(widths[j]) if j else str(r[j]).ljust(widths[j])
                               for j in range(len(headers)))
    what = {None: "yields", "running": "efficiency w.r.t. no_cuts [%]",
            "relative": "efficiency w.r.t. previous cut [%]"}[eff]
    title = (f"{table.topology} {table.channel}, {table.kind}, "
             f"{'unweighted' if table.unweighted else 'weighted'} {what}")
    out = [title, line(headers), "  ".join("-" * w for w in widths)] + [line(r) for r in rows]
    if table.data_note:
        out.append(f"Data: {table.data_note}")
    return "\n".join(out)


def format_latex(table: CutflowTable, *, era: str, lumi: float, year: str,
                 efficiency_mode: str | None = None) -> str:
    """CMS-style LaTeX table; signal columns after a vertical rule."""
    eff = efficiency_mode
    has_data = table.data is not None
    n_bkg = len(table.backgrounds) + 1 + int(has_data)
    col_spec = "l | " + " ".join(["c"] * n_bkg)
    if table.signals:
        col_spec += " | " + " ".join(["c"] * len(table.signals))

    header = (["Selection"] + [c.label for c in table.backgrounds] + ["Total bkg."]
              + (["Data"] if has_data else []) + [c.label for c in table.signals])
    fmt = (lambda x: format_percent(x)) if eff else (lambda x: latex_count(x, table.unweighted))

    total = _cells(table.total, eff)
    data = _cells(table.data, eff) if has_data else None
    lines = []
    if eff == "running":
        what = ("single-cut efficiencies (\\%) relative to the first row" if table.kind == "onecut"
                else "running efficiencies (\\%) relative to the first row")
    elif eff == "relative":
        what = "efficiencies (\\%) relative to the previous row"
    else:
        what = "raw simulated event counts" if table.unweighted else "expected event yields"
    rows_desc = ("each row applies that single cut alone" if table.kind == "onecut"
                 else "each row applies that cut and all preceding cuts")
    notes = []
    if not table.unweighted:
        notes.append("Weighted simulation uses the base event weight, without lepton scale factors.")
    if has_data and table.blinded.any():
        notes.append("Data in the signal-region rows is blinded.")
    caption = (f"{era}: {what} for the {table.topology}, {CHANNEL_WORDS.get(table.channel, table.channel)} "
               f"channel; {rows_desc}. Signal columns show $(m_{{W_R}},\\,m_{{N}})$ in GeV. "
               f"Normalized to ${lumi:.1f}\\,\\mathrm{{fb}}^{{-1}}$ ({year}). " + " ".join(notes))
    lines.append(r"\begin{table}[htp]")
    lines.append(f"  \\topcaption{{{caption.strip()}}}")
    variant = table_variant(table.kind, table.unweighted, eff)
    lines.append(f"  \\label{{tab:cutflow_{variant}_{table.topology}_{table.channel}_{year}}}")
    lines.append(r"  \centering")
    lines.append(r"  \cmsTable{")
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"      \hline")
    lines.append("      " + " & ".join(header) + r" \\")
    lines.append(r"      \hline")
    for i, cut in enumerate(table.cuts):
        cells = [cut_label(cut, "table")]
        cells += [fmt(_cells(c.cutflow.values, eff)[i]) for c in table.backgrounds]
        cells.append(fmt(total[i]))
        if has_data:
            cells.append(r"\textit{blinded}" if np.isnan(table.data[i]) else fmt(data[i]))
        cells += [fmt(_cells(c.cutflow.values, eff)[i]) for c in table.signals]
        lines.append("      " + " & ".join(cells) + r" \\")
    lines.append(r"      \hline")
    lines.append(r"    \end{tabular}")
    lines.append(r"  }")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX preview
# ---------------------------------------------------------------------------

PREVIEW_HEAD = r"""\documentclass[12pt]{article}
\usepackage[landscape,margin=0.3in,paperwidth=24in,paperheight=16in]{geometry}
\usepackage{amsmath}
\usepackage{float}

%% CMS style stubs
\usepackage{caption}
\captionsetup{font=LARGE}
\newcommand{\topcaption}[1]{\caption{#1}}
\newcommand{\cmsTable}[1]{\Large #1}
\setlength{\floatsep}{4pt}
\setlength{\textfloatsep}{4pt}
\setlength{\intextsep}{4pt}

%% Allow many floats per page
\renewcommand{\topfraction}{1.0}
\renewcommand{\bottomfraction}{1.0}
\renewcommand{\textfraction}{0.0}
\renewcommand{\floatpagefraction}{0.99}
\setcounter{totalnumber}{10}
\setcounter{topnumber}{10}
\setcounter{bottomnumber}{10}

\begin{document}
"""


def write_preview(out_dir: Path, sections: Sequence[tuple[str, Sequence[str]]], title: str = "",
                  notes: Sequence[str] = ()) -> Path:
    """``out_dir/preview.tex`` inputting the table files of each (LaTeX heading, file names)
    section; *notes* (LaTeX) are printed under the title."""
    lines = [PREVIEW_HEAD]
    if title:
        lines.append(f"{{\\LARGE \\bfseries {title}}}\\par\\medskip")
    for note in notes:
        lines.append(f"{{\\Large {note}}}\\par")
    for i, (heading, files) in enumerate(s for s in sections if s[1]):
        if i:
            lines.append(r"\clearpage")
        lines.append(f"{{\\LARGE \\bfseries {heading}}}\\bigskip")
        for j, name in enumerate(files):
            if j:
                lines.append(r"\vspace{3em}")
            lines.append(f"\\input{{{name}}}")
    lines.append(r"\end{document}")
    path = Path(out_dir) / "preview.tex"
    path.write_text("\n".join(lines) + "\n")
    return path


def compile_preview(path: Path) -> bool:
    """Run pdflatex on *path* in its directory; False (with the log tail) on failure."""
    result = subprocess.run(["pdflatex", "-interaction=nonstopmode", path.name],
                            cwd=str(path.parent), capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("pdflatex failed on %s:\n%s", path, "\n".join(result.stdout.splitlines()[-15:]))
        return False
    logger.info("Preview PDF: %s", path.with_suffix(".pdf"))
    return True
