"""Tables of the systematic uncertainties of one region.

For every process of a region (each background group, their sum and the
requested signal points) and every systematic source listed in
``data/systematics.yaml``, the table holds the relative change of the yield
under the up and the down variation, and the largest per-bin change, at the
binning the plots (and the datacards) use.  The variations come from
:mod:`wrplotter.systematics` exactly as for the band of the stack plots: a
source missing from a file leaves that file's nominal in place, composite
sources are envelopes, and ``pdf`` / ``alphas`` are formed from the PDF
members at the table's binning.  A source whose variations equal the nominal
(the analyzer produced it for other regions or samples only) does not affect
the process and is shown as a dash.

Outputs: plain text, Markdown, CMS-style LaTeX and a long-format CSV
(``bin/make_systematics_tables.py``).
"""
from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from hist import Hist

from .histogram_cache import sum_files
from .inputs import Inputs
from .regions import Region
from .sample_groups import SampleGroup
from .stack import Binning, HistLoader, hist_key, hist_yield
from .systematics import sample_variations, total_variations

logger = logging.getLogger(__name__)

TOTAL_KEY = "total_bkg"
NO_EFFECT = "-"
LEGEND = f"{NO_EFFECT}: the source does not affect that process here (no variation, or one equal to the nominal)."

# readable names of the analyzer's sources (lower-cased labels), for the tables
SOURCE_LABELS = {
    "lumi": "Luminosity",
    "pileup": "Pileup",
    "l1prefiring": "L1 prefiring",
    "muonrecosf": "Muon reco SF",
    "muonidsf": "Muon ID SF",
    "muonisosf": "Muon iso SF",
    "muontrigsf": "Muon trigger SF",
    "loosemuonrecosf": "Loose muon reco SF",
    "loosemuonidsf": "Loose muon ID SF",
    "electronrecosf": "Electron reco SF",
    "electronidsf": "Electron ID (HEEP) SF",
    "electrontrigsf": "Electron trigger SF",
    "looseelectronrecosf": "Loose electron reco SF",
    "lsfsf": "LSF SF",
    "jer": "JER",
    "jes": "JES",
    "elescale": "Electron energy scale",
    "elesmear": "Electron energy smearing",
    "muscale": "Muon momentum scale",
    "musmear": "Muon momentum smearing",
    "dyzptqcdscale": "DY Z-pT: QCD scale",
    "dyzptqcdpdf": "DY Z-pT: QCD PDF",
    "dyzptqcdalphas": "DY Z-pT: QCD alpha_s",
    "dyzptew1": "DY Z-pT: EW 1",
    "dyzptew2": "DY Z-pT: EW 2",
    "dyzptew3": "DY Z-pT: EW 3",
    "dyreshapestat": "DY jet-pT reshape: stat.",
    "dyreshapechannel": "DY jet-pT reshape: ee vs mumu",
    "scale": "QCD scales (muR, muF envelope)",
    "psisr": "PS ISR",
    "psfsr": "PS FSR",
    "pdf": "PDF",
    "alphas": "alpha_s",
}


def source_label(source: str) -> str:
    return SOURCE_LABELS.get(source, source)


@dataclass
class ProcessSysts:
    """Nominal and variations of one process in one region, at the table binning."""

    key: str                      # group key, TOTAL_KEY or the signal sample name
    label: str                    # plain-text column label
    tex_label: str                # LaTeX column label
    nominal: Hist
    variations: dict[str, dict[str, Hist]]
    binning: Binning = field(default_factory=Binning)

    @property
    def yield_(self) -> float:
        return hist_yield(self.nominal, self.binning)

    def effective(self, source: str) -> dict[str, Hist] | None:
        """The up/down pair of *source* when it moves this process, else None."""
        dirs = self.variations.get(source)
        if not dirs:
            return None
        nom = self.nominal.values()
        if all(np.array_equal(dirs[d].values(), nom) for d in ("up", "down")):
            return None
        return dirs

    def yield_change(self, source: str) -> tuple[float, float] | None:
        """``(up, down)`` relative change of the yield in percent; None when the source has no effect."""
        dirs = self.effective(source)
        n = self.yield_
        if dirs is None or n == 0:
            return None
        return tuple(100.0 * (hist_yield(dirs[d], self.binning) - n) / n for d in ("up", "down"))

    def max_bin_change(self, source: str) -> float | None:
        """Largest |relative change| (percent) over the bins with a non-zero nominal."""
        dirs = self.effective(source)
        if dirs is None:
            return None
        nom = self.nominal.values()
        mask = nom > 0
        if not mask.any():
            return None
        worst = 0.0
        for d in ("up", "down"):
            rel = np.abs(dirs[d].values()[mask] - nom[mask]) / nom[mask]
            worst = max(worst, float(rel.max()))
        return 100.0 * worst


@dataclass
class SystTable:
    region: Region
    variable: str
    binning: Binning
    processes: list[ProcessSysts]
    sources: list[str]            # sources that move at least one process, in table order

    @property
    def backgrounds(self) -> list[ProcessSysts]:
        return [p for p in self.processes if p.key != TOTAL_KEY and not p.key.startswith("signal_")]

    @property
    def total(self) -> ProcessSysts | None:
        return next((p for p in self.processes if p.key == TOTAL_KEY), None)

    @property
    def signals(self) -> list[ProcessSysts]:
        return [p for p in self.processes if p.key.startswith("signal_")]

    def missing(self, process: ProcessSysts) -> list[str]:
        """Sources of the table that do not move *process*."""
        return [s for s in self.sources if process.effective(s) is None]


def _process_variations(load: HistLoader, samples: Sequence[str], region: Region, variable: str,
                        systematics: Sequence[str]) -> tuple[Hist | None, dict[str, dict[str, Hist]]]:
    """Nominal and variations of a process made of *samples* (summed over sub-eras and samples)."""
    key = hist_key(region.name, variable)
    per_sample = []
    for sample in samples:
        per_file = load.per_file(sample, key)
        h = sum_files(per_file, key) if per_file else None
        if h is None:
            continue
        per_sample.append((h, sample_variations(per_file, region.name, variable, systematics,
                                                load=load.file, rebin=load.rebin)))
    if not per_sample:
        return None, {}
    nominal = None
    for h, _ in per_sample:
        nominal = h if nominal is None else nominal + h
    if len(per_sample) == 1:
        return nominal, per_sample[0][1]
    return nominal, total_variations(per_sample)


def build_table(region: Region, variable: str, *, inputs: Inputs, groups: Sequence[SampleGroup],
                binning: Binning, systematics: Sequence[str], signals: Sequence[str] = ()
                ) -> SystTable | None:
    """The systematics table of one (region, variable), or None without any MC.

    *groups* in the order their columns should appear (data groups are
    skipped); *signals* are sample names such as ``signal_WR4000_N2100``.
    """
    load = HistLoader(inputs, binning)
    processes: list[ProcessSysts] = []
    per_group = []
    for group in groups:
        if group.kind == "data":
            continue
        nominal, variations = _process_variations(load, group.samples, region, variable, systematics)
        if nominal is None:
            logger.warning("%s: no %s histogram for %s; left out of the table", region.name, variable, group.key)
            continue
        processes.append(ProcessSysts(group.key, group.key, group.tlatex_alias, nominal, variations, binning))
        per_group.append((nominal, variations))
    if not processes:
        return None
    total = None
    for nominal, _ in per_group:
        total = nominal if total is None else total + nominal
    processes.append(ProcessSysts(TOTAL_KEY, "Total bkg.", "Total bkg.", total,
                                  total_variations(per_group), binning))
    for sample in signals:
        nominal, variations = _process_variations(load, [sample], region, variable, systematics)
        if nominal is None:
            logger.debug("%s: no %s for signal %s", region.name, variable, sample)
            continue
        point = sample.removeprefix("signal_")
        processes.append(ProcessSysts(sample, point, _signal_tex(point), nominal, variations, binning))
    sources = [s for s in systematics if any(p.effective(s) is not None for p in processes)]
    return SystTable(region, variable, binning, processes, sources)


def _signal_tex(point: str) -> str:
    """``WR4000_N2100`` -> ``(4000, 2100)``."""
    try:
        wr, n = point.split("_")
        return f"({wr.removeprefix('WR')}, {n.removeprefix('N')})"
    except ValueError:
        return point


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _cell(change: tuple[float, float] | None) -> str:
    if change is None:
        return NO_EFFECT
    up, down = change
    return f"{up:+.1f} / {down:+.1f}"


def _max_cell(value: float | None) -> str:
    return NO_EFFECT if value is None else f"{value:.1f}"


def _title(table: SystTable) -> str:
    return (f"{table.region.name} / {table.variable}: relative change of the yield (%) under each "
            f"systematic variation, up / down; second block: largest per-bin |change| (%)")


def format_text(table: SystTable) -> str:
    """Fixed-width table: rows = sources, columns = processes (up / down in percent)."""
    w0 = max(len(source_label(s)) for s in table.sources) + 2 if table.sources else 10
    w = max(16, max(len(p.label) for p in table.processes) + 2)
    lines = [_title(table), LEGEND, ""]
    header = f"{'Source':<{w0}}" + "".join(f"{p.label:>{w}}" for p in table.processes)
    lines += [f"{'Yield':<{w0}}" + "".join(f"{p.yield_:>{w}.2f}" for p in table.processes), header,
              "-" * len(header)]
    for s in table.sources:
        lines.append(f"{source_label(s):<{w0}}" + "".join(f"{_cell(p.yield_change(s)):>{w}}" for p in table.processes))
    lines += ["", f"{'Max per-bin |change| (%)':<{w0}}" + "".join(f"{p.label:>{w}}" for p in table.processes),
              "-" * len(header)]
    for s in table.sources:
        lines.append(f"{source_label(s):<{w0}}" + "".join(f"{_max_cell(p.max_bin_change(s)):>{w}}" for p in table.processes))
    return "\n".join(lines)


def format_markdown(table: SystTable) -> str:
    lines = [f"### {table.region.name} / {table.variable}", "",
             f"Relative change of the yield (%) under each variation, up / down. {LEGEND}", "",
             "| Source | " + " | ".join(p.label for p in table.processes) + " |",
             "|---|" + "---:|" * len(table.processes),
             "| Yield | " + " | ".join(f"{p.yield_:.2f}" for p in table.processes) + " |"]
    for s in table.sources:
        lines.append(f"| {source_label(s)} | " + " | ".join(_cell(p.yield_change(s)) for p in table.processes) + " |")
    lines += ["", "Largest per-bin |change| (%).", "",
              "| Source | " + " | ".join(p.label for p in table.processes) + " |",
              "|---|" + "---:|" * len(table.processes)]
    for s in table.sources:
        lines.append(f"| {source_label(s)} | " + " | ".join(_max_cell(p.max_bin_change(s)) for p in table.processes) + " |")
    return "\n".join(lines) + "\n"


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def _tex_cell(change: tuple[float, float] | None) -> str:
    if change is None:
        return "--"
    up, down = change
    return f"${up:+.1f}\\,/\\,{down:+.1f}$"


def format_latex(table: SystTable, *, era: str, lumi: float, year: str) -> str:
    """CMS-style LaTeX table (``\\topcaption``, ``\\cmsTable``), signal columns after a rule."""
    bkg = table.backgrounds + ([table.total] if table.total else [])
    sig = table.signals
    col_spec = "l | " + " ".join(["c"] * len(bkg)) + (" | " + " ".join(["c"] * len(sig)) if sig else "")
    header = ["Source"] + [p.tex_label for p in bkg] + [p.tex_label for p in sig]
    region = _tex_escape(table.region.name)
    caption = (f"{_tex_escape(era)}: relative change (\\%) of the expected yield in the {region} region "
               f"under each systematic variation, up\\,/\\,down, at the fit binning of "
               f"{_tex_escape(table.variable)}. Signal columns show $(m_{{W_R}},\\,m_{{N}})$ in GeV. "
               f"Normalized to ${lumi:.1f}\\,\\mathrm{{fb}}^{{-1}}$ ({year}). A dash: the source does not "
               "affect that process.")
    lines = [r"\begin{table}[htp]", f"  \\topcaption{{{caption}}}",
             f"  \\label{{tab:syst_{table.region.name}_{year}}}", r"  \centering", r"  \cmsTable{",
             f"    \\begin{{tabular}}{{{col_spec}}}", r"      \hline",
             "      " + " & ".join(header) + r" \\", r"      \hline",
             "      " + " & ".join(["Yield"] + [f"{p.yield_:.1f}" for p in bkg + sig]) + r" \\",
             r"      \hline"]
    for s in table.sources:
        cells = [_tex_escape(source_label(s))] + [_tex_cell(p.yield_change(s)) for p in bkg + sig]
        lines.append("      " + " & ".join(cells) + r" \\")
    lines += [r"      \hline", r"    \end{tabular}", r"  }", r"\end{table}"]
    return "\n".join(lines)


CSV_FIELDS = ("era", "region", "variable", "process", "source", "up_pct", "down_pct", "max_bin_pct", "nominal")


def csv_rows(table: SystTable, era: str) -> list[dict]:
    """Long-format rows, one per (process, source); a source without effect gives empty cells."""
    rows = []
    for p in table.processes:
        for s in table.sources:
            change = p.yield_change(s)
            worst = p.max_bin_change(s)
            rows.append({"era": era, "region": table.region.name, "variable": table.variable,
                         "process": p.label, "source": s,
                         "up_pct": "" if change is None else f"{change[0]:.3f}",
                         "down_pct": "" if change is None else f"{change[1]:.3f}",
                         "max_bin_pct": "" if worst is None else f"{worst:.3f}",
                         "nominal": f"{p.yield_:.4f}"})
    return rows


def format_csv(rows: Sequence[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def per_bin_text(table: SystTable, process: ProcessSysts) -> str:
    """Per-bin relative changes (%) of one process: rows = sources, columns = bins."""
    edges = process.nominal.axes[0].edges
    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]
    w0 = max(len(source_label(s)) for s in table.sources) + 2 if table.sources else 10
    w = max(14, max(len(l) for l in labels) + 2)
    nom = process.nominal.values()
    lines = [f"{table.region.name} / {table.variable} / {process.label}: per-bin change (%), up / down", "",
             f"{'Nominal':<{w0}}" + "".join(f"{v:>{w}.2f}" for v in nom),
             f"{'Source':<{w0}}" + "".join(f"{l:>{w}}" for l in labels)]
    lines.append("-" * len(lines[-1]))
    for s in table.sources:
        dirs = process.effective(s)
        if dirs is None:
            lines.append(f"{source_label(s):<{w0}}" + "".join(f"{NO_EFFECT:>{w}}" for _ in labels))
            continue
        cells = []
        for i in range(len(labels)):
            if nom[i] <= 0:
                cells.append(NO_EFFECT)
                continue
            up = 100.0 * (dirs["up"].values()[i] - nom[i]) / nom[i]
            down = 100.0 * (dirs["down"].values()[i] - nom[i]) / nom[i]
            cells.append(f"{up:+.1f}/{down:+.1f}")
        lines.append(f"{source_label(s):<{w0}}" + "".join(f"{c:>{w}}" for c in cells))
    return "\n".join(lines)
