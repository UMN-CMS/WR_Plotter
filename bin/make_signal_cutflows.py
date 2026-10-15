#!/usr/bin/env python3
"""Signal cutflow tables: yields and efficiencies of the selection per (WR, N) mass point.

Modes (--mode, repeatable):

    fixed-wr         one table per WR mass, one column per N mass
    fixed-n          one table per N mass, one column per WR mass
    wr-compare       one table, one column per WR mass (at its highest N)
    channel-compare  one table per WR mass, ee and mumu columns side by side

The mass points are the WRAnalyzer_signal_WR<m>_N<m>.root files of --dir (a
signal-only directory works too); a year or Run 3 (--era 2024, --era Run3) sums
the files of its eras.  The cut sequence is read from the files and must be the
same for every mass point; the labels come from data/cutflows.yaml.  The first
row counts the skimmed events that pass the event filters; weighted rows use the
analyzer's base event weight, without lepton (and LSF) scale factors, except the
last row, which includes the region's scale factors.

Every table comes as yields plus efficiencies (running and relative; single-cut
with --onecut), or as the one --efficiency asked for.

Without --channel, the fixed-wr and wr-compare tables of ee and mumu are written
as LaTeX, with all of them as text in signal_cutflows.txt and a preview.tex
(compiled to preview.pdf when pdflatex is available):

    tables/<Run>/<Year>/<Era>/<dir>/signal_outputs/signal_cutflows/signal_<table>_<topology>_<channel>_<variant>.tex

With --channel, the fixed-wr tables of that channel are printed instead
(--format and --mode override both defaults).

Examples:
    python bin/make_signal_cutflows.py --era RunIII2024Summer24 --dir 20261008_skims_v2
    python bin/make_signal_cutflows.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --channel mumu --wr-mass 4000
    python bin/make_signal_cutflows.py --era RunIII2024Summer24 --dir 20260406_signals --channel ee --boosted --mode fixed-n
    python bin/make_signal_cutflows.py --era Run3 --dir 20261008_skims_v2 --mode channel-compare --efficiency running
"""

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wrplotter.cli_utils import add_era_args, setup_logging
from wrplotter.config import era_label, load_lumi
from wrplotter.cutflow import (CHANNEL_LATEX, CHANNEL_WORDS, CUTFLOW_SF_ROW, EFFICIENCIES, Cutflow,
                               compile_preview, cut_label, efficiency, format_count, format_percent,
                               latex_count, sf_words, signal_cutflows, table_variant, write_preview)
from wrplotter.inputs import Inputs, make_inputs, mass_point
from wrplotter.paths import SIGNAL_OUTPUTS, repo_root

MODES = ("fixed-wr", "fixed-n", "wr-compare", "channel-compare")
CHANNELS = ("ee", "mumu")

# What one column of a mode is: (text, LaTeX).
COLUMNS = {"fixed-wr": ("N mass", "$m_N$"), "fixed-n": ("WR mass", "$m_{W_R}$"),
           "wr-compare": ("(WR, N) masses", r"$(m_{W_R},\,m_N)$"), "channel-compare": ("N mass", "$m_N$")}
HEADINGS = {"fixed-wr": r"Fixed $m_{W_R}$", "fixed-n": r"Fixed $m_N$",
            "wr-compare": r"$m_{W_R}$ comparison (highest $m_N$ each)",
            "channel-compare": r"$ee$ vs.\ $\mu\mu$ per $m_{W_R}$"}
VARIANTS = {"cumulative": "yields", "onecut": "one-cut yields", "running_eff": "running efficiency",
            "relative_eff": "relative efficiency", "single_eff": "single-cut efficiency"}


@dataclass
class SignalTable:
    mode: str
    mass: int | None      # the fixed WR (fixed-wr, channel-compare) or N (fixed-n) mass
    topology: str
    kind: str             # cumulative or onecut
    groups: list[tuple[str, list[tuple[str, Cutflow]]]]   # (channel, [(column header, cutflow)])

    @property
    def channels(self) -> list[str]:
        return [channel for channel, _ in self.groups]

    @property
    def name(self) -> str:
        what = {"fixed-n": f"N{self.mass}", "wr-compare": "wr_compare"}.get(self.mode, f"WR{self.mass}")
        return f"signal_{what}_{self.topology}_{'_vs_'.join(self.channels)}"

    def subject(self, latex: bool) -> str:
        if self.mode == "wr-compare":
            return r"the highest $m_N$ of each $m_{W_R}$" if latex else "highest N per WR mass"
        if self.mode == "fixed-n":
            return f"$m_N = {self.mass}$ GeV" if latex else f"N = {self.mass} GeV"
        return f"$m_{{W_R}} = {self.mass}$ GeV" if latex else f"WR = {self.mass} GeV"

    def row_labels(self, latex: bool) -> list[str]:
        """One label per cut; side-by-side channels whose cuts differ show both."""
        label = (lambda cut: cut_label(cut, "table", self.topology)) if latex else str
        sequences = [cols[0][1].cuts for _, cols in self.groups]
        return [" / ".join(dict.fromkeys(map(label, cuts))) for cuts in zip(*sequences)]


def layout(points: list[str], mode: str) -> list[tuple[int | None, list[str]]]:
    """(fixed mass, column mass points) of each table of *mode*, by mass."""
    masses = {p: mass_point(p) for p in points}
    by = lambda index: [(m, sorted((p for p, wn in masses.items() if wn[index] == m), key=masses.get))
                        for m in sorted({wn[index] for wn in masses.values()})]
    if mode == "fixed-n":
        return by(1)
    if mode == "wr-compare":
        return [(None, [cols[-1] for _, cols in by(0)])] if points else []
    return by(0)


def column_header(point: str, mode: str) -> str:
    wr, n = mass_point(point)
    return {"fixed-n": str(wr), "wr-compare": f"({wr},{n})"}.get(mode, str(n))


def build_tables(inputs: Inputs, points: list[str], *, mode: str, topology: str, kind: str,
                 channels: tuple[str, ...]) -> list[SignalTable]:
    """The tables of *mode* for *channels* (several only for channel-compare).

    Raises ValueError when the mass points do not share one cut sequence, or
    when side-by-side channels have sequences of different lengths.
    """
    flows = {ch: signal_cutflows(inputs, points, topology, ch, kind) for ch in channels}
    lengths = {ch: len(next(iter(f.values())).cuts) for ch, f in flows.items() if f}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Cut sequences of different lengths cannot be shown side by side: {lengths}")
    common = [p for p in points if all(p in f for f in flows.values())]
    return [SignalTable(mode, mass, topology, kind,
                        [(ch, [(column_header(p, mode), flows[ch][p]) for p in cols]) for ch in channels])
            for mass, cols in layout(common, mode)]


def _cells(cf: Cutflow, eff: str | None, latex: bool) -> list[str]:
    if eff:
        return [format_percent(x) for x in efficiency(cf.values, eff)]
    return [latex_count(x) if latex else format_count(x) for x in cf.values]


def format_text(table: SignalTable, eff: str | None, era: str) -> str:
    many = len(table.groups) > 1
    columns = [(f"{ch} {header}" if many else header, _cells(cf, eff, latex=False))
               for ch, cols in table.groups for header, cf in cols]
    headers = ["Cut"] + [header for header, _ in columns]
    rows = [[label] + [cells[i] for _, cells in columns]
            for i, label in enumerate(table.row_labels(latex=False))]
    widths = [max(len(r[j]) for r in [headers, *rows]) for j in range(len(headers))]
    line = lambda r: "  ".join(c.rjust(w) if j else c.ljust(w) for j, (c, w) in enumerate(zip(r, widths)))
    what = VARIANTS[table_variant(table.kind, False, eff)] + (" [%]" if eff else "")
    title = (f"{era_label(era)} {table.topology} {'+'.join(table.channels)}, {table.subject(latex=False)}: "
             f"{what}; columns: {COLUMNS[table.mode][0]} [GeV]")
    return "\n".join([title, line(headers), "  ".join("-" * w for w in widths), *map(line, rows)])


def format_latex(table: SignalTable, eff: str | None, *, era: str, lumi: float, year: str) -> str:
    """CMS-style LaTeX table; side-by-side channels are split by a vertical rule."""
    if eff == "running":
        what = f"{'single-cut' if table.kind == 'onecut' else 'running'} efficiencies (\\%) relative to the first row"
    elif eff == "relative":
        what = "efficiencies (\\%) relative to the previous row"
    else:
        what = "expected signal yields"
    rows = ("each row applies that single cut alone" if table.kind == "onecut"
            else "each row applies that cut and all preceding cuts")
    words = " and ".join(CHANNEL_WORDS[ch] for ch in table.channels)
    caption = (f"{era_label(era)}: {what} for {table.subject(latex=True)}, {table.topology} selection, {words} "
               f"channel{'s' if len(table.groups) > 1 else ''}; {rows}. Columns show "
               f"{COLUMNS[table.mode][1]} in GeV. Normalized to ${lumi:.1f}\\,\\mathrm{{fb}}^{{-1}}$ "
               f"({year}). The first row counts the events of the skims (two leptons, and two AK4 jets or one "
               f"AK8 jet) that pass the event filters. Weighted simulation uses the base event weight, without "
               f"{sf_words(table.topology)}"
               + (f", except the last row, which includes the region's {sf_words(table.topology)}."
                  if any(CUTFLOW_SF_ROW in cf.cuts for _, cols in table.groups for _, cf in cols) else "."))
    columns = [_cells(cf, eff, latex=True) for _, cols in table.groups for _, cf in cols]
    col_spec = "l | " + " | ".join(" ".join("c" * len(cols)) for _, cols in table.groups)
    lines = [r"\begin{table}[htp]", f"  \\topcaption{{{caption}}}",
             f"  \\label{{tab:{table.name}_{table_variant(table.kind, False, eff)}_{year}}}",
             r"  \centering", r"  \cmsTable{", f"    \\begin{{tabular}}{{{col_spec}}}", r"      \hline"]
    if len(table.groups) > 1:
        lines.append("      & " + " & ".join(
            f"\\multicolumn{{{len(cols)}}}{{{'|c' if i else 'c'}}}{{{CHANNEL_LATEX[ch]}}}"
            for i, (ch, cols) in enumerate(table.groups)) + r" \\")
    lines.append("      " + " & ".join(["Selection"] + [h for _, cols in table.groups for h, _ in cols]) + r" \\")
    lines.append(r"      \hline")
    for i, label in enumerate(table.row_labels(latex=True)):
        lines.append("      " + " & ".join([label] + [cells[i] for cells in columns]) + r" \\")
    lines += [r"      \hline", r"    \end{tabular}", r"  }", r"\end{table}"]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    add_era_args(parser)
    parser.add_argument("--channel", action="append", choices=CHANNELS, default=None,
                        help="Channel (repeatable). Default: ee and mumu, written as LaTeX.")
    parser.add_argument("--mode", action="append", choices=MODES, default=None,
                        help="Table layout (repeatable). Default: fixed-wr, plus wr-compare "
                             "without --channel.")
    parser.add_argument("--wr-mass", type=int, nargs="*", default=None,
                        help="WR masses to include (default: all found).")
    parser.add_argument("--boosted", action="store_true", help="Boosted instead of resolved cutflows.")
    parser.add_argument("--onecut", action="store_true", help="Each cut applied alone (not N-1).")
    parser.add_argument("--efficiency", choices=["running", "relative"], default=None,
                        help="Only this efficiency (default: yields and every efficiency).")
    parser.add_argument("--format", choices=["terminal", "latex"], default=None,
                        help="Default: latex without --channel, terminal with it.")
    parser.add_argument("-o", "--output-dir", default=None, help="Override the LaTeX output directory.")
    parser.add_argument("--no-pdf", action="store_true", help="Write preview.tex but do not compile it.")
    args = parser.parse_args()
    if args.onecut and args.efficiency == "relative":
        parser.error("--efficiency relative needs cumulative cutflows (drop --onecut)")
    return args


def main():
    args = parse_args()
    setup_logging(args.verbose)

    info = load_lumi(args.era)
    run, year, lumi = info["run"], str(info["year"]), float(info["lumi"])
    try:
        inputs = make_inputs(args.era, args.dir, with_dy=False)[0]
    except (FileNotFoundError, ValueError) as exc:
        logging.error(str(exc))
        sys.exit(2)

    points = [p for p in inputs.signal_points() if not args.wr_mass or mass_point(p)[0] in args.wr_mass]
    if not points:
        logging.error("No signal files%s in %s.", f" for WR masses {args.wr_mass}" if args.wr_mass else "",
                      ", ".join(map(str, inputs.dirs)))
        sys.exit(1)
    logging.info("%d signal mass points, WR masses %s", len(points),
                 ", ".join(map(str, sorted({mass_point(p)[0] for p in points}))))

    batch = args.channel is None
    channels = args.channel or list(CHANNELS)
    modes = args.mode or (["fixed-wr", "wr-compare"] if batch else ["fixed-wr"])
    fmt = args.format or ("latex" if batch else "terminal")
    topology = "boosted" if args.boosted else "resolved"
    kind = "onecut" if args.onecut else "cumulative"
    variants = [args.efficiency] if args.efficiency else [None, *EFFICIENCIES[kind]]

    tables, n_failed = [], 0
    for mode in modes:
        for chs in [CHANNELS] if mode == "channel-compare" else [(ch,) for ch in channels]:
            try:
                tables += build_tables(inputs, points, mode=mode, topology=topology, kind=kind, channels=chs)
            except ValueError as exc:
                logging.error("%s %s %s: %s", mode, topology, "+".join(chs), exc)
                n_failed += 1
    if not tables:
        logging.error("No signal cutflow tables made.")
        sys.exit(1)

    if fmt == "terminal":
        print("\n\n".join(format_text(t, eff, args.era) for t in tables for eff in variants))
    else:
        out_dir = (Path(args.output_dir) if args.output_dir else
                   repo_root() / "tables" / run / year / args.era / args.dir / SIGNAL_OUTPUTS / "signal_cutflows")
        out_dir.mkdir(parents=True, exist_ok=True)
        sections: dict[tuple[str, str], list[str]] = {}
        for table in tables:
            for eff in variants:
                variant = table_variant(kind, False, eff)
                name = f"{table.name}_{variant}.tex"
                (out_dir / name).write_text(format_latex(table, eff, era=args.era, lumi=lumi, year=year) + "\n")
                sections.setdefault((table.mode, variant), []).append(name)
        (out_dir / "signal_cutflows.txt").write_text(
            "\n\n".join(format_text(t, eff, args.era) for t in tables for eff in variants) + "\n")
        title = f"Signal cutflows: {args.era}, {args.dir or '(no --dir)'}, {topology}".replace("_", r"\_")
        preview = write_preview(out_dir, [(f"{HEADINGS[mode]}: {VARIANTS[variant]}", files)
                                          for (mode, variant), files in sections.items()],
                                title=f"{title}, ${lumi:.2f}\\,\\mathrm{{fb}}^{{-1}}$")
        logging.info("Wrote %d LaTeX tables, signal_cutflows.txt and preview.tex to %s",
                     sum(map(len, sections.values())), out_dir)
        if not args.no_pdf:
            if shutil.which("pdflatex"):
                n_failed += not compile_preview(preview)
            else:
                logging.info("pdflatex not found; preview.tex not compiled.")

    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
