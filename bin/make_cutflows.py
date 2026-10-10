#!/usr/bin/env python3
"""Cutflow tables and plots for an analysis set, from the analyzer's cutflow histograms.

For every topology (resolved, boosted) and channel (ee, mumu, em):

- yields: {cumulative, onecut} x {weighted, unweighted};
- efficiencies (weighted only): running and relative for cumulative,
  single-cut for onecut.

Tables (LaTeX per variant, all of them as text in cutflows.txt, and a
preview.tex compiled to preview.pdf when pdflatex is available):

    tables/<Run>/<Year>/<Era>/<dir>/cutflows_dy_<tag>/cutflow_<variant>_<topology>_<channel>.tex

Plots:

    <plots>/<Run>/<Year>/<Era>/<dir>/cutflows_dy_<tag>/<topology>_<channel>/<variant>.pdf

The cut sequence is read from the files.  The data column comes from the
channel's trigger stream (data/cutflows.yaml).  Blinding is done by the
analyzer: rows a data cutflow lacks are shown as blinded, and data files made
before analyzer-side blinding give no data column (with a note).

Examples:
    python bin/make_cutflows.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --local-plots
    python bin/make_cutflows.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --dy lo_inc --no-plots
    python bin/make_cutflows.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --topology resolved --channel mumu --signal WR4000_N2100 --local-plots
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wrplotter.cli_utils import (add_dy_arg, add_era_args, add_output_args, dy_specs,
                                 parse_multi, setup_logging)
from wrplotter.config import default_signals, load_lumi
from wrplotter.cutflow import (CHANNELS, EFFICIENCIES, KINDS, TOPOLOGIES, build_table,
                               compile_preview, format_latex, format_text, table_variant,
                               write_preview)
from wrplotter.cutflow_plots import plot_efficiency, plot_yields
from wrplotter.inputs import make_inputs
from wrplotter.paths import plots_base, repo_root, save_figure

# Sections of preview.tex: (variant, heading), in this order.
PREVIEW_SECTIONS = [
    ("cumulative", "Cumulative yields (weighted)"),
    ("cumulative_unweighted", "Cumulative yields (unweighted)"),
    ("onecut", "One-cut yields (weighted)"),
    ("onecut_unweighted", "One-cut yields (unweighted)"),
    ("running_eff", "Running efficiency"),
    ("relative_eff", "Relative efficiency"),
    ("single_eff", "Single-cut efficiency"),
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    add_era_args(parser)
    add_dy_arg(parser, multiple=False)
    add_output_args(parser)
    parser.add_argument("--topology", action="append", choices=TOPOLOGIES, default=None,
                        help="Topology to process (repeatable). Default: both.")
    parser.add_argument("--channel", action="append", choices=CHANNELS, default=None,
                        help="Channel to process (repeatable). Default: all three.")
    parser.add_argument("--signal", "-s", dest="signal", action="append", default=None,
                        help="Signal mass point(s) shown as columns, e.g. WR4000_N2100. Repeat or "
                             "comma-separate. Default: per era and topology (data/lumi.yaml).")
    parser.add_argument("--no-plots", action="store_true", help="Tables only.")
    parser.add_argument("--no-tables", action="store_true", help="Plots only.")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Write preview.tex but do not compile it.")
    args = parser.parse_args()
    if args.no_plots and args.no_tables:
        parser.error("--no-plots and --no-tables together leave nothing to do")
    args.signal = [s.removeprefix("signal_") for s in parse_multi(args.signal) or []]
    return args


def _tex_path(variant: str, table) -> str:
    return f"cutflow_{variant}_{table.topology}_{table.channel}.tex"


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("&", r"\&")


def main():
    args = parse_args()
    setup_logging(args.verbose)

    info = load_lumi(args.era)
    run, year, lumi, com = info["run"], str(info["year"]), float(info["lumi"]), info.get("com", 13.6)
    try:
        inputs = make_inputs(args.era, args.dir, dy_specs(args.dy))[0]
    except (FileNotFoundError, ValueError) as exc:
        logging.error(str(exc))
        sys.exit(2)

    subdir = f"cutflows_dy_{inputs.dy.tag}"
    topologies = args.topology or list(TOPOLOGIES)
    channels = args.channel or list(CHANNELS)

    # (table, efficiency mode or None for yields), in output order
    entries = []
    n_failed = 0
    for topology in topologies:
        signals = args.signal or default_signals(info, boosted=topology == "boosted",
                                                 for_cutflow=True)
        logging.info("%s signal columns: %s", topology, ", ".join(signals) or "none")
        for channel in channels:
            for kind in KINDS:
                for unweighted in (False, True):
                    try:
                        table = build_table(inputs, topology, channel, kind,
                                            unweighted=unweighted, signal_points=signals)
                    except ValueError as exc:
                        logging.error("%s %s %s: %s", topology, channel, kind, exc)
                        n_failed += 1
                        continue
                    if table is None:
                        logging.warning("No MC cutflow for %s/%s/%s%s; skipped.", topology, channel,
                                        kind, " (unweighted)" if unweighted else "")
                        continue
                    entries.append((table, None))
                    if not unweighted:
                        entries += [(table, mode) for mode in EFFICIENCIES[kind]]
    if not entries:
        logging.error("No cutflows found in %s.", ", ".join(str(d) for d in inputs.dirs))
        sys.exit(1)

    notes = list(dict.fromkeys(t.data_note for t, _ in entries if t.data_note))
    for note in notes:
        logging.info("Data: %s", note)

    if not args.no_tables:
        out_dir = repo_root() / "tables" / run / year / args.era / args.dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, list[str]] = {}
        texts = []
        for table, mode in entries:
            variant = table_variant(table.kind, table.unweighted, mode)
            name = _tex_path(variant, table)
            (out_dir / name).write_text(
                format_latex(table, era=args.era, lumi=lumi, year=year, efficiency_mode=mode) + "\n")
            written.setdefault(variant, []).append(name)
            texts.append(format_text(table, efficiency_mode=mode))

        title = f"Cutflows: {args.era}, {args.dir or '(no --dir)'}, DY sample {inputs.dy.tag}"
        (out_dir / "cutflows.txt").write_text(
            f"{title}, {lumi:.2f} fb^-1\n\n" + "\n\n".join(texts) + "\n")
        sections = [(heading, written[variant]) for variant, heading in PREVIEW_SECTIONS
                    if written.get(variant)]
        preview = write_preview(out_dir, sections,
                                title=f"{_tex_escape(title)}, ${lumi:.2f}\\,\\mathrm{{fb}}^{{-1}}$",
                                notes=[f"Data: {_tex_escape(note)}" for note in notes])
        logging.info("Wrote %d LaTeX tables, cutflows.txt and preview.tex to %s",
                     sum(map(len, written.values())), out_dir)
        if not args.no_pdf:
            if shutil.which("pdflatex"):
                n_failed += not compile_preview(preview)
            else:
                logging.info("pdflatex not found; preview.tex not compiled.")

    if not args.no_plots:
        base = plots_base(run, year, args.era, args.dir or None, local=args.local_plots) / subdir
        n_saved = 0
        for table, mode in entries:
            variant = table_variant(table.kind, table.unweighted, mode)
            if mode is None:
                fig = plot_yields(table, era=args.era, lumi=lumi, com=com)
            else:
                fig = plot_efficiency(table, mode, era=args.era, lumi=lumi, com=com)
            outpath = base / f"{table.topology}_{table.channel}" / f"{variant}.pdf"
            try:
                save_figure(fig, outpath)
                n_saved += 1
            except Exception as exc:  # EOS upload errors must not stop the batch
                logging.error("Failed to save %s: %s", outpath, exc)
                n_failed += 1
            finally:
                plt.close(fig)
        logging.info("Saved %d cutflow plot(s) to %s", n_saved, base)

    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
