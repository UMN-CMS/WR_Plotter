#!/usr/bin/env python3
"""Tables of the systematic uncertainties of an analysis set.

For every region, the relative change (%) of each process's yield under every
up/down variation the files carry (data/systematics.yaml), plus the largest
per-bin change, at the binning of the plots (the fit binning for the mass
observables).  By default only the region's mass observable is tabulated
(mass_fourobject resolved, mass_twoobject boosted); -v adds other variables.

    tables/<Run>/<Year>/<Era>/<dir>/systematics_dy_<tag>/systematics_<variable>_<region>.{txt,md,tex}
    tables/<Run>/<Year>/<Era>/<dir>/systematics_dy_<tag>/systematics.{txt,md,csv}   (all regions)

Examples:
    python bin/make_systematics_tables.py --era RunIII2024Summer24 --dir 20261011_systs
    python bin/make_systematics_tables.py --era Run3 --dir 20261011_systs -r resolved_sr --per-bin
    python bin/make_systematics_tables.py --era RunIISummer20UL18 --dir 20261011_systs -s signal_WR6000_N3100
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wrplotter.cli_utils import (add_dy_arg, add_era_args, add_selection_args, dy_specs, parse_multi,
                                 select_regions, select_variables, setup_logging)
from wrplotter.config import (configured_variables, default_signals, index_plot_settings, load_lumi,
                              load_plot_settings, load_systematics)
from wrplotter.inputs import make_inputs
from wrplotter.paths import repo_root
from wrplotter.sample_groups import load_sample_groups
from wrplotter.stack import axis_config
from wrplotter.syst_tables import (build_table, csv_rows, format_csv, format_latex, format_markdown,
                                   format_text, per_bin_text)

MASS_VARIABLE = {"resolved": "mass_fourobject", "boosted": "mass_twoobject"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    add_era_args(parser, required=True)
    add_dy_arg(parser, multiple=False)
    add_selection_args(parser)
    parser.add_argument("--signal", "-s", dest="signal", action="append", default=None,
                        help="Signal sample(s) tabulated in the signal regions, e.g. signal_WR4000_N2100. "
                             "Default: per era and topology (data/lumi.yaml).")
    parser.add_argument("--per-bin", action="store_true",
                        help="Also write the per-bin changes of every process (text only).")
    parser.add_argument("--dry-run", action="store_true", help="List the tables that would be made and exit.")
    args = parser.parse_args()
    args.regions = parse_multi(args.regions)
    args.variables = parse_multi(args.variables)
    args.signal = parse_multi(args.signal)
    return args


def main():
    args = parse_args()
    setup_logging(args.verbose)

    info = load_lumi(args.era)
    run, year, lumi = info["run"], str(info["year"]), float(info["lumi"])
    regions = select_regions(args.era, args.regions)
    region_cfgs, common_vars = index_plot_settings(load_plot_settings(args.era))
    requested = select_variables(args.variables) if args.variables else None
    plan = []
    for region in regions:
        wanted = requested or [v for v in select_variables([MASS_VARIABLE[region.topology]])]
        for v, cfg in configured_variables(region_cfgs, common_vars, region.name, wanted):
            plan.append((region, v, cfg))
    try:
        inputs = make_inputs(args.era, args.dir, dy_specs(args.dy))[0]
    except (FileNotFoundError, ValueError) as exc:
        logging.error(str(exc))
        sys.exit(2)

    if args.dry_run:
        print(f"Dry run - {args.era} ({run} {year}, {lumi:.2f} fb^-1), DY sample {inputs.dy.tag}: "
              f"{len(plan)} table(s)")
        for r, v, _ in plan:
            print(f"  {r.name:30s} {v.name}")
        sys.exit(0)

    groups, order = load_sample_groups()
    groups = [groups[k] for k in reversed(order)]   # DY first, data groups skipped by build_table
    systematics = load_systematics()
    out_dir = repo_root() / "tables" / run / year / args.era / args.dir / f"systematics_dy_{inputs.dy.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    texts, mds, rows = [], [], []
    n_tables = 0
    for region, variable, vcfg in plan:
        signals = []
        if region.is_signal_region:
            signals = args.signal or default_signals(info, boosted=region.is_boosted)
        binning = axis_config(vcfg, variable_rebin=True, density=False).binning
        table = build_table(region, variable.name, inputs=inputs, groups=groups, binning=binning,
                            systematics=systematics, signals=signals)
        if table is None:
            logging.warning("No MC histograms for %s / %s; skipped.", region.name, variable.name)
            continue
        if not table.sources:
            logging.warning("%s / %s: the files carry no systematic variations.", region.name, variable.name)
        n_tables += 1
        stem = f"systematics_{variable.name}_{region.name}"
        text = format_text(table)
        if args.per_bin:
            text += "\n\n" + "\n\n".join(per_bin_text(table, p) for p in table.processes)
        (out_dir / f"{stem}.txt").write_text(text + "\n")
        (out_dir / f"{stem}.md").write_text(format_markdown(table))
        (out_dir / f"{stem}.tex").write_text(format_latex(table, era=args.era, lumi=lumi, year=year) + "\n")
        texts.append(text)
        mds.append(format_markdown(table))
        rows += csv_rows(table, args.era)
        total = table.total
        worst = sorted(((abs(c[0]) + abs(c[1])) / 2, s) for s in table.sources
                       if (c := total.yield_change(s)) is not None)[-3:]
        logging.info("%s / %s: %d sources; largest on the total background: %s", region.name, variable.name,
                     len(table.sources), ", ".join(f"{s} {v:.1f}%" for v, s in reversed(worst)) or "none")

    if not n_tables:
        logging.error("No tables made.")
        sys.exit(1)
    title = f"Systematics: {args.era}, {args.dir or '(no --dir)'}, DY sample {inputs.dy.tag}, {lumi:.2f} fb^-1"
    (out_dir / "systematics.txt").write_text(title + "\n\n" + "\n\n\n".join(texts) + "\n")
    (out_dir / "systematics.md").write_text(f"# {title}\n\n" + "\n".join(mds))
    (out_dir / "systematics.csv").write_text(format_csv(rows))
    logging.info("Wrote %d table(s) (txt, md, tex) plus systematics.{txt,md,csv} to %s", n_tables, out_dir)


if __name__ == "__main__":
    main()
