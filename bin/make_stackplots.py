#!/usr/bin/env python3
"""Stacked MC + data plots for every region and variable of an analysis set.

One full set of plots is made per Drell-Yan sample (``--dy``, default: the
analyzer's default sample), each in its own folder:

    <plots>/<Run>/<Year>/<Era>/<dir>/stack_dy_<tag>/<region>/<variable>_<region>.pdf

Every variable is drawn in uniform bins (data/plot_settings).  --run2-binning draws
the m(W_R) observables in the Run 2 fit bins instead, into stack_dy_<tag>_run2_binning/.

Signal regions show the background stack with signal overlays.  Their data
is blinded by the analyzer; it appears only if the data files were made with
the analyzer's --unblind (Run 2 only).

Examples:
    python bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --local-plots
    python bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --dy lo_ht,lo_inc --local-plots
    python bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261008_skims_v2 -r resolved_dy_cr -v mass_fourobject --local-plots
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wrplotter.cli_utils import (add_dy_arg, add_era_args, add_output_args, add_selection_args,
                                 dy_specs, parse_multi, select_regions, select_variables,
                                 setup_logging)
from wrplotter.config import (configured_variables, default_signals, index_plot_settings,
                              list_eras, load_lumi, load_plot_settings,
                              load_region_shorthands, load_systematics)
from wrplotter.inputs import make_inputs
from wrplotter.paths import plots_base, save_figure
from wrplotter.plotting_helpers import plot_stack
from wrplotter.regions import regions_by_name
from wrplotter.sample_groups import load_sample_groups
from wrplotter.stack import axis_config, build_stack
from wrplotter.variables import build_variables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    add_era_args(parser, required=False)
    add_dy_arg(parser, multiple=True)
    add_selection_args(parser)
    add_output_args(parser)
    parser.add_argument("--plot-config", "-c", dest="plot_config", default=None,
                        help="YAML with rebin/xlim/ylim per (region, variable); default: the era's.")
    parser.add_argument("--signal", "-s", dest="signal", action="append", default=None,
                        help="Signal sample(s) overlaid in signal regions, e.g. signal_WR4000_N2100. "
                             "Default: per era and topology (data/lumi.yaml).")
    parser.add_argument("--run2-binning", action="store_true",
                        help="Draw the m(W_R) observables in the Run 2 fit bins (rebin_variable) instead of "
                             "uniform bins; plots go to stack_dy_<tag>_run2_binning/.")
    parser.add_argument("--per-gev", action="store_true",
                        help="With --run2-binning, show events per GeV instead of per bin "
                             "(Run 2 drew events per bin).")
    parser.add_argument("--no-syst", action="store_true",
                        help="Stat-only band even if the files carry systematic variations.")
    parser.add_argument("--extra-label", dest="extra_label", default=None,
                        help="Extra line printed under the region label.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the plots that would be made and exit.")
    parser.add_argument("--list-eras", action="store_true", help="List eras and exit.")
    parser.add_argument("--list-regions", action="store_true", help="List regions and exit.")
    parser.add_argument("--list-variables", action="store_true", help="List variables and exit.")
    args = parser.parse_args()

    if args.list_eras:
        print("Available eras:\n" + "\n".join(f"  - {e}" for e in list_eras()))
        sys.exit(0)
    if args.list_variables:
        print("Available variables:\n"
              + "\n".join(f"  - {v}" for v in sorted(v.name for v in build_variables())))
        sys.exit(0)
    if not args.era:
        parser.error("--era is required (unless using --list-eras or --list-variables)")
    if args.list_regions:
        print(f"Regions for era '{args.era}' (name: data stream):")
        for name, variants in sorted(regions_by_name(args.era).items()):
            print(f"  - {name:30s} {', '.join(r.primary_dataset for r in variants)}")
        print("\nShorthand aliases:")
        for short, names in load_region_shorthands().items():
            print(f"  - {short:30s} -> {', '.join(names)}")
        sys.exit(0)

    args.regions = parse_multi(args.regions)
    args.variables = parse_multi(args.variables)
    args.signal = parse_multi(args.signal)
    return args


def main():
    args = parse_args()
    setup_logging(args.verbose)

    info = load_lumi(args.era)
    run, year, lumi, com = info["run"], info["year"], info["lumi"], info.get("com", 13.6)
    regions = select_regions(args.era, args.regions)
    variables = select_variables(args.variables)
    region_cfgs, common_vars = index_plot_settings(load_plot_settings(args.plot_config or args.era))
    plan = [(r, v, cfg) for r in regions
            for v, cfg in configured_variables(region_cfgs, common_vars, r.name, variables)]

    try:
        inputs_list = make_inputs(args.era, args.dir, dy_specs(args.dy))
    except (FileNotFoundError, ValueError) as exc:
        logging.error(str(exc))
        sys.exit(2)

    if args.dry_run:
        print(f"Dry run - {args.era} ({run} {year}, {lumi:.2f} fb^-1 @ {com:.1f} TeV)")
        print(f"{len(plan)} plot(s) per DY sample ({', '.join(i.dy.tag for i in inputs_list)}):")
        for r, v, _ in plan:
            print(f"  {r.name:30s} {v.name}")
        sys.exit(0)

    groups, order = load_sample_groups()
    groups = [groups[k] for k in order]
    systematics = [] if args.no_syst else load_systematics()
    base = plots_base(run, year, args.era, args.dir or None, local=args.local_plots)
    logging.info("Era %s (%s %s, %.2f fb^-1 @ %.1f TeV); %d plot(s) x %d DY sample(s) -> %s",
                 args.era, run, year, lumi, com, len(plan), len(inputs_list), base)

    n_saved = n_skipped = n_failed = 0
    for inputs in inputs_list:
        out_dir = base / (f"stack_dy_{inputs.dy.tag}" + ("_run2_binning" if args.run2_binning else ""))
        for i, (region, variable, vcfg) in enumerate(plan, 1):
            logging.info("[DY %s %d/%d] %s / %s", inputs.dy.tag, i, len(plan), region.name, variable.name)
            axes = axis_config(vcfg, variable_rebin=args.run2_binning, density=args.per_gev)
            signals = []
            if region.is_signal_region:
                signals = args.signal or default_signals(info, boosted=region.is_boosted)
            stack = build_stack(region, variable.name, inputs=inputs, groups=groups,
                                binning=axes.binning, signals=signals,
                                systematics=systematics)
            if stack.empty:
                logging.warning("  no histograms for %s / %s, skipped", region.name, variable.name)
                n_skipped += 1
                continue

            # The DY sample is named only where DY matters: the flavor CRs hold next to none.
            # (a long sample name, e.g. "LO HT-binned, Z + jet pT reweight", breaks after its comma)
            lines = [] if "flavor_cr" in region.name else [f"DY: {inputs.dy.label}".replace(", ", ",\n")]
            label = "\n".join(lines + ([args.extra_label] if args.extra_label else [])) or None
            fig = plot_stack(
                region, variable,
                stack_list=[h for _, h in stack.mc],
                stack_colors=[g.color for g, _ in stack.mc],
                stack_labels=[g.tlatex_alias for g, _ in stack.mc],
                data_hist=[stack.data] if stack.data is not None else [],
                xlim=axes.xlim, ylim=axes.ylim, ratio_ylim=axes.ratio_ylim,
                lumi=lumi, com=com,
                syst_variations=stack.syst,
                show_data=not stack.data_blinded,
                signal_hists=stack.signals,
                extra_label=label,
                per_unit=axes.binning.per_gev,
            )
            outpath = out_dir / region.name / f"{variable.name}_{region.name}.pdf"
            try:
                save_figure(fig, outpath)
                n_saved += 1
            except Exception as exc:  # EOS upload errors must not stop the batch
                logging.error("  failed to save %s: %s", outpath, exc)
                n_failed += 1
            finally:
                plt.close(fig)

    logging.info("Done: %d saved, %d skipped, %d failed. Output: %s", n_saved, n_skipped, n_failed, base)
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
