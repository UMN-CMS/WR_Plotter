#!/usr/bin/env python3
"""Compare Drell-Yan samples, alone and inside the total background.

Needs at least two --dy samples; the first is the reference.  For every region
and variable of the era's plot settings:

    <plots>/<Run>/<Year>/<Era>/<dir>/dy_compare/abs/<region>/<variable>_<region>.pdf
        DYJets of each sample with the ratio to the reference;
    .../dy_compare/shape/<region>/...
        the same, each sample scaled to unit yield;
    .../dy_compare/total/<region>/...
        the total background built with each DY sample (the other backgrounds
        are shared).  Control regions show data with Data/Bkg. per sample;
        signal regions compare the totals to the reference (their data is
        blinded by the analyzer);
    .../dy_compare/total_scaled/<region>/...   (control regions only)
        as total, with each prediction scaled to the data yield in the plotted
        range (factor in the legend), so only the shapes are compared.

A non-default DY sample is read from <dir>_<TAG>_dy, or from DIR with --dy TAG=DIR.

Examples:
    python bin/compare_dy.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --dy lo_ht --dy lo_inc --local-plots
    python bin/compare_dy.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --dy lo_ht,lo_inc -r resolved_dy_cr -v mass_fourobject --local-plots
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
from wrplotter.config import (configured_variables, index_plot_settings, load_lumi,
                              load_plot_settings)
from wrplotter.dy_compare import plot_dy_overlay, plot_total_comparison
from wrplotter.inputs import DY_SAMPLE, make_inputs
from wrplotter.paths import plots_base, save_figure
from wrplotter.sample_groups import load_sample_groups
from wrplotter.stack import HistLoader, axis_config, build_stack, hist_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    add_era_args(parser)
    add_dy_arg(parser, multiple=True)
    add_selection_args(parser)
    add_output_args(parser)
    parser.add_argument("--plot-config", "-c", dest="plot_config", default=None,
                        help="YAML with rebin/xlim/ylim per (region, variable); default: the era's.")
    parser.add_argument("--uniform-rebin", action="store_true",
                        help="Use the uniform rebin factor even where variable-width edges (rebin_variable, "
                             "the Run 2 m(W_R) binning) are set.")
    parser.add_argument("--extra-label", dest="extra_label", default=None,
                        help="Extra line printed under the region label.")
    args = parser.parse_args()
    args.dy = dy_specs(args.dy)
    if not args.dy or len(args.dy) < 2:
        parser.error("give at least two DY samples, e.g. --dy lo_ht --dy lo_inc")
    args.regions = parse_multi(args.regions)
    args.variables = parse_multi(args.variables)
    return args


def save(fig, path: Path) -> bool:
    try:
        save_figure(fig, path)
        return True
    except Exception as exc:  # EOS upload errors must not stop the batch
        logging.error("  failed to save %s: %s", path, exc)
        return False
    finally:
        plt.close(fig)


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
        inputs_list = make_inputs(args.era, args.dir, args.dy)
    except (FileNotFoundError, ValueError) as exc:
        logging.error(str(exc))
        sys.exit(2)

    groups, order = load_sample_groups()
    groups = [groups[k] for k in order]
    base = plots_base(run, year, args.era, args.dir or None, local=args.local_plots) / "dy_compare"
    logging.info("Era %s (%s %s, %.2f fb^-1 @ %.1f TeV); DY samples %s (reference %s); "
                 "%d (region, variable) pair(s) -> %s", args.era, run, year, lumi, com,
                 ", ".join(i.dy.tag for i in inputs_list), inputs_list[0].dy.tag, len(plan), base)

    n_saved = n_skipped = n_failed = 0
    for i, (region, variable, vcfg) in enumerate(plan, 1):
        logging.info("[%d/%d] %s / %s", i, len(plan), region.name, variable.name)
        axes = axis_config(vcfg, variable_rebin=not args.uniform_rebin)
        key = hist_key(region.name, variable.name)
        name = f"{variable.name}_{region.name}.pdf"
        outputs = []

        # DY-only overlays
        dy_hists = [(inp.dy, HistLoader(inp, axes.binning)(DY_SAMPLE, key))
                    for inp in inputs_list]
        missing = [dy.tag for dy, h in dy_hists if h is None]
        if missing:
            logging.debug("  no %s for DY sample(s) %s; overlays skipped", key, missing)
        else:
            for kind, normalize in (("abs", False), ("shape", True)):
                fig = plot_dy_overlay(region, variable, dy_hists, binning=axes.binning,
                                      xlim=axes.xlim, lumi=lumi, com=com, normalize=normalize,
                                      ratio_ylim=axes.ratio_ylim, extra_label=args.extra_label)
                outputs.append((fig, base / kind / region.name / name))

        # Total background with each DY sample (no signals, no systematics)
        stacks = [(inp.dy, build_stack(region, variable.name, inputs=inp, groups=groups,
                                       binning=axes.binning))
                  for inp in inputs_list]
        totals = [(dy, s.total) for dy, s in stacks]
        if any(t is None for _, t in totals):
            logging.debug("  no MC for %s; total comparison skipped", key)
        else:
            ref = stacks[0][1]
            variants = [("total", False)]
            if not region.is_signal_region and ref.data is not None:
                variants.append(("total_scaled", True))
            for kind, scaled in variants:
                fig = plot_total_comparison(region, variable, totals, data=ref.data,
                                            data_blinded=ref.data_blinded, xlim=axes.xlim,
                                            ylim=axes.ylim, lumi=lumi, com=com,
                                            ratio_ylim=axes.ratio_ylim,
                                            extra_label=args.extra_label,
                                            scale_to_data=scaled, binning=axes.binning)
                outputs.append((fig, base / kind / region.name / name))

        if not outputs:
            logging.warning("  no histograms for %s / %s, skipped", region.name, variable.name)
            n_skipped += 1
            continue
        for fig, path in outputs:
            if save(fig, path):
                n_saved += 1
            else:
                n_failed += 1

    logging.info("Done: %d saved, %d skipped, %d failed. Output: %s", n_saved, n_skipped, n_failed, base)
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
