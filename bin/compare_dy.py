#!/usr/bin/env python3
"""Compare Drell-Yan samples, alone and inside the total background.

The samples are the named comparisons of data/dy_comparisons.yaml, each into
its own folder (all/, lo/, nlo/, ...; the first sample is the reference): every
one by default, the ones --comparison names, or two or more --dy samples, the
first the reference, straight into dy_compare/.  For each (region, variable)
pair of the plot set:

    <plots>/<Run>/<Year>/<Era>/<dir>/dy_compare/[<comparison>/]abs/<region>/<variable>_<region>.pdf
        DYJets of each sample with the ratio to the reference;
    .../shape/<region>/...
        the same, each sample scaled to unit yield;
    .../stat/<region>/...
        the statistical power of each sample per bin: its relative statistical
        uncertainty, 1/sqrt(N_eff), and its effective number of simulated events
        N_eff = (sum w)^2 / sum w^2;
    .../total/<region>/...
        the total background built with each DY sample (the other backgrounds
        are shared).  Control regions show data with Data/Bkg. per sample;
        signal regions compare the totals to the reference (their data is
        blinded by the analyzer);
    .../total_scaled/<region>/...   (control regions only)
        as total, with each prediction scaled to the data yield in the plotted
        range (factor in the legend), so only the shapes are compared.

--plots core (the default) draws the pairs of the 'core' plot set of
data/dy_comparisons.yaml (the DY CRs and SRs, the fit observables and the
variables the DY corrections act on); --plots all every configured pair.
-r / -v restrict either further.

Every variable is drawn in uniform bins (the plot settings' ``rebin``).  With
--run2-binning the m(W_R) observables use the Run 2 fit bins instead
(``rebin_variable``), and the plots go to dy_compare_run2_binning/.

Every DY sample is a file in --dir (data/dy_samples.yaml): the default, lo_ht,
is WRAnalyzer_DYJets.root, the LO HT-binned DY without the Z-pT reweight and the
jet-pT reshape (the analyzer default since 2026-10-12); any other tag is
WRAnalyzer_DYJets_<TAG>.root, e.g. lo_ht_zpt and lo_ht_zpt_reshape, the same
sample with --dy-reweight zpt or zpt+reshape.  --dy TAG=DIR reads the tag's file
from DIR (with --comparison, --dy TAG=DIR only says where a sample is).  Sets
made before 2026-10-12 used one directory per variant, <dir>_<TAG>_dy, which is
still read when the file is not in --dir, and hold the reweighted DY in <dir>
itself: read it with --dy lo_ht_zpt_reshape=<dir>.  A file whose reweighting
(read from its event-weight histograms) is not the one the tag stands for is
refused.  A comparison with a sample that cannot be found is skipped and
reported; the others are still drawn.

Examples:
    python bin/compare_dy.py --era RunIII2024Summer24 --dir 20261013_skims_v2 --local-plots
    python bin/compare_dy.py --era RunIII2024Summer24 --dir 20261013_skims_v2 --comparison all --local-plots
    python bin/compare_dy.py --era RunIII2024Summer24 --dir 20261012_skims_v2 --comparison lo,lo_ht_vs_nlo --plots all --local-plots
    python bin/compare_dy.py --era RunIII2024Summer24 --dir 20261011_skims_v2 --comparison lo --dy lo_ht=20261008_skims_v2,lo_ht_zpt_reshape=20261011_skims_v2 --local-plots
    python bin/compare_dy.py --era RunIII2024Summer24 --dir 20261012_skims_v2 --dy lo_ht,lo_inc,nlo_inc -r resolved_dy_cr -v mass_fourobject --local-plots
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
from wrplotter.config import (configured_variables, index_plot_settings, load_dy_comparisons,
                              load_lumi, load_plot_settings)
from wrplotter.dy_compare import plot_dy_overlay, plot_dy_stat, plot_total_comparison
from wrplotter.inputs import DY_SAMPLE, make_inputs
from wrplotter.paths import plots_base, save_figure
from wrplotter.regions import expand_region_requests
from wrplotter.sample_groups import load_sample_groups
from wrplotter.stack import HistLoader, axis_config, build_stack, hist_key


def parse_args() -> argparse.Namespace:
    cfg = load_dy_comparisons()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    add_era_args(parser)
    add_dy_arg(parser, multiple=True)
    add_selection_args(parser)
    add_output_args(parser)
    parser.add_argument("--comparison", action="append", default=None,
                        help="Named comparison(s) of data/dy_comparisons.yaml, repeat or comma-separate: "
                             f"{', '.join(cfg['comparisons'])}. Default (no --dy either): every one.")
    parser.add_argument("--plots", default="core", choices=[*cfg["plot_sets"], "all"],
                        help="Plot set of data/dy_comparisons.yaml, or 'all' (every configured pair). "
                             "Default: core.")
    parser.add_argument("--plot-config", "-c", dest="plot_config", default=None,
                        help="YAML with rebin/xlim/ylim per (region, variable); default: the era's.")
    parser.add_argument("--run2-binning", action="store_true",
                        help="Draw the m(W_R) observables in the Run 2 fit bins (rebin_variable) instead of "
                             "uniform bins; plots go to dy_compare_run2_binning/.")
    parser.add_argument("--extra-label", dest="extra_label", default=None,
                        help="Extra line printed under the region label.")
    args = parser.parse_args()
    args.dy = dy_specs(args.dy) or []
    args.comparison = parse_multi(args.comparison)
    if not args.comparison and not args.dy:
        args.comparison = list(cfg["comparisons"])      # every named comparison
    if args.comparison:
        unknown = [c for c in args.comparison if c not in cfg["comparisons"]]
        if unknown:
            parser.error(f"unknown comparison(s) {unknown}; data/dy_comparisons.yaml has "
                         f"{', '.join(cfg['comparisons'])}")
        bare = [s for s in args.dy if "=" not in s]
        if bare:
            parser.error(f"with --comparison, --dy only takes TAG=DIR (where a sample is); got {bare}")
    elif len(args.dy) < 2:
        parser.error("give at least two DY samples, e.g. --dy lo_ht --dy lo_inc, or --comparison NAME "
                     "(or neither: every named comparison)")
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


def plot_pairs(era: str, plot_set: str) -> set[tuple[str, str]] | None:
    """The (region, variable) names of *plot_set*; None for 'all'."""
    if plot_set == "all":
        return None
    pairs = set()
    for regions, variables in load_dy_comparisons()["plot_sets"][plot_set].items():
        for region in expand_region_requests(era, [regions]):
            pairs.update((region.name, v) for v in variables)
    return pairs


def comparisons_to_draw(args) -> list[tuple[str, list[str]]]:
    """[(output subfolder, DY specs with the reference first)]: one per named comparison,
    or the --dy list in dy_compare/ itself."""
    if not args.comparison:
        return [("", args.dy)]
    where = {s.partition("=")[0].strip(): s for s in args.dy}
    named = load_dy_comparisons()["comparisons"]
    return [(name, [where.get(tag, tag) for tag in named[name]]) for name in args.comparison]


def draw(name, inputs_list, plan, base, args, lumi, com, groups) -> tuple[int, int, int]:
    """Every plot of one comparison; returns (saved, skipped, failed)."""
    logging.info("%s: DY samples %s (reference %s) -> %s", name or "comparison",
                 ", ".join(i.dy.tag for i in inputs_list), inputs_list[0].dy.tag, base)
    n_saved = n_skipped = n_failed = 0
    for i, (region, variable, vcfg) in enumerate(plan, 1):
        logging.info("[%d/%d] %s / %s", i, len(plan), region.name, variable.name)
        axes = axis_config(vcfg, variable_rebin=args.run2_binning)
        key = hist_key(region.name, variable.name)
        file_name = f"{variable.name}_{region.name}.pdf"
        outputs = []

        # DY-only overlays
        dy_hists = [(inp.dy, HistLoader(inp, axes.binning)(DY_SAMPLE, key)) for inp in inputs_list]
        missing = [dy.tag for dy, h in dy_hists if h is None]
        if missing:
            logging.debug("  no %s for DY sample(s) %s; overlays skipped", key, missing)
        else:
            for kind, normalize in (("abs", False), ("shape", True)):
                fig = plot_dy_overlay(region, variable, dy_hists, binning=axes.binning,
                                      xlim=axes.xlim, lumi=lumi, com=com, normalize=normalize,
                                      ratio_ylim=axes.ratio_ylim, extra_label=args.extra_label)
                outputs.append((fig, base / kind / region.name / file_name))
            fig = plot_dy_stat(region, variable, dy_hists, xlim=axes.xlim, lumi=lumi, com=com,
                               extra_label=args.extra_label)
            outputs.append((fig, base / "stat" / region.name / file_name))

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
                outputs.append((fig, base / kind / region.name / file_name))

        if not outputs:
            logging.warning("  no histograms for %s / %s, skipped", region.name, variable.name)
            n_skipped += 1
            continue
        for fig, path in outputs:
            if save(fig, path):
                n_saved += 1
            else:
                n_failed += 1
    return n_saved, n_skipped, n_failed


def main():
    args = parse_args()
    setup_logging(args.verbose)

    info = load_lumi(args.era)
    run, year, lumi, com = info["run"], info["year"], info["lumi"], info.get("com", 13.6)
    regions = select_regions(args.era, args.regions)
    variables = select_variables(args.variables)
    region_cfgs, common_vars = index_plot_settings(load_plot_settings(args.plot_config or args.era))
    wanted = plot_pairs(args.era, args.plots)
    plan = [(r, v, cfg) for r in regions
            for v, cfg in configured_variables(region_cfgs, common_vars, r.name, variables)
            if wanted is None or (r.name, v.name) in wanted]
    if wanted is not None:
        drawn = {(r.name, v.name) for r, v, _ in plan}
        absent = sorted(p for p in wanted if p not in drawn
                        and (not args.regions or any(r.name == p[0] for r in regions))
                        and (not args.variables or p[1] in args.variables))
        for region_name, var_name in absent:
            logging.warning("plot set '%s': %s / %s is not in the plot settings, skipped",
                            args.plots, region_name, var_name)
    if not plan:
        logging.error("Nothing to draw: no (region, variable) pair of the plot settings matches.")
        sys.exit(2)

    groups, order = load_sample_groups()
    groups = [groups[k] for k in order]
    top = (plots_base(run, year, args.era, args.dir or None, local=args.local_plots)
           / ("dy_compare_run2_binning" if args.run2_binning else "dy_compare"))
    logging.info("Era %s (%s %s, %.2f fb^-1 @ %.1f TeV); plot set %s, %d (region, variable) pair(s)",
                 args.era, run, year, lumi, com, args.plots, len(plan))

    inputs_by_spec, refused = {}, {}
    totals = [0, 0, 0]
    skipped_comparisons = []
    for name, specs in comparisons_to_draw(args):
        for spec in specs:
            if spec in inputs_by_spec or spec in refused:
                continue
            try:
                (inputs_by_spec[spec],) = make_inputs(args.era, args.dir, [spec])
            except (FileNotFoundError, ValueError) as exc:
                refused[spec] = str(exc)
        bad = [s for s in specs if s in refused]
        if bad:
            for s in bad:
                logging.error("%s: DY sample %s: %s", name or "comparison", s, refused[s])
            logging.error("%s skipped.", name or "Comparison")
            skipped_comparisons.append(name or "comparison")
            continue
        counts = draw(name, [inputs_by_spec[s] for s in specs], plan, top / name if name else top,
                      args, lumi, com, groups)
        totals = [a + b for a, b in zip(totals, counts)]

    logging.info("Done: %d saved, %d skipped, %d failed%s. Output: %s", *totals,
                 f"; comparisons skipped: {', '.join(skipped_comparisons)}" if skipped_comparisons else "",
                 top)
    if totals[2] or skipped_comparisons:
        sys.exit(1)


if __name__ == "__main__":
    main()
