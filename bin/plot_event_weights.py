#!/usr/bin/env python3
"""MC event-weight distributions per region: the total weight and each of its factors.

For every region, one line per MC group (signals too in signal regions), with
the first --dy sample as the Drell-Yan group:

    <plots>/<Run>/<Year>/<Era>/<dir>/event_weights/<region>.pdf
        log10|w| of the region's nominal weight, w > 0 solid and w < 0 dashed;
        the legend gives N, N_eff and the fraction of negative weights.
    <plots>/.../event_weights/components/<region>/<component>.pdf
        each factor of that weight (generator weight, normalization, pileup,
        lepton SFs, ..., and the product of the corrections), unit-normalized.
    tables/<Run>/<Year>/<Era>/<dir>/event_weights/<region>.txt
        the same numbers as text.

With two or more --dy samples the DY samples are also overlaid on their own,
under event_weights/dy_samples/.

The histograms are written by the analyzer for MC only (event_weights/ in the
ROOT files); files made before that need an analyzer rerun.

Examples:
    python bin/plot_event_weights.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --local-plots
    python bin/plot_event_weights.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --dy lo_ht,lo_inc --local-plots
    python bin/plot_event_weights.py --era RunIII2024Summer24 --dir 20261008_skims_v2 -r resolved_sr --no-components --local-plots
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wrplotter.cli_utils import (add_dy_arg, add_era_args, add_output_args, dy_specs, parse_multi,
                                 select_regions, setup_logging)
from wrplotter.config import default_signals, load_lumi
from wrplotter.event_weights import (component_names, format_summary, has_event_weights,
                                     plot_component, plot_weight_distributions, read_components,
                                     read_distribution)
from wrplotter.inputs import DY_SAMPLE, make_inputs
from wrplotter.paths import plots_base, repo_root, save_figure
from wrplotter.plotting_helpers import SIGNAL_COLORS, signal_label
from wrplotter.sample_groups import load_sample_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    add_era_args(parser)
    add_dy_arg(parser, multiple=True)
    parser.add_argument("--region", "-r", dest="regions", action="append", default=None,
                        help="Region name(s) or shorthand(s). Repeat or comma-separate.")
    add_output_args(parser)
    parser.add_argument("--signal", "-s", dest="signal", action="append", default=None,
                        help="Signal sample(s) shown in signal regions, e.g. signal_WR4000_N2100. "
                             "Default: per era and topology (data/lumi.yaml).")
    parser.add_argument("--no-components", action="store_true",
                        help="Only the total weight, not its factors.")
    args = parser.parse_args()
    args.regions = parse_multi(args.regions)
    args.signal = parse_multi(args.signal)
    return args


def _files(inputs, samples):
    return [f for s in samples for f in inputs.files_for(s)]


def _entries(inputs, region, groups, signals):
    """(name, label, color, files) per MC group, plus signals in signal regions;
    *name* is the plain-text name used in the summary tables."""
    entries = [(g.key, g.tlatex_alias, g.color, _files(inputs, g.samples)) for g in groups]
    if region.is_signal_region:
        for i, name in enumerate(signals):
            entries.append((name, signal_label(name), SIGNAL_COLORS[i % len(SIGNAL_COLORS)],
                            inputs.files_for(name)))
    return [e for e in entries if e[3]]


def _save(fig, path, counters):
    if fig is None:
        return
    try:
        save_figure(fig, path)
        counters["saved"] += 1
    except Exception as exc:  # EOS upload errors must not stop the batch
        logging.error("  failed to save %s: %s", path, exc)
        counters["failed"] += 1
    finally:
        plt.close(fig)


def _plot_set(out_dir, region, entries, *, text, lumi, com, components, counters):
    """Total-weight plot and (optionally) one plot per component.

    Returns ``(totals, comps)`` as ``[(name, WeightDistribution)]`` and
    ``[(name, {key: Hist})]`` for the summary table.
    """
    totals = [(name, label, color, read_distribution(files, region.name))
              for name, label, color, files in entries]
    totals = [t for t in totals if not t[3].empty]
    if not totals:
        logging.warning("  %s: no event-weight histograms, skipped", region.name)
        return [], []
    _save(plot_weight_distributions([t[1:] for t in totals], text=text, lumi=lumi, com=com, year=region.year),
          out_dir / f"{region.name}.pdf", counters)

    comps = []
    if components:
        comps = [(name, label, color, read_components(files, region.name))
                 for name, label, color, files in entries]
        comps = [c for c in comps if c[3]]
        if not comps:
            logging.warning("  %s: no weight-component histograms (analyzer output predates them)",
                            region.name)
        for comp in component_names([k for *_, hists in comps for k in hists]):
            _save(plot_component(comp, [c[1:] for c in comps], text=text, lumi=lumi, com=com,
                                 year=region.year),
                  out_dir / "components" / region.name / f"{comp}.pdf", counters)
    return [(t[0], t[3]) for t in totals], [(c[0], c[3]) for c in comps]


def main():
    args = parse_args()
    setup_logging(args.verbose)

    info = load_lumi(args.era)
    run, year, lumi, com = info["run"], info["year"], info["lumi"], info.get("com", 13.6)
    regions = select_regions(args.era, args.regions)
    try:
        inputs_list = make_inputs(args.era, args.dir, dy_specs(args.dy))
    except (FileNotFoundError, ValueError) as exc:
        logging.error(str(exc))
        sys.exit(2)
    main_inputs = inputs_list[0]

    groups, order = load_sample_groups()
    mc_groups = [groups[k] for k in order if groups[k].kind == "mc"]
    mc_files = [f for g in mc_groups for f in _files(main_inputs, g.samples)]
    if not any(has_event_weights(f) for f in mc_files):
        logging.error("None of the MC files in %s has event_weights/: rerun the analyzer "
                      "(it writes them for MC since the blinding/event-weight update).",
                      ", ".join(str(d) for d in main_inputs.dirs))
        sys.exit(2)

    base = plots_base(run, year, args.era, args.dir or None, local=args.local_plots) / "event_weights"
    tables = repo_root() / "tables" / run / year / args.era
    tables = (tables / args.dir if args.dir else tables) / "event_weights"
    tables.mkdir(parents=True, exist_ok=True)
    logging.info("Era %s, %d region(s), DY sample(s) %s -> %s", args.era, len(regions),
                 ", ".join(i.dy.tag for i in inputs_list), base)

    counters = {"saved": 0, "failed": 0}
    components = not args.no_components
    for region in regions:
        logging.info("Region %s", region.name)
        signals = args.signal or default_signals(info, boosted=region.is_boosted)
        text = f"{region.tlatex_alias}\nDY: {main_inputs.dy.label}"
        totals, comps = _plot_set(base, region, _entries(main_inputs, region, mc_groups, signals),
                                  text=text, lumi=lumi, com=com, components=components,
                                  counters=counters)
        if totals:
            summary = format_summary(f"{region.name} ({args.era}, {args.dir or 'base dir'}, "
                                     f"DY: {main_inputs.dy.tag})", totals, comps)
            (tables / f"{region.name}.txt").write_text(summary)

        if len(inputs_list) > 1:
            dy_entries = [(f"DY {i.dy.tag}", i.dy.label, i.dy.color, i.files_for(DY_SAMPLE))
                          for i in inputs_list]
            _plot_set(base / "dy_samples", region, dy_entries, text=region.tlatex_alias,
                      lumi=lumi, com=com, components=components, counters=counters)

    logging.info("Done: %d saved, %d failed. Plots: %s  Tables: %s",
                 counters["saved"], counters["failed"], base, tables)
    if counters["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
