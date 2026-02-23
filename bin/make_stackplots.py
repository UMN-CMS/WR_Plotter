#!/usr/bin/env python3

# ── Standard library ────────────────────────────────────────────────────────────
import sys
import logging
from pathlib import Path
import argparse

# ── Third-party ────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

# ── Local imports (after we add repo to sys.path below) ────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wrplotter.plotting_helpers import plot_stack
from wrplotter.paths import repo_root, output_dir, save_figure, input_dirs_for_era
from wrplotter.histogram_cache import load_and_rebin
from wrplotter.regions import regions_for_era, expand_region_requests
from wrplotter.variables import build_variables
from wrplotter.sample_groups import load_sample_groups
from wrplotter.config import (list_eras, load_lumi, load_plot_settings, load_kfactors,
                               get_kfactor, index_plot_settings, configured_variables,
                               load_systematics, default_signals, load_region_shorthands)
from wrplotter.cli_utils import parse_multi, setup_logging, add_era_args


def collect_syst_variations(sample, hist_obj, region_name, variable_name,
                            rebin, input_dirs, input_lumis, era, syst_hists,
                            scales, systematics):
    """Load systematic up/down variations for a single MC sample.

    Populates the nested syst_hists dict in-place. Uses the nominal
    histogram as fallback so samples without a given systematic
    contribute delta=0 rather than biasing the envelope.
    """
    for syst in systematics:
        for direction in ["up", "down"]:
            syst_hist_key = (
                f"syst_{syst}{direction}_{region_name}/"
                f"{variable_name}_syst_{syst}{direction}_{region_name}"
            )
            syst_obj = load_and_rebin(
                input_dirs=input_dirs,
                sample=sample,
                hist_key=syst_hist_key,
                n_rebin=rebin,
                sublumis=input_lumis,
                era_for_scale=era,
                get_kfactor_fn=get_kfactor,
                scales=scales,
            )
            if syst_obj is None:
                syst_obj = hist_obj
            syst_hists.setdefault(region_name, {}) \
                      .setdefault(variable_name, {}) \
                      .setdefault(syst, {}) \
                      .setdefault(direction, {})[sample] = syst_obj


def load_signal_overlays(region, hist_key, rebin, input_dirs, input_lumis,
                         era, era_info, signal_arg, scales):
    """Load signal sample histograms for overlay on signal-region plots.

    Returns a dict {sample_name: hist} for all successfully loaded signals.
    Uses Region.is_boosted to pick the default signal when none is specified.
    """
    signal_hists = {}
    if not region.is_signal_region:
        return signal_hists

    if signal_arg:
        signal_samples = signal_arg
    else:
        signal_samples = default_signals(era_info, boosted=region.is_boosted)

    for sig_sample in signal_samples:
        sig_hist = load_and_rebin(
            input_dirs=input_dirs,
            sample=sig_sample,
            hist_key=hist_key,
            n_rebin=rebin,
            sublumis=input_lumis,
            era_for_scale=era,
            get_kfactor_fn=get_kfactor,
            scales=scales,
        )
        if sig_hist is not None:
            signal_hists[sig_sample] = sig_hist
            logging.debug("  Signal loaded: %s", sig_sample)

    if signal_hists:
        logging.info(
            "  Overlaying %d signal sample(s): %s",
            len(signal_hists), ", ".join(signal_hists),
        )

    return signal_hists

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CR plot commands")

    add_era_args(parser, required=False)
    parser.add_argument("--name",dest="name",type=str,default="",help="Append a suffix to the filenames",)
    parser.add_argument("--plot-config",dest="plot_config",type=str,default=None,help="YAML file with rebin/xlim/ylim for each (region,variable)",)
    parser.add_argument("--region","-r",dest="regions",action="append",default=None,help="Region name(s). Repeat or comma-separate: -r a -r b  or  -r a,b",)
    parser.add_argument("--variable","-v",dest="variables",action="append",default=None,help="Variable name(s). Repeat or comma-separate: -v x -v y  or  -v x,y",)

    parser.add_argument("--variable-rebin",action="store_true",help="Use variable-width bin edges (from rebin_variable in YAML) where available.",)
    parser.add_argument("--local-plots",action="store_true",help="Save plots to a local folder instead of EOS.",)
    parser.add_argument("--unblind",action="store_true",help="Show data in signal regions (default: blinded).",)
    parser.add_argument("--signal","-s",dest="signal",action="append",default=None,help="Signal sample(s) to overlay on SR plots. Repeat or comma-separate: -s signal_WR4000_N2100 -s signal_WR6000_N3100. Defaults depend on era and region topology.",)
    parser.add_argument("--extra-label",dest="extra_label",type=str,default=None,help="Optional label printed below the lumi line on each plot.",)
    parser.add_argument("--dry-run",action="store_true",help="Print which (region, variable) combinations would be plotted without loading histograms or saving files.",)

    # List options for discovery
    parser.add_argument("--list-eras",action="store_true",help="List all available eras and exit.",)
    parser.add_argument("--list-regions",action="store_true",help="List all available regions for the specified era and exit.",)
    parser.add_argument("--list-variables",action="store_true",help="List all available variables and exit.",)

    args = parser.parse_args()

    # Handle list commands early
    if args.list_eras:
        print("Available eras:")
        for era in list_eras():
            print(f"  - {era}")
        sys.exit(0)

    if args.list_variables:
        print("Available variables:")
        for var in sorted([v.name for v in build_variables()]):
            print(f"  - {var}")
        sys.exit(0)

    if args.list_regions:
        if not args.era:
            print("Error: --list-regions requires --era to be specified")
            sys.exit(1)
        from wrplotter.regions import regions_by_name
        regions_map = regions_by_name(args.era)
        print(f"Available regions for era '{args.era}':")
        print("\nRegion names (use with --region):")
        for name in sorted(regions_map.keys()):
            variants = regions_map[name]
            datasets = ", ".join(sorted(set(r.primary_dataset for r in variants)))
            print(f"  - {name:35s} (datasets: {datasets})")

        # Show shorthands (read from data/region_shorthands.yaml — single source of truth)
        print("\nShorthand aliases:")
        for shorthand, expands_to in load_region_shorthands().items():
            expanded_str = ", ".join(expands_to)
            print(f"  - {shorthand:35s} -> {expanded_str}")

        print("\nDataset-specific syntax:")
        print("  - <region_name>:muon    (e.g., wr_resolved_flavor_cr:muon)")
        print("  - <region_name>:egamma  (e.g., wr_resolved_flavor_cr:egamma)")
        sys.exit(0)

    # Now era is required for normal operation
    if not args.era:
        parser.error("--era is required (unless using --list-eras or --list-variables)")

    args.regions   = parse_multi(args.regions)
    args.variables = parse_multi(args.variables)
    args.signal    = parse_multi(args.signal)

    # Check for Run3 unblinding
    if args.unblind and args.era:
        info = load_lumi(args.era)
        if info.get("run") == "Run3":
            parser.error("Run3 data is not ready to be unblinded. Remove --unblind flag.")

    return args

def setup_context(args) -> dict:
    working_dir = repo_root()
    info = load_lumi(args.era)

    era  = args.era
    run  = info["run"]
    year = info["year"]
    lumi = info["lumi"]
    com  = info.get("com", 13.6)

    input_dirs, input_lumis = input_dirs_for_era(era, working_dir, args.dir or "")

    if args.local_plots:
        local_base = repo_root() / "plots" / run / year / era
        if args.dir:
            local_base = local_base / args.dir
        local_base.mkdir(parents=True, exist_ok=True)
        out_dir = local_base
    else:
        out_dir = output_dir(run, year, era, args.dir or None)

    groups, order = load_sample_groups()
    ordered_groups = [groups[k] for k in order if k in groups]
    data_group_keys = {k for k, g in groups.items() if g.kind == "data"}

    for sg in ordered_groups:
        logging.debug(
            "  SampleGroup(%s): kind=%s, %d sample(s), color=%s",
            sg.key, sg.kind, len(sg.samples), sg.color,
        )
    logging.info(
        "Sample groups (%d): %s",
        len(ordered_groups),
        ", ".join(f"{sg.key}[{sg.kind}]" for sg in ordered_groups),
    )

    regions = regions_for_era(era)
    variables = build_variables()
    scales = load_kfactors()
    systematics = load_systematics()

    return dict(
        era=era, run=run, year=year, lumi=lumi, com=com,
        era_info=info,
        input_dirs=input_dirs, input_lumis=input_lumis,
        output_dir=out_dir,
        sample_groups=ordered_groups, data_group_keys=data_group_keys,
        regions=regions, variables=variables,
        scales=scales, systematics=systematics,
    )

def main():
    args = parse_args()
    setup_logging(args.verbose)

    ctx = setup_context(args)

    regions   = ctx["regions"]
    variables = ctx["variables"]

    if args.regions:
        try:
            regions = expand_region_requests(args.era, args.regions)
            logging.info(
                "Restricting to %d region(s): %s",
                len(regions),
                ", ".join(f"{r.name}:{r.primary_dataset}" for r in regions),
            )
        except ValueError as e:
            logging.error(str(e))
            sys.exit(2)

    if args.variables:
        valid_vars = {v.name for v in build_variables()}
        unknown = [v for v in args.variables if v not in valid_vars]
        if unknown:
            logging.error(f"Unknown variable(s): {unknown}. Valid: {sorted(valid_vars)}")
            sys.exit(2)
        name_to_var = {v.name: v for v in variables}
        variables = [name_to_var[n] for n in args.variables if n in name_to_var]
        logging.info(
            "Restricting to %d variable(s): %s",
            len(variables), ", ".join(v.name for v in variables),
        )

    plot_settings = load_plot_settings(args.plot_config or args.era)
    region_cfgs, common_vars = index_plot_settings(plot_settings)

    missing_regions = [r.name for r in regions if r.name not in region_cfgs]
    if missing_regions:
        logging.warning(
            "%d region(s) have no explicit plot-config block — falling back to common_variables: %s",
            len(missing_regions), ", ".join(missing_regions),
        )

    input_dirs   = ctx["input_dirs"]
    input_lumis  = ctx["input_lumis"]
    era          = ctx["era"]
    run          = ctx["run"]
    year         = ctx["year"]
    lumi         = ctx["lumi"]
    com          = ctx["com"]
    out_dir      = ctx["output_dir"]
    scales       = ctx["scales"]
    systematics  = ctx["systematics"]

    # ── Dry-run: just print what would be plotted and exit ─────────────────────
    if args.dry_run:
        total = sum(
            1 for r in regions
            for _ in configured_variables(region_cfgs, common_vars, r.name, variables)
        )
        print(f"Dry run — {era}  ({run} {year}, {lumi:.2f} fb\u207b\xb9 @ {com:.1f} TeV)")
        print(f"{total} plot(s) would be generated:\n")
        for r in regions:
            var_cfgs = configured_variables(region_cfgs, common_vars, r.name, variables)
            if not var_cfgs:
                continue
            print(f"  {r.name}  [{r.primary_dataset}]")
            for v, _ in var_cfgs:
                print(f"    {v.name}")
        sys.exit(0)

    # ── Startup summary ────────────────────────────────────────────────────────
    total_plots = sum(
        1 for r in regions
        for _ in configured_variables(region_cfgs, common_vars, r.name, variables)
    )
    logging.info(
        "Era: %s  (%s %s, %.2f fb\u207b\xb9 @ %.1f TeV)",
        era, run, year, lumi, com,
    )
    logging.info("Output: %s", out_dir)
    logging.info(
        "Generating %d plot(s) across %d region(s) and %d variable(s)",
        total_plots,
        len({r.name for r in regions}),
        len(variables),
    )

    plot_n = 0
    n_failures = 0
    n_skipped = 0

    for region in regions:
        # syst_hists is reset per-region so memory doesn't grow unboundedly
        syst_hists: dict = {}

        for variable, vcfg in configured_variables(region_cfgs, common_vars, region.name, variables):
            plot_n += 1
            logging.info("[%d/%d]  %s / %s", plot_n, total_plots, region.name, variable.name)

            if args.variable_rebin and 'rebin_variable' in vcfg:
                rebin = vcfg['rebin_variable']
            else:
                rebin = vcfg.get('rebin', 1)
            xmin, xmax = map(float, vcfg.get('xlim', (0.0, 1.0)))
            ymin, ymax = map(float, vcfg.get('ylim', (1.0, 1e6)))
            ratio_ymin, ratio_ymax = map(float, vcfg.get('ratio_ylim', (0.5, 2.0)))

            xlim = (xmin, xmax)
            ylim = (ymin, ymax)
            ratio_ylim = (ratio_ymin, ratio_ymax)

            stack_list   = []
            stack_colors = []
            stack_labels = []
            stack_sgs    = []   # parallel list of SampleGroup objects for ordering
            data_hist    = []

            hist_key = f"{region.name}/{variable.name}_{region.name}"

            for sample_group in ctx["sample_groups"]:
                combined = None
                is_data_group = (getattr(sample_group, "kind", "mc") == "data")
                if is_data_group and sample_group.key != region.primary_dataset:
                    continue

                for sample in sample_group.samples:
                    hist_obj = load_and_rebin(
                        input_dirs=input_dirs,
                        sample=sample,
                        hist_key=hist_key,
                        n_rebin=rebin,
                        sublumis=input_lumis,
                        era_for_scale=era,
                        get_kfactor_fn=get_kfactor,
                        scales=scales,
                    )
                    if hist_obj is None:
                        continue
                    combined = hist_obj if (combined is None) else (combined + hist_obj)

                    if not is_data_group:
                        collect_syst_variations(
                            sample, hist_obj, region.name, variable.name,
                            rebin, input_dirs, input_lumis, era, syst_hists,
                            scales, systematics,
                        )

                if combined is None:
                    continue

                if is_data_group:
                    data_hist.append(combined)
                else:
                    color = getattr(sample_group, "color", "#000000")
                    label = (
                        getattr(sample_group, "tlatex_alias", None)
                        or getattr(sample_group, "label", None)
                        or getattr(sample_group, "name", None)
                        or ""
                    )
                    stack_list.append(combined)
                    stack_colors.append(color)
                    stack_labels.append(label)
                    stack_sgs.append(sample_group)

            # Reorder MC stack using the stack_position field on each SampleGroup.
            # Groups with stack_position="top" are moved to the top of the visual stack
            # (last in list) in normal regions, or to the bottom (first in list) in
            # flavor CRs where they contribute a small fraction.
            if stack_sgs:
                items = list(zip(stack_list, stack_colors, stack_labels, stack_sgs))
                top_idxs = [i for i, it in enumerate(items) if it[3].stack_position == "top"]
                if top_idxs:
                    for i in sorted(top_idxs, reverse=True):
                        top_item = items.pop(i)
                        if "flavor_cr" in region.name:
                            items.insert(0, top_item)   # bottom of visual stack
                        else:
                            items.append(top_item)       # top of visual stack
                    stack_list, stack_colors, stack_labels, stack_sgs = (
                        [list(x) for x in zip(*items)]
                    )

            if not stack_list and not data_hist:
                logging.warning(
                    "  [%d/%d]  %s / %s — no histograms found, skipping.",
                    plot_n, total_plots, region.name, variable.name,
                )
                n_skipped += 1
                continue

            show_data = args.unblind or not region.is_signal_region

            signal_hists = load_signal_overlays(
                region, hist_key, rebin, input_dirs, input_lumis,
                era, ctx["era_info"], args.signal, scales,
            )

            fig = plot_stack(
                region, variable,
                stack_list=stack_list, stack_colors=stack_colors,
                stack_labels=stack_labels, data_hist=data_hist,
                xlim=xlim, ylim=ylim, lumi=lumi, com=com,
                ratio_ylim=ratio_ylim, syst_hists=syst_hists,
                show_data=show_data, signal_hists=signal_hists,
                extra_label=args.extra_label,
            )
            outpath = f"{out_dir}/{region.name}_{region.primary_dataset}/{variable.name}_{region.name}.pdf"

            try:
                save_figure(fig, outpath)
                logging.debug("  Saved: %s", outpath)
            except Exception as e:
                logging.error("  Failed to save %s: %s", outpath, e)
                n_failures += 1
            finally:
                plt.close(fig)

    n_saved = plot_n - n_failures - n_skipped
    if n_failures:
        logging.error(
            "Finished with errors — %d saved, %d skipped, %d failed (of %d planned).",
            n_saved, n_skipped, n_failures, total_plots,
        )
        sys.exit(1)
    elif n_skipped:
        logging.info(
            "Done — %d saved, %d skipped (no histograms). Output: %s",
            n_saved, n_skipped, out_dir,
        )
    else:
        logging.info("Done — %d/%d plot(s) saved. Output: %s", n_saved, total_plots, out_dir)

if __name__ == '__main__':
    main()
