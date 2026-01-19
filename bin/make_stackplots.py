#!/usr/bin/env python3

# ── Standard library ────────────────────────────────────────────────────────────
import sys
import logging
from pathlib import Path
from collections import defaultdict
import argparse

# ── Third-party ────────────────────────────────────────────────────────────────
import yaml
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep

# ── Local imports (after we add repo to sys.path below) ────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from python.plotter import Plotter
from src.plotting_helpers import custom_log_formatter, set_y_label, plot_stack
from python.io import data_path, read_yaml, read_json,output_dir, save_figure
from python.histo import load_and_rebin
from python.regions import regions_for_era, expand_region_requests
from python.variables import Variable, build_variables
from python.sample_groups import load_sample_groups
from python.config import list_eras,load_lumi,load_plot_settings,load_kfactors,get_kfactor,index_plot_settings, get_var_cfg

_ERA_CHOICES = list_eras()
SCALES = load_kfactors()

FONT_SIZE_TITLE  = 20
FONT_SIZE_LABEL  = 20
FONT_SIZE_LEGEND = 18

def _parse_multi(opt):
    """
    Accepts a list of strings from argparse with action='append'.
    Each string may itself be a comma-separated list.
    Returns a deduped list preserving first-seen order, or None if empty.
    """
    if not opt:
        return None
    items = []
    for part in opt:
        items.extend([x.strip() for x in part.split(",") if x.strip()])
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out or None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CR plot commands")

    parser.add_argument("--era",dest="era",type=str,choices=_ERA_CHOICES,required=False,help="Specify the era",)
    parser.add_argument("--dir",dest="dir",type=str,default="",help="Optional subdirectory under the input & default EOS output paths",)
    parser.add_argument("--name",dest="name",type=str,default="",help="Append a suffix to the filenames",)
    parser.add_argument("--plot-config",dest="plot_config",type=str,default=None,help="YAML file with rebin/xlim/ylim for each (region,variable)",)
    parser.add_argument("--region","-r",dest="regions",action="append",default=None,help="Region name(s). Repeat or comma-separate: -r a -r b  or  -r a,b",)
    parser.add_argument("--variable","-v",dest="variables",action="append",default=None,help="Variable name(s). Repeat or comma-separate: -v x -v y  or  -v x,y",)

    parser.add_argument("--local-plots",action="store_true",help="Save plots to a local folder instead of EOS.",)
    parser.add_argument("--unblind",action="store_true",help="Show data in signal regions (default: blinded).",)

    # List options for discovery
    parser.add_argument("--list-eras",action="store_true",help="List all available eras and exit.",)
    parser.add_argument("--list-regions",action="store_true",help="List all available regions for the specified era and exit.",)
    parser.add_argument("--list-variables",action="store_true",help="List all available variables and exit.",)

    args = parser.parse_args()

    # Handle list commands early
    if args.list_eras:
        print("Available eras:")
        for era in _ERA_CHOICES:
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
        from python.regions import regions_by_name
        regions_map = regions_by_name(args.era)
        print(f"Available regions for era '{args.era}':")
        print("\nRegion names (use with --region):")
        for name in sorted(regions_map.keys()):
            variants = regions_map[name]
            datasets = ", ".join(sorted(set(r.primary_dataset for r in variants)))
            print(f"  - {name:35s} (datasets: {datasets})")

        # Show shorthands
        print("\nShorthand aliases:")
        shorthands = {
            "resolved_dy_cr": "wr_ee_resolved_dy_cr, wr_mumu_resolved_dy_cr",
            "resolved_sr": "wr_ee_resolved_sr, wr_mumu_resolved_sr",
            "resolved_flavor_cr": "wr_resolved_flavor_cr",
            "boosted_dy_cr": "wr_ee_boosted_dy_cr, wr_mumu_boosted_dy_cr",
            "boosted_sr": "wr_ee_boosted_sr, wr_mumu_boosted_sr",
            "boosted_flavor_cr": "wr_emu_boosted_flavor_cr, wr_mue_boosted_flavor_cr",
        }
        for shorthand, expands_to in shorthands.items():
            print(f"  - {shorthand:35s} -> {expands_to}")

        print("\nDataset-specific syntax:")
        print("  - <region_name>:muon    (e.g., wr_resolved_flavor_cr:muon)")
        print("  - <region_name>:egamma  (e.g., wr_resolved_flavor_cr:egamma)")
        sys.exit(0)

    # Now era is required for normal operation
    if not args.era:
        parser.error("--era is required (unless using --list-eras or --list-variables)")

    args.regions   = _parse_multi(args.regions)
    args.variables = _parse_multi(args.variables)

    return args

def setup_plotter(args) -> Plotter:
    working_dir = REPO_DIR
    info = load_lumi(args.era)

    plotter = Plotter()
    plotter.era  = args.era
    plotter.run  = info["run"]
    plotter.year = info["year"]
    plotter.lumi = info["lumi"]
    plotter.scale = True

    if "sub_eras" in info:
        dirs = []
        for se in info["sub_eras"]:
            se_info = load_lumi(se)
            d = working_dir / "rootfiles" / se_info["run"] / se_info["year"] / se
            if args.dir:
                d = d / args.dir
            dirs.append(d)
        plotter.input_directory = dirs
        plotter.input_lumis = info["sub_lumis"]
    else:
        base = working_dir / "rootfiles" / plotter.run / plotter.year / plotter.era
        if args.dir:
            base = base / args.dir
        plotter.input_directory = [base]
        plotter.input_lumis = [plotter.lumi]

    if args.local_plots:
        local_base = REPO_DIR / "plots" / plotter.run / plotter.year / plotter.era
        if args.dir:
            local_base = local_base / args.dir
        local_base.mkdir(parents=True, exist_ok=True)
        plotter.output_directory = local_base
    else:
        plotter.output_directory = output_dir(plotter.run, plotter.year, plotter.era, args.dir or None)

    groups, order = load_sample_groups(args.era)


    print(groups)
    print(order)
    ordered_groups = [groups[k] for k in order if k in groups]
    plotter.sample_groups = ordered_groups

    plotter.data_group_keys = {k for k, g in groups.items() if g.kind == "data"}

    plotter.print_samples()

    regions = regions_for_era(args.era)
    plotter.regions_to_draw = regions
    plotter.print_regions()

    variables = build_variables()
    plotter.variables_to_draw = variables
    plotter.print_variables()

    return plotter

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    plotter = setup_plotter(args)

    if args.regions:
        try:
            expanded = expand_region_requests(args.era, args.regions)
            plotter.regions_to_draw = expanded
            logging.info(
                "Restricted to regions (expanded): %s",
                [f"{r.name}:{r.primary_dataset}" for r in expanded],
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
        name_to_var = {v.name: v for v in plotter.variables_to_draw}
        plotter.variables_to_draw = [name_to_var[n] for n in args.variables if n in name_to_var]
        logging.info(f"Restricted to variables: {args.variables}")

    if args.plot_config is None:
        args.plot_config = f"data/plot_settings/{args.era}.yaml"

    plot_settings = load_plot_settings(args.plot_config or args.era)

    region_cfgs, common_vars = index_plot_settings(plot_settings)
    plotter.plot_settings = plot_settings

    missing_regions = [r.name for r in plotter.regions_to_draw if r.name not in region_cfgs]
    if missing_regions:
        logging.warning(f"Regions missing explicit blocks in YAML (will use common_variables fallback where possible): {missing_regions}")

    input_dirs = plotter.input_directory

    for region in plotter.regions_to_draw:
        logging.info(f"Processing region '{region.name}'")

        for variable in plotter.variables_to_draw:
            vcfg = get_var_cfg(region_cfgs, common_vars, region.name, variable.name)
            if vcfg is None:
                logging.warning(f"  Skipping '{region.name}/{variable.name}': no settings and no common fallback.")
                continue

            rebin = vcfg.get('rebin', 1)
            xmin, xmax = map(float, vcfg.get('xlim', (0.0, 1.0)))
            ymin, ymax = map(float, vcfg.get('ylim', (1.0, 1e6)))
            ratio_ymin, ratio_ymax = map(float, vcfg.get('ratio_ylim', (0.5, 2.0)))

            plotter.configure_axes(nrebin=rebin, xlim=(xmin, xmax), ylim=(ymin, ymax))
            plotter.ratio_ylim = (ratio_ymin, ratio_ymax)
            plotter.reset_stack()
            plotter.reset_data()

            hist_key = f"{region.name}/{variable.name}_{region.name}"

            any_content = False
            for sample_group in plotter.sample_groups:
                combined = None
                is_data_group = (getattr(sample_group, "kind", "mc") == "data")
                if is_data_group and sample_group.key != region.primary_dataset:
                    continue
                
                for sample in sample_group.samples:
                    hist_obj = load_and_rebin(
                        input_dirs=plotter.input_directory,
                        sample=sample,
                        hist_key=hist_key,
                        plotter=plotter,
                        is_data_group=is_data_group,
                        sublumis=plotter.input_lumis,
                        era_for_scale=plotter.era,
                        get_kfactor_fn=get_kfactor,
                        scales=SCALES,
                    )
                    if hist_obj is None:
                        continue
                    combined = hist_obj if (combined is None) else (combined + hist_obj)

                    # --- load systematic variations ---
                    if not hasattr(plotter, "syst_hists"):
                        plotter.syst_hists = {}

                    for syst in ["lumi"]:  # add more later (PU, JES, etc.)
                        for direction in ["up", "down"]:
                            syst_hist_key = (
                                f"syst_{syst}{direction}_{region.name}/"
                                f"{variable.name}_syst_{syst}{direction}_{region.name}"
                            )
                            syst_obj = load_and_rebin(
                                input_dirs=plotter.input_directory,
                                sample=sample,
                                hist_key=syst_hist_key,
                                plotter=plotter,
                                is_data_group=is_data_group,
                                sublumis=plotter.input_lumis,
                                era_for_scale=plotter.era,
                                get_kfactor_fn=get_kfactor,
                                scales=SCALES,
                            )
                            if syst_obj is not None:
                                plotter.syst_hists.setdefault(region.name, {}) \
                                                   .setdefault(variable.name, {}) \
                                                   .setdefault(syst, {}) \
                                                   .setdefault(direction, {})[sample] = syst_obj

                if combined is None:
                    continue

                any_content = True
                if is_data_group:
                    plotter.store_data(combined)
                else:
                    color = getattr(sample_group, "color", "#000000")
                    label = (
                        getattr(sample_group, "tlatex_alias", None)
                        or getattr(sample_group, "label", None)
                        or getattr(sample_group, "name", None)
                        or ""
                    )
                    plotter.accumulate_histogram(combined, color, label)

            if not plotter.stack_list and not plotter.data_hist:
                logging.warning(f"  Skipped '{region.name}/{variable.name}' (no histograms found).")
                continue

            # Determine if this is a signal region and if we should blind it
            is_signal_region = 'sr' in region.name.lower()
            show_data = args.unblind or not is_signal_region

            fig = plot_stack(plotter, region, variable, show_data=show_data)
            outpath = f"{plotter.output_directory}/{region.name}_{region.primary_dataset}/{variable.name}_{region.name}.pdf"

            try:
                save_figure(fig, outpath)
                logging.info(f"    Saved: {outpath}")
            except Exception as e:
                logging.error(f"    Failed to save {outpath}: {e}")
            finally:
                plt.close(fig)

if __name__ == '__main__':
    main()
