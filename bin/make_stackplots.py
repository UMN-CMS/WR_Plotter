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
from wrplotter.io import repo_root, output_dir, save_figure
from wrplotter.histo import load_and_rebin
from wrplotter.regions import regions_for_era, expand_region_requests
from wrplotter.variables import build_variables
from wrplotter.sample_groups import load_sample_groups
from wrplotter.config import list_eras,load_lumi,load_plot_settings,load_kfactors,get_kfactor,index_plot_settings, configured_variables
from wrplotter.cli_utils import parse_multi, setup_logging

_ERA_CHOICES = list_eras()
SCALES = load_kfactors()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CR plot commands")

    parser.add_argument("--era",dest="era",type=str,choices=_ERA_CHOICES,required=False,help="Specify the era",)
    parser.add_argument("--dir",dest="dir",type=str,default="",help="Optional subdirectory under the input & default EOS output paths",)
    parser.add_argument("--name",dest="name",type=str,default="",help="Append a suffix to the filenames",)
    parser.add_argument("--plot-config",dest="plot_config",type=str,default=None,help="YAML file with rebin/xlim/ylim for each (region,variable)",)
    parser.add_argument("--region","-r",dest="regions",action="append",default=None,help="Region name(s). Repeat or comma-separate: -r a -r b  or  -r a,b",)
    parser.add_argument("--variable","-v",dest="variables",action="append",default=None,help="Variable name(s). Repeat or comma-separate: -v x -v y  or  -v x,y",)

    parser.add_argument("--variable-rebin",action="store_true",help="Use variable-width bin edges (from rebin_variable in YAML) where available.",)
    parser.add_argument("--local-plots",action="store_true",help="Save plots to a local folder instead of EOS.",)
    parser.add_argument("--unblind",action="store_true",help="Show data in signal regions (default: blinded).",)
    parser.add_argument("--signal","-s",dest="signal",action="append",default=None,help="Signal sample(s) to overlay on SR plots. Repeat or comma-separate: -s signal_WR4000_N2100 -s signal_WR6000_N3100. Defaults depend on era and region topology.",)

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
        from wrplotter.regions import regions_by_name
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

    if "sub_eras" in info:
        input_dirs = []
        for se in info["sub_eras"]:
            se_info = load_lumi(se)
            d = working_dir / "rootfiles" / se_info["run"] / se_info["year"] / se
            if args.dir:
                d = d / args.dir
            input_dirs.append(d)
        input_lumis = info["sub_lumis"]
    else:
        base = working_dir / "rootfiles" / run / year / era
        if args.dir:
            base = base / args.dir
        input_dirs = [base]
        input_lumis = [lumi]

    if args.local_plots:
        local_base = repo_root() / "plots" / run / year / era
        if args.dir:
            local_base = local_base / args.dir
        local_base.mkdir(parents=True, exist_ok=True)
        out_dir = local_base
    else:
        out_dir = output_dir(run, year, era, args.dir or None)

    groups, order = load_sample_groups(era)
    ordered_groups = [groups[k] for k in order if k in groups]
    data_group_keys = {k for k, g in groups.items() if g.kind == "data"}

    for sg in ordered_groups:
        sg.print()

    regions = regions_for_era(era)
    variables = build_variables()

    return dict(
        era=era, run=run, year=year, lumi=lumi, com=com,
        era_info=info,
        input_dirs=input_dirs, input_lumis=input_lumis,
        output_dir=out_dir,
        sample_groups=ordered_groups, data_group_keys=data_group_keys,
        regions=regions, variables=variables,
    )

def main():
    args = parse_args()
    setup_logging()

    ctx = setup_context(args)

    regions   = ctx["regions"]
    variables = ctx["variables"]

    if args.regions:
        try:
            regions = expand_region_requests(args.era, args.regions)
            logging.info(
                "Restricted to regions (expanded): %s",
                [f"{r.name}:{r.primary_dataset}" for r in regions],
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
        logging.info(f"Restricted to variables: {args.variables}")

    plot_settings = load_plot_settings(args.plot_config or args.era)
    region_cfgs, common_vars = index_plot_settings(plot_settings)

    missing_regions = [r.name for r in regions if r.name not in region_cfgs]
    if missing_regions:
        logging.warning(f"Regions missing explicit blocks in YAML (will use common_variables fallback where possible): {missing_regions}")

    input_dirs  = ctx["input_dirs"]
    input_lumis = ctx["input_lumis"]
    era         = ctx["era"]
    run         = ctx["run"]
    lumi        = ctx["lumi"]
    com         = ctx["com"]
    out_dir     = ctx["output_dir"]

    syst_hists = {}

    for region in regions:
        logging.info(f"Processing region '{region.name}'")

        for variable, vcfg in configured_variables(region_cfgs, common_vars, region.name, variables):
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
                        scales=SCALES,
                    )
                    if hist_obj is None:
                        continue
                    combined = hist_obj if (combined is None) else (combined + hist_obj)

                    # --- load systematic variations (MC only) ---
                    if is_data_group:
                        continue
                    for syst in ["lumi", "pileup",
                                 "muonrecosf", "muonidsf", "muonisosf", "muontrigsf",
                                 "electronrecosf", "electronidsf", "electrontrigsf"]:
                        for direction in ["up", "down"]:
                            syst_hist_key = (
                                f"syst_{syst}{direction}_{region.name}/"
                                f"{variable.name}_syst_{syst}{direction}_{region.name}"
                            )
                            syst_obj = load_and_rebin(
                                input_dirs=input_dirs,
                                sample=sample,
                                hist_key=syst_hist_key,
                                n_rebin=rebin,
                                sublumis=input_lumis,
                                era_for_scale=era,
                                get_kfactor_fn=get_kfactor,
                                scales=SCALES,
                            )
                            # Use nominal as fallback so that samples without
                            # syst histograms contribute delta=0 rather than
                            # being absent (which would bias the envelope).
                            if syst_obj is None:
                                syst_obj = hist_obj
                            syst_hists.setdefault(region.name, {}) \
                                      .setdefault(variable.name, {}) \
                                      .setdefault(syst, {}) \
                                      .setdefault(direction, {})[sample] = syst_obj

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

            if not stack_list and not data_hist:
                logging.warning(f"  Skipped '{region.name}/{variable.name}' (no histograms found).")
                continue

            # Determine if this is a signal region and if we should blind it
            is_signal_region = 'sr' in region.name.lower()
            show_data = args.unblind or not is_signal_region

            # --- Load signal overlays for signal regions ---
            signal_hists = {}
            if is_signal_region:
                if args.signal:
                    signal_samples = args.signal
                else:
                    era_info = ctx["era_info"]
                    is_boosted_region = 'boosted' in region.name.lower()
                    if is_boosted_region:
                        signal_samples = [era_info.get("default_signal_boosted", "signal_WR4000_N100")]
                    else:
                        signal_samples = [era_info.get("default_signal_resolved", "signal_WR4000_N2100")]

                for sig_sample in signal_samples:
                    sig_hist = load_and_rebin(
                        input_dirs=input_dirs,
                        sample=sig_sample,
                        hist_key=hist_key,
                        n_rebin=rebin,
                        sublumis=input_lumis,
                        era_for_scale=era,
                        get_kfactor_fn=get_kfactor,
                        scales=SCALES,
                    )
                    if sig_hist is not None:
                        signal_hists[sig_sample] = sig_hist
                        logging.info(f"    Loaded signal: {sig_sample}")
                if signal_hists:
                    logging.info(f"    {len(signal_hists)} signal sample(s) will be overlaid")

            fig = plot_stack(
                region, variable,
                stack_list=stack_list, stack_colors=stack_colors,
                stack_labels=stack_labels, data_hist=data_hist,
                xlim=xlim, ylim=ylim, lumi=lumi, com=com,
                ratio_ylim=ratio_ylim, syst_hists=syst_hists,
                show_data=show_data, signal_hists=signal_hists,
            )
            outpath = f"{out_dir}/{region.name}_{region.primary_dataset}/{variable.name}_{region.name}.pdf"

            try:
                save_figure(fig, outpath)
                logging.info(f"    Saved: {outpath}")
            except Exception as e:
                logging.error(f"    Failed to save {outpath}: {e}")
            finally:
                plt.close(fig)

if __name__ == '__main__':
    main()
