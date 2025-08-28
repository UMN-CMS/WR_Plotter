#!/usr/bin/env python3

# ── Standard library ────────────────────────────────────────────────────────────
import sys
import logging
from pathlib import Path
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
# Add repo root to sys.path so we can import our local packages
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
import json
_LUMI_PATH = REPO_DIR / "data" / "lumi.json"
with open(_LUMI_PATH) as _f:
    _ERA_CHOICES = sorted(json.load(_f).keys())

from python.plotter import Plotter
from src import custom_log_formatter, save_figure, set_y_label
from python.util import resolve_eos_user, build_output_path
from python.regions import Region, build_regions
from python.variables import Variable, build_variables
from python.config import load_lumi, load_kfactors, get_kfactor
from python.sample_groups import load_sample_groups

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

    # required era
    parser.add_argument(
        "--era",
        dest="era",
        type=str,
        choices=_ERA_CHOICES,
        required=True,
        help="Specify the era",
    )

    # paths / config
    parser.add_argument(
        "--dir",
        dest="dir",
        type=str,
        default="",
        help="Optional subdirectory under the input & default EOS output paths",
    )
    parser.add_argument(
        "--name",
        dest="name",
        type=str,
        default="",
        help="Append a suffix to the filenames",
    )
    parser.add_argument(
        "--plot-config",
        dest="plot_config",
        type=str,
        default=None,
        help="YAML file with rebin/xlim/ylim for each (region,variable)",
    )

    # category
    parser.add_argument(
        "--cat",
        dest="category",
        type=str,
        choices=["dy_cr", "flavor_cr"],
        default="dy_cr",
        help="Plot category",
    )

    # filters: allow multiple via repeat or commas
    parser.add_argument(
        "--region",
        "-r",
        dest="regions",
        action="append",
        default=None,
        help="Region name(s). Repeat or comma-separate: -r a -r b  or  -r a,b",
    )
    parser.add_argument(
        "--variable",
        "-v",
        dest="variables",
        action="append",
        default=None,
        help="Variable name(s). Repeat or comma-separate: -v x -v y  or  -v x,y",
    )

    args = parser.parse_args()

    # normalize multi-value args into lists (or None)
    args.regions   = _parse_multi(args.regions)
    args.variables = _parse_multi(args.variables)

    return args

def load_plot_settings(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def setup_plotter(args) -> Plotter:
    working_dir = REPO_DIR
    info = load_lumi(args.era)

    plotter = Plotter()
    plotter.era  = args.era
    plotter.run  = info["run"]
    plotter.year = info["year"]
    plotter.lumi = info["lumi"]
    plotter.scale = True

    # Build input directories and lumis (supports combined eras via sub_eras)
    if "sub_eras" in info:
        dirs = []
        for se in info["sub_eras"]:
            se_info = load_lumi(se)  # ensures sub-era exists and has run/year
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

    eos_user = resolve_eos_user()
    if args.dir:
        plotter.output_directory = build_output_path(plotter.run, plotter.year, plotter.era, args.dir, eos_user)
    else:
        plotter.output_directory = build_output_path(plotter.run, plotter.year, plotter.era, None, eos_user)

    # (Optional) ensure it exists if build_output_path doesn't create it
#    Path(plotter.output_directory).mkdir(parents=True, exist_ok=True)

    # Load groups & order from YAML
    groups, order = load_sample_groups(args.era)

    # category-specific tweak: swap dy and ttbar for flavor_cr
    if args.category == "flavor_cr":
        swap = {"dy": "ttbar", "ttbar": "dy"}
        order = [swap.get(k, k) for k in order]

    # Ordered list of groups, but skip unknown keys gracefully
    ordered_groups = [groups[k] for k in order if k in groups]
    plotter.sample_groups = ordered_groups

    # Expose data-group keys on the plotter (for later filtering in main)
    plotter.data_group_keys = {k for k, g in groups.items() if g.kind == "data"}

    plotter.print_samples()

    # Regions/variables
    regions = build_regions(args.era, args.category)
    plotter.regions_to_draw = regions
    plotter.print_regions()

    variables = build_variables()
    plotter.variables_to_draw = variables
    plotter.print_variables()

    return plotter

def load_and_rebin(input_dirs: list[Path],sample: str,hist_key: str,plotter: Plotter,is_data_group: bool):
    combined = None

    original_lumi = plotter.lumi

    for indir, sublumi in zip(plotter.input_directory, plotter.input_lumis):
        fp = indir / f"WRAnalyzer_{sample}.root"
        try:
            with uproot.open(fp) as f:
                raw_hist = f[hist_key].to_hist()
        except (FileNotFoundError, KeyError):
            continue

        # Rebin first
#        if "mass_fourobject" in hist_key:
#            variable_edges = [0, 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 8000]
#            rebinned = plotter.rebin_hist(raw_hist, variable_edges)
#        elif "pt_leading_jet" in hist_key:
#            variable_edges = [0, 40, 100, 200, 400, 600, 800, 1000, 1500, 2000]
#            rebinned = plotter.rebin_hist(raw_hist, variable_edges)
#        elif "mass_dijet" in hist_key:
#            variable_edges = [0, 200, 400, 600, 800, 1000, 1250, 1500, 2000, 4000]
#            rebinned = plotter.rebin_hist(raw_hist, variable_edges)
#        else:
        rebinned = plotter.rebin_hist(raw_hist)

        if not is_data_group:
            plotter.lumi = sublumi
            rebinned = plotter.scale_hist(rebinned)

        # apply per-era per-sample k-factor
        era_for_scale = indir.name if indir.name in SCALES else plotter.era
        k = get_kfactor(SCALES, era_for_scale, sample, default=1.0)
        rebinned = rebinned * k

        combined = rebinned if (combined is None) else (combined + rebinned)

    plotter.lumi = original_lumi
    return combined


def reorder_legend(ax, priority_labels=("MC stat. unc.", "Data"), fontsize=FONT_SIZE_LEGEND):
    handles, labels = ax.get_legend_handles_labels()
    idx_map = {label: i for i, label in enumerate(labels)}
    mc_idx = idx_map.get(priority_labels[0], None)
    data_idx = idx_map.get(priority_labels[1], None)

    bkg_idxs = [i for i in range(len(labels)) if i not in {mc_idx, data_idx}]
    bkg_idxs.reverse()

    new_order = []
    if mc_idx is not None:
        new_order.append(mc_idx)
    if data_idx is not None:
        new_order.append(data_idx)
    new_order.extend(bkg_idxs)

    new_handles = [handles[i] for i in new_order]
    new_labels  = [labels[i]  for i in new_order]
    ax.legend(new_handles, new_labels, loc="best", fontsize=fontsize)


def plot_stack(plotter, region, variable):
    bkg_stack = plotter.stack_list
    bkg_labels = plotter.stack_labels
    bkg_colors = plotter.stack_colors

    tot = sum(plotter.stack_list)     # “total MC” histogram
    data = sum(plotter.data_hist)     # “data” histogram
    edges = tot.axes[0].edges

    hep.style.use("CMS")
    fig, (ax, rax) = plt.subplots(2, 1,gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1),sharex=True)

    hep.histplot(bkg_stack,stack=True,label=bkg_labels,color=bkg_colors,histtype='fill',ax=ax)

    hep.histplot(data,label="Data",xerr=True,color='k',histtype='errorbar',ax=ax)

    errps = {'hatch': '////', 'facecolor': 'none', 'lw': 0, 'edgecolor': 'k', 'alpha': 0.5}
    hep.histplot(tot,histtype='band',ax=ax,**errps,label='MC stat. unc.')

    data_vals = data.values()
    tot_vals  = tot.values()
    ratio     = np.zeros_like(data_vals, dtype=float)
    ratio_err = np.zeros_like(data_vals, dtype=float)
    mask = tot_vals > 0

    if np.any(mask):
        ratio[mask]     = data_vals[mask] / tot_vals[mask]
        ratio_err[mask] = np.sqrt(data_vals[mask]) / tot_vals[mask]

    tot_vars = tot.variances()
    mc_err = np.sqrt(tot_vars)
    rel_err = np.zeros_like(tot_vals, dtype=float)
    rel_err[mask] = mc_err[mask] / tot_vals[mask]

    hep.histplot(ratio,edges,yerr=ratio_err,xerr=True,ax=rax,histtype="errorbar",color="k",capsize=4,label="Data",)

    band_low  = np.ones_like(rel_err) - rel_err
    band_high = np.ones_like(rel_err) + rel_err
    band_low[~mask] = np.nan
    band_high[~mask] = np.nan
    rax.stairs(band_low, edges, baseline=band_high, **errps)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylim(*plotter.ylim)
    ax.text(0.05, 0.96,region.tlatex_alias,transform=ax.transAxes,fontsize=FONT_SIZE_TITLE,verticalalignment='top')
    hep.cms.label(loc=0,ax=ax,data=region.unblind_data,label="Work in Progress",lumi=f"{plotter.lumi:.1f}",com=13.6,fontsize=FONT_SIZE_LABEL)

    if variable.unit:
        xlabel = f"{variable.tlatex_alias} [{variable.unit}]"
    else:
        xlabel = variable.tlatex_alias

    rax.set_xlabel(xlabel)
    ax.set_xlabel("")
    rax.set_ylabel("Data/Sim.")
    rax.set_ylim(0.7, 1.3)
    rax.set_yticks([0.8, 1.0, 1.2])
    rax.axhline(1.0, ls='--', color='k')
    ax.set_xlim(*plotter.xlim)
    set_y_label(ax, tot, variable)
    reorder_legend(ax)

    return fig

def _index_plot_settings(plot_settings: dict) -> tuple[dict, dict]:
    """Return (region_cfgs, common_vars)."""
    common_vars = plot_settings.get("common_variables", {})
    region_cfgs = {r: plot_settings.get(r, {}) for r in plot_settings.keys() if r != "common_variables"}
    return region_cfgs, common_vars

def _get_var_cfg(region_cfgs, common_vars, region_name, var_name):
    reg = region_cfgs.get(region_name, {})
    return reg.get(var_name, common_vars.get(var_name))

def _ensure_output_leaf(plotter: Plotter, region: Region) -> None:
    leaf = Path(plotter.output_directory) / f"{region.name}_{region.primary_dataset}"
    leaf.mkdir(parents=True, exist_ok=True)

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    plotter = setup_plotter(args)

    # Restrict regions if requested
    if args.regions:
        valid_regions = {r.name for r in build_regions(args.era, args.category)}
        unknown = [r for r in args.regions if r not in valid_regions]
        if unknown:
            logging.error(f"Unknown region(s): {unknown}. Valid: {sorted(valid_regions)}")
            sys.exit(2)
        name_to_region = {r.name: r for r in plotter.regions_to_draw}
        plotter.regions_to_draw = [name_to_region[n] for n in args.regions if n in name_to_region]
        logging.info(f"Restricted to regions: {args.regions}")

    # Restrict variables if requested
    if args.variables:
        valid_vars = {v.name for v in build_variables()}
        unknown = [v for v in args.variables if v not in valid_vars]
        if unknown:
            logging.error(f"Unknown variable(s): {unknown}. Valid: {sorted(valid_vars)}")
            sys.exit(2)
        name_to_var = {v.name: v for v in plotter.variables_to_draw}
        plotter.variables_to_draw = [name_to_var[n] for n in args.variables if n in name_to_var]
        logging.info(f"Restricted to variables: {args.variables}")

    # Pick default config file by era if not provided
    if args.plot_config is None:
        args.plot_config = f"data/plot_settings/{args.era}.yaml"

    cfg_path = Path(args.plot_config)
    if not cfg_path.is_file():
        logging.error(f"Plot-config YAML not found: {cfg_path}")
        sys.exit(1)

    # Load & index settings
    plot_settings = load_plot_settings(str(cfg_path))
    region_cfgs, common_vars = _index_plot_settings(plot_settings)
    plotter.plot_settings = plot_settings  # if other internals want it

    # Sanity: all planned regions must exist in YAML (or have common fallbacks for every var)
    missing_regions = [r.name for r in plotter.regions_to_draw if r.name not in region_cfgs]
    if missing_regions:
        logging.warning(f"Regions missing explicit blocks in YAML (will use common_variables fallback where possible): {missing_regions}")

    input_dirs = plotter.input_directory

    for region in plotter.regions_to_draw:
        logging.info(f"Processing region '{region.name}'")
#        _ensure_output_leaf(plotter, region)

        for variable in plotter.variables_to_draw:
            vcfg = _get_var_cfg(region_cfgs, common_vars, region.name, variable.name)
            if vcfg is None:
                logging.warning(f"  Skipping '{region.name}/{variable.name}': no settings and no common fallback.")
                continue

            rebin = vcfg.get('rebin', 1)
            xmin, xmax = map(float, vcfg.get('xlim', (0.0, 1.0)))
            ymin, ymax = map(float, vcfg.get('ylim', (1.0, 1e6)))

            plotter.configure_axes(nrebin=rebin, xlim=(xmin, xmax), ylim=(ymin, ymax))
            plotter.reset_stack()
            plotter.reset_data()

            hist_key = f"{region.name}/{variable.name}_{region.name}"

            # Load/accumulate
            any_content = False
            for sample_group in plotter.sample_groups:
                combined = None

                is_data_group = (getattr(sample_group, "kind", "mc") == "data")
                # only the primary data stream (keys like "muon", "egamma")

                # If in wr_mumu_resolved_dy_cr, skip the electron root file 
                if is_data_group and sample_group.key != region.primary_dataset:
                    continue
                

                for sample in sample_group.samples:
                    hist_obj = load_and_rebin(input_dirs, sample, hist_key, plotter, is_data_group)
                    if hist_obj is None:
                        continue
                    combined = hist_obj if (combined is None) else (combined + hist_obj)

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
                        or ""   # final fallback, no group_key
                    )
                    plotter.accumulate_histogram(combined, color, label)

            # only skip if BOTH MC and Data are missing
            if not plotter.stack_list and not plotter.data_hist:
                logging.warning(f"  Skipped '{region.name}/{variable.name}' (no histograms found).")
                continue

            fig = plot_stack(plotter, region, variable)
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
