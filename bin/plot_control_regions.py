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

from python.plotter import Variable, Region, Plotter
from python import predefined_samples as ps
from src import custom_log_formatter, save_figure, set_y_label
from python.util import resolve_eos_user, build_output_path
from python.config import load_lumi

DATA_GROUPS = {"EGamma", "SingleMuon", "Muon"}

FONT_SIZE_TITLE  = 20
FONT_SIZE_LABEL  = 20
FONT_SIZE_LEGEND = 18

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CR plot commands')
    parser.add_argument('--era', dest='era', type=str, choices=_ERA_CHOICES, required=True, help='Specify the era')
    parser.add_argument("--dir",dest="dir",type=str,default="",help="Optional subdirectory under the input & default EOS output paths")
    parser.add_argument('--name',type=str,default="",help='Append a suffix to the filenames')
    parser.add_argument('--plot-config',dest='plot_config',type=str,default=None,help='YAML file with rebin/xlim/ylim for each (region,variable)')
    parser.add_argument('--cat',dest='category',type=str,choices=["dy_cr", "flavor_cr"],default="dy_cr",help='Append a suffix to the filenames')
    parser.add_argument('--signal-mode',action='store_true',help='Ignore backgrounds/data; plot each WRAnalyzer_signal_*.root per mass point')
    return parser.parse_args()

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

    base_groups = [
        f"SampleGroup_{args.era}_Other",
        f"SampleGroup_{args.era}_Nonprompt",
        f"SampleGroup_{args.era}_TTbar",
        f"SampleGroup_{args.era}_DY",
        f"SampleGroup_{args.era}_EGamma",
        f"SampleGroup_{args.era}_Muon",
    ]
    if args.category == "dy_cr":
        sample_group_names = base_groups.copy()
    else:  # flavor_cr → swap TTbar and DY
        sample_group_names = base_groups.copy()
        sample_group_names[2], sample_group_names[3] = sample_group_names[3], sample_group_names[2]

    plotter.sample_groups = []
    for name in sample_group_names:
        try:
            sg = getattr(ps, name)
        except AttributeError:
            logging.error(f"SampleGroup '{name}' not found in predefined_samples.py")
            sys.exit(1)
        plotter.sample_groups.append(sg)

    if args.category == "dy_cr":
        plotter.regions_to_draw = [
            Region('wr_mumu_resolved_dy_cr','Muon',unblind_data = True,tlatex_alias=f"$\\mu\\mu$\nResolved DY CR\n{args.era}\nNLO $p_{{T}}^{{ll}}$ DY"),
            Region('wr_ee_resolved_dy_cr', 'EGamma', unblind_data = True, tlatex_alias=f"ee\nResolved DY CR\n{args.era}\nNLO $p_{{T}}^{{ll}}$ DY"),
        ]
    else:  # flavor_cr
        plotter.regions_to_draw = [
            Region('wr_resolved_flavor_cr', 'Muon', unblind_data = True,tlatex_alias=f"$e\\mu$\nResolved Flavor CR\n{args.era}"),
            Region('wr_resolved_flavor_cr', 'EGamma', unblind_data = True,tlatex_alias=f"$e\\mu$\nResolved Flavor CR\n{args.era}"),
        ]

    plotter.print_samples()
    plotter.print_regions()

    plotter.variables_to_draw = [
        Variable('mass_fourobject', r'$m_{lljj}$', 'GeV'),
        Variable('pt_leading_jet', r'$p_{T}$ of the leading jet', 'GeV'),
        Variable('mass_dijet', r'$m_{jj}$', 'GeV'),
        Variable('pt_leading_lepton', r'$p_{T}$ of the leading lepton', 'GeV'),
        Variable('eta_leading_lepton', r'$\eta$ of the leading lepton', ''),
        Variable('phi_leading_lepton', r'$\phi$ of the leading lepton', ''),
        Variable('pt_subleading_lepton', r'$p_{T}$ of the subleading lepton', 'GeV'),
        Variable('eta_subleading_lepton', r'$\eta$ of the subleading lepton', ''),
        Variable('phi_subleading_lepton', r'$\phi$ of the subleading lepton', ''),
        Variable('eta_leading_jet', r'$\eta$ of the leading jet', ''),
        Variable('phi_leading_jet', r'$\phi$ of the leading jet', ''),
        Variable('pt_subleading_jet', r'$p_{T}$ of the subleading jet', 'GeV'),
        Variable('eta_subleading_jet', r'$\eta$ of the subleading jet', ''),
        Variable('phi_subleading_jet', r'$\phi$ of the subleading jet', ''),
        Variable('mass_dilepton', r'$m_{ll}$', 'GeV'),
        Variable('pt_dilepton', r'$p_{T}^{ll}$', 'GeV'),
        Variable('pt_dijet', r'$p_{T}^{jj}$', 'GeV'),
        Variable('mass_threeobject_leadlep', r'$m_{l_{\mathrm{pri}}jj}$', 'GeV'),
        Variable('pt_threeobject_leadlep', r'$p^{T}_{l_{\mathrm{pri}}jj}$', 'GeV'),
        Variable('mass_threeobject_subleadlep', r'$m_{l_{\mathrm{sec}}jj}$', 'GeV'),
        Variable('pt_threeobject_subleadlep', r'$p^{T}_{l_{\mathrm{sec}}jj}$', 'GeV'),
        Variable('pt_fourobject', r'$p^{T}_{lljj}$', 'GeV'),
    ]

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
            if sample == "DYJets": #ptll: 0.941. #HT: 1.4, NLO: ?
                rebinned = rebinned * 0.941
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

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    plotter = setup_plotter(args)

    if args.plot_config is None:
        args.plot_config = f"data/plot_settings/{args.era}.yaml"

    if not Path(args.plot_config).is_file():
        logging.error(f"Plot‐config YAML not found: {args.plot_config}")
        sys.exit(1)

    plot_settings = load_plot_settings(args.plot_config)

    common_vars = plot_settings.get("common_variables", {})
    plotter.plot_settings = plot_settings

    missing_regions = [
        r.name for r in plotter.regions_to_draw
        if r.name not in plot_settings
    ]
    if missing_regions:
        logging.error(f"Missing YAML entries for regions: {missing_regions}")
        sys.exit(1)

    input_dirs = plotter.input_directory

    for region in plotter.regions_to_draw:
        logging.info(f"Processing region '{region.name}'")
        for variable in plotter.variables_to_draw:
            logging.info(f"  Variable '{variable.name}'")

            cfg_region = plot_settings.get(region.name, {})
            cfg = cfg_region.get(variable.name, common_vars.get(variable.name))

            if cfg is None:
                logging.error(f"Missing settings for '{variable.name}' (region '{region.name}') "
                              f"and no common_variables fallback found")
                continue

            rebin  = cfg['rebin']
            xmin, xmax = map(float, cfg['xlim'])
            ymin, ymax = map(float, cfg['ylim'])

            plotter.configure_axes(nrebin=rebin,xlim=(xmin, xmax),ylim=(ymin, ymax))
            plotter.reset_stack()
            plotter.reset_data()

            hist_key = f"{region.name}/{variable.name}_{region.name}"

            for sample_group in plotter.sample_groups:
                combined = None

                is_data_group = (sample_group.name in DATA_GROUPS)
                if is_data_group and sample_group.name != region.primary_dataset:
                    continue

                for sample in sample_group.samples:
                    hist_obj = load_and_rebin(input_dirs, sample, hist_key, plotter, is_data_group)
                    if hist_obj is None:
                        continue

                    combined = hist_obj if (combined is None) else (combined + hist_obj)

                if combined is None:
                    logging.warning(f"    No histograms found for group '{sample_group.name}'")
                    continue

                if is_data_group:
                    plotter.store_data(combined)
                else:
                    plotter.accumulate_histogram(combined, sample_group.color,sample_group.tlatex_alias)

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
