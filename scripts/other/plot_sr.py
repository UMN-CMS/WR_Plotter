#!/usr/bin/env python3
"""
plot_sr.py

Generate stacked control‐region plots (DY_CR or Flavor_CR) for a given era.
Usage example:
    python3 scripts/plot_sr.py --era 2022 --dir dy_ht
    python3 scripts/plot_sr.py --era Run3Summer22EE --dir dy_ht
    python3 scripts/plot_sr.py --era Run3Summer22
    python3 scripts/plot_sr.py --era RunIISummer20UL18 --dir dy_ht
"""

# Standard library
import sys
import logging
from pathlib import Path
import argparse

# Third‐party
import yaml
import numpy as np
import hist
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
from hist.intervals import ratio_uncertainty
import subprocess

# Custom imports
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
from python.plotter import SampleGroup, Variable, Region, Plotter
from python import predefined_samples as ps
from src import custom_log_formatter, save_figure, set_y_label

# Configuration for supported eras
ERA_CONFIG = {
    "RunIISummer20UL18":   {"run": "RunII", "year": "2018", "lumi": 59.832422397},
    "Run3Summer22":        {"run": "Run3",  "year": "2022", "lumi": 7.9804},
    "Run3Summer22EE":      {"run": "Run3",  "year": "2022", "lumi": 26.6717},
    "2022":                {"run": "Run3",  "year": "2022", "lumi": 26.6717 + 7.9804},
}

# Set of data‐like sample groups
DATA_GROUPS = {"EGamma", "SingleMuon", "Muon"}

# Font size constants
FONT_SIZE_TITLE  = 20
FONT_SIZE_LABEL  = 20
FONT_SIZE_LEGEND = 18

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CR plot commands')
    parser.add_argument(
        '--era',
        dest='era',
        type=str,
        choices=list(ERA_CONFIG.keys()),
        required=True,
        help='Specify the era (e.g. Run3Summer22EE)'
    )
    parser.add_argument(
        "--dir",
        dest="dir",
        type=str,
        default="",
        help="Optional subdirectory under the input & default EOS output paths"
    )
    parser.add_argument(
        '--name',
        type=str,
        default="",
        help='Append a suffix to the filenames'
    )
    parser.add_argument(
        '--plot-config',
        dest='plot_config',
        type=str,
        default=None,
        help='YAML file with rebin/xlim/ylim for each (region,variable)'
    )

    parser.add_argument(
        '--signal-mode',
        action='store_true',
        help='Ignore backgrounds/data; plot each WRAnalyzer_signal_*.root per mass point'
    )

    return parser.parse_args()

def setup_logging():
    """
    Configure root logger to INFO level using a simple default format.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def load_plot_settings(config_path: str) -> dict:
    """
    Load YAML configuration for rebin/xlim/ylim.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)

def setup_plotter(args) -> Plotter:
    """
    Create and configure a Plotter instance based on CLI args.
    - Sets run/year/lumi from ERA_CONFIG
    - Builds input_directory (list of Paths) and input_lumis (list of floats)
    - Defines sample_groups, regions_to_draw, and variables_to_draw
    """
    working_dir = Path('/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter')
    config = ERA_CONFIG[args.era]

    plotter = Plotter()
    plotter.era   = args.era
    plotter.run   = config["run"]
    plotter.year  = config["year"]
    plotter.lumi  = config["lumi"]  # total lumi (for CMS label)
    plotter.scale = True

    # ── BUILD A LIST OF INPUT DIRECTORIES ──
    # If era == "2022", use both Run3Summer22 and Run3Summer22EE
    if args.era == "2022":
        dir1 = working_dir / 'rootfiles' / plotter.run / plotter.year / 'Run3Summer22'
        dir2 = working_dir / 'rootfiles' / plotter.run / plotter.year / 'Run3Summer22EE'
        if args.dir:
            dir1 = dir1 / args.dir
            dir2 = dir2 / args.dir
        plotter.input_directory = [dir1, dir2]

        # Build a parallel list of lumis (same order as input_directory)
        lumi1 = ERA_CONFIG["Run3Summer22"]["lumi"]     # 7.9804
        lumi2 = ERA_CONFIG["Run3Summer22EE"]["lumi"]   # 26.6717
        plotter.input_lumis = [lumi1, lumi2]
    else:
        base = working_dir / 'rootfiles' / plotter.run / plotter.year / plotter.era
        if args.dir:
            base = base / args.dir
        plotter.input_directory = [base]

        # Single‐era case: use that era’s lumi
        plotter.input_lumis = [config["lumi"]]

    # ── OUTPUT DIRECTORY (unchanged) ──
    if args.dir:
        plotter.output_directory = Path(
            f"/eos/user/w/wijackso/{plotter.run}/{plotter.year}/{plotter.era}/{args.dir}"
        )
    else:
        plotter.output_directory = Path(
            f"/eos/user/w/wijackso/{plotter.run}/{plotter.year}/{plotter.era}"
        )

    # Define sample group ordering (DY, TTbar, etc.)
    sample_group_names = [
        f"SampleGroup_{args.era}_Other",
        f"SampleGroup_{args.era}_Nonprompt",
        f"SampleGroup_{args.era}_DY",
        f"SampleGroup_{args.era}_TTbar",
        f"SampleGroup_{args.era}_EGamma",
        f"SampleGroup_{args.era}_Muon",
    ]

    plotter.sample_groups = []
    for name in sample_group_names:
        try:
            sg = getattr(ps, name)
        except AttributeError:
            logging.error(f"SampleGroup '{name}' not found in predefined_samples.py")
            sys.exit(1)
        plotter.sample_groups.append(sg)

    # Define regions to draw
    plotter.regions_to_draw = [
        Region(
            'wr_mumu_resolved_sr',
            'Muon',
            unblind_data = True,
            tlatex_alias=f"$\mu\mu$\nResolved SR" #"$\mu\mu$\nResolved SR\n{args.era}\nDY w/ dijet mass reweight"
        ),
        Region(
            'wr_ee_resolved_sr',
            'EGamma',
            unblind_data = True,
            tlatex_alias=f"ee\nResolved SR" #"ee\nResolved SR\n{args.era}\nDY w/ dijet mass reweight"
        ),
    ]

    plotter.print_samples()
    plotter.print_regions()

    # Define Variables
    plotter.variables_to_draw = [
        Variable('mass_fourobject', r'$m_{lljj}$', 'GeV'),
#        Variable('pt_leading_jet', r'$p_{T}$ of the leading jet', 'GeV'),
#        Variable('mass_dijet', r'$m_{jj}$', 'GeV'),
#        Variable('pt_leading_lepton', r'$p_{T}$ of the leading lepton', 'GeV'),
#        Variable('eta_leading_lepton', r'$\eta$ of the leading lepton', ''),
#        Variable('phi_leading_lepton', r'$\phi$ of the leading lepton', ''),
#        Variable('pt_subleading_lepton', r'$p_{T}$ of the subleading lepton', 'GeV'),
#        Variable('eta_subleading_lepton', r'$\eta$ of the subleading lepton', ''),
#        Variable('phi_subleading_lepton', r'$\phi$ of the subleading lepton', ''),
#        Variable('eta_leading_jet', r'$\eta$ of the leading jet', ''),
#        Variable('phi_leading_jet', r'$\phi$ of the leading jet', ''),
#        Variable('pt_subleading_jet', r'$p_{T}$ of the subleading jet', 'GeV'),
#        Variable('eta_subleading_jet', r'$\eta$ of the subleading jet', ''),
#        Variable('phi_subleading_jet', r'$\phi$ of the subleading jet', ''),
#        Variable('mass_dilepton', r'$m_{ll}$', 'GeV'),
#        Variable('pt_dilepton', r'$p_{T}^{ll}$', 'GeV'),
#        Variable('pt_dijet', r'$p_{T}^{jj}$', 'GeV'),
#        Variable('mass_threeobject_leadlep', r'$m_{l_{\mathrm{pri}}jj}$', 'GeV'),
#        Variable('pt_threeobject_leadlep', r'$p^{T}_{l_{\mathrm{pri}}jj}$', 'GeV'),
#        Variable('mass_threeobject_subleadlep', r'$m_{l_{\mathrm{sec}}jj}$', 'GeV'),
#        Variable('pt_threeobject_subleadlep', r'$p^{T}_{l_{\mathrm{sec}}jj}$', 'GeV'),
#        Variable('pt_fourobject', r'$p^{T}_{lljj}$', 'GeV'),
    ]

    plotter.print_variables()

    return plotter

def load_and_rebin(
    input_dirs: list[Path],
    sample: str,
    hist_key: str,
    plotter: Plotter,
    is_data_group: bool
):
    """
    Try to open WRAnalyzer_{sample}.root in each directory of input_dirs,
    extract hist_key, rebin it, and—if this is MC—scale it by the corresponding
    lumi in plotter.input_lumis. Sum across sub-eras. Return the combined Hist, or None.
    """
    combined = None

    # Save original lumi so we can restore it afterward
    original_lumi = plotter.lumi

    for indir, sublumi in zip(plotter.input_directory, plotter.input_lumis):
        fp = indir / f"WRAnalyzer_{sample}.root"
        try:
            with uproot.open(fp) as f:
                raw_hist = f[hist_key].to_hist()
        except (FileNotFoundError, KeyError):
            # Skip if file missing or histogram not present
            continue

        # Rebin first
        if "mass_fourobject" in hist_key:
            variable_edges = [0, 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 8000]
            rebinned = plotter.rebin_hist(raw_hist, variable_edges)
        elif "pt_leading_jet" in hist_key:
            variable_edges = [0, 40, 100, 200, 400, 600, 800, 1000, 1500, 2000]
            rebinned = plotter.rebin_hist(raw_hist, variable_edges)
        else:
            rebinned = plotter.rebin_hist(raw_hist)

        if not is_data_group:
            # Only MC gets scaled by its sub-era lumi
            plotter.lumi = sublumi
            rebinned = plotter.scale_hist(rebinned)

        combined = rebinned if (combined is None) else (combined + rebinned)

    # Restore the original (total) lumi for CMS label
    plotter.lumi = original_lumi
    return combined

def reorder_legend(ax, priority_labels=("MC stat. unc.", "Data"), fontsize=FONT_SIZE_LEGEND):
    """
    Reorder the legend so that:
      1) 'MC stat. unc.' appears first
      2) 'Data' appears second
      3) All other backgrounds appear in reverse order
    """
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
    """
    Draws:
      - Top pad: stacked MC histograms + data points + MC stat band
      - Bottom pad: Data/MC ratio with Poisson‐based error bars on data
                    and a hatched MC stat‐uncertainty band centered at 1.
    """
    bkg_stack = plotter.stack_list
    bkg_labels = plotter.stack_labels
    bkg_colors = plotter.stack_colors

    tot = sum(plotter.stack_list)     # “total MC” histogram
    data = sum(plotter.data_hist)     # “data” histogram
    edges = tot.axes[0].edges

#    vals  = tot.values()      # array of MC yields, one entry per bin
#    vars  = tot.variances()   # array of per‐bin variances (None if unweighted)

    # Find the approximate bin‐index whose center is ~500 GeV:
#    bin_centers = 0.5*(edges[:-1] + edges[1:])
#    i500 = np.argmin(np.abs(bin_centers - 480.0))

#    print(f"480 GeV bin index = {i500}")
#    print("bin_center (GeV):", bin_centers[i500])
#    print(" tot_vals   =", vals[i500])
#    print(" tot_vars   =", (vars[i500] if vars is not None else "None"))
#    mc_err_i500 = np.sqrt(vars[i500]) if (vars is not None) else np.sqrt(vals[i500])
#    print(" σ_MC (abs) =", mc_err_i500)
#    print(" rel_err    =", mc_err_i500/vals[i500])

    # -- Top pad styling --
    hep.style.use("CMS")
    fig, (ax, rax) = plt.subplots(
        2, 1,
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1),
        sharex=True
    )

    # (1) Plot stacked MC (filled histograms)
    hep.histplot(
        bkg_stack,
        stack=True,
        label=bkg_labels,
        color=bkg_colors,
        histtype='fill',
        ax=ax
    )

    # (2) Plot data (points + Poisson error bars)
    hep.histplot(
        data,
        label="Data",
        color='k',
        histtype='errorbar',
        xerr=True,
        ax=ax
    )

    # (3) Over‐draw MC stat. uncertainty band on top pad
    errps = {'hatch': '////', 'facecolor': 'none', 'lw': 0, 'edgecolor': 'k', 'alpha': 0.5}
    hep.histplot(
        tot,
        histtype='band',
        ax=ax,
        **errps,
        label='MC stat. unc.'
    )

    # -- Compute Data/MC ratio and errors for the bottom pad --
    data_vals = data.values()        # array of bin contents for data
    tot_vals  = tot.values()         # array of bin contents for MC/stack
    ratio     = np.zeros_like(data_vals, dtype=float)
    ratio_err = np.zeros_like(data_vals, dtype=float)
    mask = tot_vals > 0

    if np.any(mask):
        # (a) Data/MC ratio and data‐only Poisson error
        ratio[mask]     = data_vals[mask] / tot_vals[mask]
        ratio_err[mask] = np.sqrt(data_vals[mask]) / tot_vals[mask]

    # (b) MC stat error: try to read tot.variances(), else fallback to √(tot_vals)
    tot_vars = tot.variances()  # returns array of per‐bin variances
    mc_err = np.sqrt(tot_vars)

    # Now form relative MC error = mc_err / tot_vals  (only where tot_vals > 0)
    rel_err = np.zeros_like(tot_vals, dtype=float)
    rel_err[mask] = mc_err[mask] / tot_vals[mask]

    # -- Bottom pad: draw data points with error bars --
    hep.histplot(
        ratio,
        edges,
        yerr=ratio_err,
        ax=rax,
        histtype="errorbar",
        color="k",
        capsize=4,
        label="Data",
        xerr=True
    )

    # -- Bottom pad: draw a *single* hatched MC‐uncertainty band around y=1.0 --
    band_low  = np.ones_like(rel_err) - rel_err
    band_high = np.ones_like(rel_err) + rel_err

    # Mask out zero‐MC bins so no patch is drawn there
    band_low[~mask] = np.nan
    band_high[~mask] = np.nan

    # Use stairs(...) with baseline=band_high to fill from band_high down to band_low
    rax.stairs(band_low, edges, baseline=band_high, **errps)

    # -- Axes formatting --
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylim(*plotter.ylim)

    ax.text(
        0.05, 0.96,
        region.tlatex_alias,
        transform=ax.transAxes,
        fontsize=FONT_SIZE_TITLE,
        verticalalignment='top'
    )

    hep.cms.label(
        loc=0,
        ax=ax,
        data=region.unblind_data,
        label="Work in Progress",
        lumi=f"{plotter.lumi:.1f}",
        com=13,
        fontsize=FONT_SIZE_LABEL
    )

    # Ratio pad labeling
    if variable.unit:
        xlabel = f"{variable.tlatex_alias} [{variable.unit}]"
    else:
        xlabel = variable.tlatex_alias

    rax.set_xlabel(xlabel)
    ax.set_xlabel("")  # no x‐label on the top pad
    rax.set_ylabel("Data/Sim.")
    rax.set_ylim(0, 2.8)
    rax.set_yticks([0, 1, 2])
    rax.axhline(1.0, ls='--', color='k')

    ax.set_xlim(*plotter.xlim)
    set_y_label(ax, tot, variable)

    reorder_legend(ax)
    return fig

def plot_signals(plotter, plot_settings):
    base = plotter.input_directory[0]
    for fp in sorted(base.glob("WRAnalyzer_signal_*.root")):
        masspoint = fp.stem.replace("WRAnalyzer_signal_", "")
        outdir = plotter.output_directory / masspoint
        # ... mkdir logic unchanged ...

        with uproot.open(fp) as f:
            for region in plotter.regions_to_draw:
                for variable in plotter.variables_to_draw:
                    hist_key = f"{region.name}/{variable.name}_{region.name}"
                    try:
                        raw = f[hist_key].to_hist()
                    except KeyError:
                        logging.warning(f"{masspoint}: missing {hist_key}")
                        continue

                    # --- load cfg up‐front ---
                    cfg = plot_settings[region.name][variable.name]
                    rebin_spec = cfg['rebin']

                    # --- now rebin with a valid spec ---
                    h = plotter.rebin_hist(raw, rebin_spec)

                    # --- scale & axes config as before ---
                    h = plotter.scale_hist(h)
                    plotter.configure_axes(
                        nrebin=cfg['rebin'],
                        xlim=tuple(map(float, cfg['xlim'])),
                        ylim=tuple(map(float, cfg['ylim']))
                    )

                    # draw single-histogram
                    fig, ax = plt.subplots()
                    hep.style.use("CMS")
                    hep.histplot(h, histtype='step', label=masspoint, ax=ax)
                    ax.set_yscale('log')

                    x_bins     = h.axes[0].edges
                    bin_widths = np.round(np.diff(x_bins), 1)
                    if np.all(bin_widths == bin_widths[0]):  # uniform binning
                        bw = bin_widths[0]
                        formatted = int(bw) if bw.is_integer() else f"{bw:.1f}"
                        ax.set_ylabel(f"Events / {formatted} {variable.unit}".strip())
                    else:                                    # variable bin widths
                        ax.set_ylabel(f"Events / bin {variable.unit}".strip())

                    # ─── continue with x-label, CMS label, legend, etc. ───
                    xlabel = variable.tlatex_alias + (f" [{variable.unit}]" if variable.unit else "")
                    ax.set_xlabel(xlabel)

#                    set_y_label(ax, tot, variable)
#                    ax.set_ylabel("Events")
                    ax.text(0.05,0.97, region.tlatex_alias,
                            transform=ax.transAxes, fontsize=FONT_SIZE_TITLE,
                            verticalalignment='top')
                    hep.cms.label(loc=0, ax=ax, data=False,
                                  label="Work in Progress",
                                  lumi=f"{plotter.lumi:.1f}",
                                  com=13, fontsize=FONT_SIZE_LABEL)
                    ax.set_xlim(*plotter.xlim)
                    ax.set_ylim(*plotter.ylim)
                    ax.legend(loc='best', fontsize=FONT_SIZE_LEGEND)

                    outname = outdir / f"{region.name}_{variable.name}.pdf"
                    save_figure(fig, outname)
                    plt.close(fig)

def main():
    args = parse_args()
    setup_logging()

    # Build & configure the Plotter
    plotter = setup_plotter(args)

    # If the user did not supply --plot-config, fill it in now:
    if args.plot_config is None:
        args.plot_config = f"data/{plotter.run}/{plotter.year}/{plotter.era}/sr_config.yaml"

    # Verify that the YAML config exists
    if not Path(args.plot_config).is_file():
        logging.error(f"Plot‐config YAML not found: {args.plot_config}")
        sys.exit(1)

    # Load YAML configurations
    plot_settings = load_plot_settings(args.plot_config)
    plotter.plot_settings = plot_settings

    if args.signal_mode:
        plot_signals(plotter, plot_settings)
        return

    # Validate YAML entries for each region
    missing_regions = [
        r.name for r in plotter.regions_to_draw
        if r.name not in plot_settings
    ]
    if missing_regions:
        logging.error(f"Missing YAML entries for regions: {missing_regions}")
        sys.exit(1)

    # Now a list of 1 or 2 input dirs
    input_dirs = plotter.input_directory

    for region in plotter.regions_to_draw:
        logging.info(f"Processing region '{region.name}'")
        for variable in plotter.variables_to_draw:
            logging.info(f"  Variable '{variable.name}'")

            if variable.name not in plot_settings[region.name]:
                logging.error(f"Missing '{variable.name}' under '{region.name}' in YAML")
                continue

            cfg = plotter.plot_settings[region.name][variable.name]
            rebin  = cfg['rebin']
            xmin, xmax = map(float, cfg['xlim'])
            ymin, ymax = map(float, cfg['ylim'])

            plotter.configure_axes(
                nrebin=rebin,
                xlim=(xmin, xmax),
                ylim=(ymin, ymax)
            )
            plotter.reset_stack()
            plotter.reset_data()

            hist_key = f"{region.name}/{variable.name}_{region.name}"

            for sample_group in plotter.sample_groups:
                combined = None

                is_data_group = (sample_group.name in DATA_GROUPS)
                # Only accept data from the region’s primary_dataset:
                if is_data_group and sample_group.name != region.primary_dataset:
                    # skip any data‐group that is not exactly the primary_dataset for this region
                    continue

                for sample in sample_group.samples:
                    print(sample)
                    hist_obj = load_and_rebin(
                        input_dirs, sample, hist_key, plotter, is_data_group
                    )
                    if hist_obj is None:
                        continue

                    combined = hist_obj if (combined is None) else (combined + hist_obj)

                if combined is None:
                    logging.warning(f"    No histograms found for group '{sample_group.name}'")
                    continue

                if is_data_group:
                    plotter.store_data(combined)
                else:
                    plotter.accumulate_histogram(
                        combined,
                        sample_group.color,
                        sample_group.tlatex_alias
                    )

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
