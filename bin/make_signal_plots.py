#!/usr/bin/env python3
"""Signal plots of one era: acceptance x efficiency, cross sections, mass shapes and resolution,
kinematics and cutflows of every (m_WR, m_N) point the analysis set holds.  The single-era
counterpart of compare_signal_eras.py (definitions in wrplotter/signal_compare.py and
wrplotter/signal_plots.py).  Only the signal files are read, so a signal-only directory works.

    <plots>/<Run>/<Year>/<Era>/<dir>/signal_plots/
        efficiency/<channel>_WR<m>.pdf            A x eps vs m_N per m_WR: resolved SR, boosted SR, combined
                                                  (AN-19-083 App. B)
        efficiency_map/<topology>_<channel>.pdf   A x eps over the (m_WR, m_N) grid
        cr_fraction/<topology>_<channel>_WR<m>.pdf signal in the DY and flavour CRs over sigma L, vs m_N
        xsec/<channel>_WR<m>.pdf, kfactor/...     sigma_LO and sigma_LO x k (fb), and k, vs m_N (AN Fig. 6)
        mass/<topology>_<channel>_WR<m>.pdf       SR mass of several m_N at one m_WR, unit area
        mass/<topology>_<channel>_mN_<frac>.pdf   SR mass of every m_WR at m_N ~ frac x m_WR (frac = 0.05, 0.5)
        resolution/<topology>_<channel>_WR<m>.pdf median / m_WR and sigma_eff / median of the peak vs m_N
        kinematics/<region>/<variable>.pdf        the other SR variables of the --points, unit area
        cutflow/<topology>_<channel>_WR<m>.pdf    cumulative efficiency after each cut of several m_N at one m_WR
        cutflow/<topology>_<channel>.pdf          the same for the --points
    tables/<Run>/<Year>/<Era>/<dir>/signal_plots/
        efficiency.csv                            per point and channel: A x eps (resolved, boosted, combined),
                                                  the CR fractions, the peak median and sigma_eff
        latex/signal_eff_<channel>_WR<m>.tex      A x eps tables per m_WR, with preview.tex

--points picks the points of the kinematics and cutflow plots (default: the cutflow_signals of
data/lumi.yaml; "all" = every point); --max-shapes limits the curves of each mass plot.

Examples:
    python bin/make_signal_plots.py --era RunIII2024Summer24 --dir 20261012_skims_v2 --local-plots
    python bin/make_signal_plots.py --era RunIII2024Summer24 --dir 20261012_skims_v2 --points WR4000_N2100 WR6000_N100 --local-plots
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wrplotter.cli_utils import add_era_args, add_output_args, parse_multi, setup_logging  # noqa: E402
from wrplotter.config import default_signals, load_lumi  # noqa: E402
from wrplotter.inputs import mass_point  # noqa: E402
from wrplotter.paths import repo_root, save_figure, plots_base  # noqa: E402
from wrplotter.signal_compare import (  # noqa: E402
    CHANNELS, MASS_VARIABLES, TOPOLOGIES, cutflow_efficiency, load_point_hist, set_points, signal_config_path,
)
from wrplotter.signal_plots import (  # noqa: E402
    KINEMATIC_VARIABLES, VARIABLE_LABELS, cross_sections, efficiency_latex, efficiency_rows, label_text,
    load_set, nearest_fraction, plot_cr_fraction_vs_mn, plot_cutflows, plot_efficiency_map,
    plot_efficiency_vs_mn, plot_resolution_vs_mn, plot_shapes, plot_xsec_vs_mn, point_label, points_by_mwr,
    spread, summarize_point, write_csv,
)
from wrplotter.stack import hist_key  # noqa: E402

SHAPE_FRACTIONS = (0.05, 0.5)      # m_N / m_WR lines of the m_WR scans


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_era_args(parser)
    add_output_args(parser)
    parser.add_argument("--points", nargs="*", default=None,
                        help="Points of the kinematics and cutflow plots (WR<m>_N<m>, 'all'); default: the "
                             "cutflow_signals of data/lumi.yaml.")
    parser.add_argument("--max-shapes", type=int, default=6, help="Curves per mass plot (default 6).")
    parser.add_argument("--analyzer-root", default=None, help="WrCoffea checkout holding data/configs (default: parent of this repo).")
    parser.add_argument("--fileset", default=None, help="Fileset the analyzer ran with, for the cross sections.")
    parser.add_argument("--no-kinematics", action="store_true", help="Skip the kinematics and cutflow plots.")
    parser.add_argument("--no-pdf", action="store_true", help="Do not compile the LaTeX preview.")
    parser.add_argument("-o", "--out", default=None, help="Write plots and tables under this directory instead.")
    args = parser.parse_args(argv)
    args.points = parse_multi(args.points)
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


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    info = load_lumi(args.era)
    try:
        era_set = load_set(args.era, args.dir)
    except (FileNotFoundError, KeyError) as exc:
        logging.error(str(exc))
        return 2
    points = set_points(era_set)
    if not points:
        logging.error("no signal files in %s", ", ".join(str(d) for d in era_set.inputs.dirs))
        return 2
    by_wr = points_by_mwr(points)
    logging.info("%s / %s: %d signal points, m_WR %s", args.era, args.dir, len(points),
                 ", ".join(str(m) for m in by_wr))

    if args.out:
        plots = Path(args.out) / "plots"
        tables = Path(args.out) / "tables"
    else:
        plots = plots_base(info["run"], info["year"], args.era, args.dir or None, local=args.local_plots) / "signal_plots"
        tables = repo_root() / "tables" / info["run"] / info["year"] / args.era / (args.dir or "") / "signal_plots"
    tables.mkdir(parents=True, exist_ok=True)
    n_saved = 0

    # ---- efficiencies, CR fractions, peak summaries of every point
    summaries = {}
    fileset = Path(args.fileset).expanduser() if args.fileset else None
    for p in points:
        try:
            summaries[p] = summarize_point(era_set, p, analyzer_root=args.analyzer_root, fileset=fileset)
        except ValueError as exc:
            logging.warning("%s: %s", p, exc)
    rows = efficiency_rows(era_set, [summaries[p] for p in points if p in summaries])
    write_csv(rows, tables / "efficiency.csv")
    logging.info("efficiency table: %s (%d rows)", tables / "efficiency.csv", len(rows))

    latex_dir = tables / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    tex_files = []
    for m_wr, pts in by_wr.items():
        summ = [summaries[p] for p in pts if p in summaries]
        if not summ:
            continue
        for ch in CHANNELS:
            if not any(s.produced.channels[ch] > 0 for s in summ):
                continue
            n_saved += save(plot_efficiency_vs_mn(era_set, summ, m_wr, ch), plots / "efficiency" / f"{ch}_WR{m_wr}.pdf")
            for t in TOPOLOGIES:
                n_saved += save(plot_cr_fraction_vs_mn(era_set, summ, m_wr, ch, t),
                                plots / "cr_fraction" / f"{t}_{ch}_WR{m_wr}.pdf")
                n_saved += save(plot_resolution_vs_mn(era_set, summ, m_wr, t, ch),
                                plots / "resolution" / f"{t}_{ch}_WR{m_wr}.pdf")
            tex = latex_dir / f"signal_eff_{ch}_WR{m_wr}.tex"
            tex.write_text(efficiency_latex(era_set, summ, m_wr, ch))
            tex_files.append(tex.name)
    for t in (*TOPOLOGIES, "combined"):
        for ch in CHANNELS:
            summ = [summaries[p] for p in points if p in summaries]
            n_saved += save(plot_efficiency_map(era_set, summ, t, ch), plots / "efficiency_map" / f"{t}_{ch}.pdf")
    if tex_files:
        preview = ["\\documentclass{article}", "\\usepackage{geometry,amsmath}", "\\geometry{margin=1.5cm}",
                   "\\newcommand{\\topcaption}[1]{\\caption{#1}}", "\\newcommand{\\cmsTable}[1]{#1}",
                   "\\newcommand{\\GeV}{\\,\\mathrm{GeV}}", "\\newcommand{\\fbinv}{\\,\\mathrm{fb}^{-1}}",
                   "\\begin{document}"] + [f"\\input{{{name}}}" for name in tex_files] + ["\\end{document}", ""]
        (latex_dir / "preview.tex").write_text("\n".join(preview))
        if not args.no_pdf and shutil.which("pdflatex"):
            subprocess.run(["pdflatex", "-interaction=batchmode", "preview.tex"], cwd=latex_dir,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    # ---- cross sections and k-factors from the analyzer's signal config
    try:
        config = fileset or signal_config_path(args.era, args.analyzer_root)
        xsecs = cross_sections(config, era_set, points)
    except (FileNotFoundError, KeyError) as exc:
        logging.warning("no cross sections: %s", exc)
        xsecs = {}
    for m_wr, pts in by_wr.items():
        for ch in CHANNELS:
            if not any(p in xsecs and ch in xsecs[p] for p in pts):
                continue
            n_saved += save(plot_xsec_vs_mn(era_set, xsecs, m_wr, pts, ch, "xsec"), plots / "xsec" / f"{ch}_WR{m_wr}.pdf")
            n_saved += save(plot_xsec_vs_mn(era_set, xsecs, m_wr, pts, ch, "k"), plots / "kfactor" / f"{ch}_WR{m_wr}.pdf")

    # ---- signal-region mass shapes
    for t in TOPOLOGIES:
        var = MASS_VARIABLES[t]
        for ch in CHANNELS:
            region = f"wr_{ch}_{t}_sr"
            for m_wr, pts in by_wr.items():
                entries = []
                for p in spread(pts, args.max_shapes):
                    h = load_point_hist(era_set, p, hist_key(region, var))
                    if h is not None and float(h.values(flow=True).sum()) > 0:
                        entries.append((rf"$m_N$ = {mass_point(p)[1]} GeV", h))
                if entries:
                    fig = plot_shapes(era_set, entries, xlabel=VARIABLE_LABELS[var],
                                      text=label_text(ch, topology=t, m_wr=m_wr), xlim=(0, min(1.3 * m_wr, 8000)))
                    n_saved += save(fig, plots / "mass" / f"{t}_{ch}_WR{m_wr}.pdf")
            for frac in SHAPE_FRACTIONS:
                entries = []
                for m_wr, pts in by_wr.items():
                    p = nearest_fraction(pts, frac)
                    h = load_point_hist(era_set, p, hist_key(region, var)) if p else None
                    if h is not None and float(h.values(flow=True).sum()) > 0:
                        entries.append((point_label(p), h))
                if entries:
                    fig = plot_shapes(era_set, entries, xlabel=VARIABLE_LABELS[var],
                                      text=label_text(ch, topology=t) + rf"$, m_N \approx {frac:g}\,m_{{W_R}}$")
                    n_saved += save(fig, plots / "mass" / f"{t}_{ch}_mN_{frac:g}.pdf")

    # ---- cutflows: several m_N at each m_WR
    for m_wr, pts in by_wr.items():
        for t in TOPOLOGIES:
            for ch in CHANNELS:
                cf_entries = []
                for p in spread(pts, args.max_shapes):
                    if p not in summaries:
                        continue
                    cf = cutflow_efficiency(era_set, p, t, ch, summaries[p].produced)
                    if cf is not None:
                        cf_entries.append((rf"$m_N$ = {mass_point(p)[1]} GeV", cf))
                if cf_entries:
                    n_saved += save(plot_cutflows(era_set, cf_entries, topology=t, channel=ch, m_wr=m_wr),
                                    plots / "cutflow" / f"{t}_{ch}_WR{m_wr}.pdf")

    # ---- kinematics and cutflows of a few points
    if not args.no_kinematics:
        if args.points and args.points != ["all"]:
            chosen = [p for p in args.points if p in points]
            missing = [p for p in args.points if p not in points]
            if missing:
                logging.warning("--points not in the set: %s", ", ".join(missing))
        elif args.points == ["all"]:
            chosen = points
        else:
            chosen = []
            for boosted in (False, True):
                chosen += [p for p in default_signals(info, boosted=boosted, for_cutflow=True) if p in points]
            chosen = list(dict.fromkeys(chosen))
        for t in TOPOLOGIES:
            for ch in CHANNELS:
                region = f"wr_{ch}_{t}_sr"
                for var in KINEMATIC_VARIABLES[t] + [MASS_VARIABLES[t]]:
                    entries = []
                    for p in chosen:
                        h = load_point_hist(era_set, p, hist_key(region, var))
                        if h is not None and float(h.values(flow=True).sum()) > 0:
                            entries.append((point_label(p), h))
                    if entries:
                        fig = plot_shapes(era_set, entries, xlabel=VARIABLE_LABELS.get(var, var),
                                          text=label_text(ch, topology=t))
                        n_saved += save(fig, plots / "kinematics" / region / f"{var}.pdf")
                cf_entries = []
                for p in chosen:
                    if p not in summaries:
                        continue
                    cf = cutflow_efficiency(era_set, p, t, ch, summaries[p].produced)
                    if cf is not None:
                        cf_entries.append((point_label(p), cf))
                if cf_entries:
                    n_saved += save(plot_cutflows(era_set, cf_entries, topology=t, channel=ch),
                                    plots / "cutflow" / f"{t}_{ch}.pdf")
    logging.info("%d plots under %s; tables under %s", n_saved, plots, tables)
    return 0


if __name__ == "__main__":
    sys.exit(main())
