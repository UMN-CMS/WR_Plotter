#!/usr/bin/env python3
"""Theory uncertainties of the signal yield against m_WR (AN-19-083 Fig. 129): the relative change of
each signal region's yield under the QCD scale (envelope of muR, muF, muR+muF), the PDF
eigenvectors, alpha_s and the parton-shower ISR / FSR variations, for the mass points of a
signal set made with ``run_analysis.py ... Signal --systs scale pdf ps``.

The variations are read exactly as for the stack-plot band and the systematics tables
(wrplotter/systematics.py, wrplotter/pdf.py): the scale and PS weights are normalized per
dataset by the analyzer, so only the acceptance moves; the PDF and alpha_s shifts are formed
from the stored members at the analysis binning.

    <plots>/<Run>/<Year>/<Era>/<dir>/signal_theory/<topology>_<channel>_<line>.pdf
        up / down change (%) of the SR yield vs m_WR for the points of one m_N line:
        ``mN100`` (the lightest m_N at each m_WR) and ``mNhalf`` (m_N closest to m_WR / 2)
    tables/<Run>/<Year>/<Era>/<dir>/signal_theory/theory_uncertainties.csv
        every point, region and source: nominal yield, up and down change (%)

Example:
    python bin/plot_signal_theory_uncertainties.py --era RunIII2024Summer24 --dir 20261012_signal_theory --local-plots
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wrplotter.cli_utils import add_era_args, add_output_args, setup_logging  # noqa: E402
from wrplotter.config import load_lumi  # noqa: E402
from wrplotter.inputs import make_inputs, mass_point  # noqa: E402
from wrplotter.paths import plots_base, repo_root, save_figure  # noqa: E402
from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header  # noqa: E402
from wrplotter.regions import regions_for_era  # noqa: E402
from wrplotter.signal_compare import CHANNEL_LABELS, CHANNELS, COLORS, MARKERS, MASS_VARIABLES, TOPOLOGIES, TOPOLOGY_LABELS  # noqa: E402
from wrplotter.stack import Binning, HistLoader  # noqa: E402
from wrplotter.syst_tables import _process_variations  # noqa: E402

SOURCES = {"scale": "QCD scale (envelope)", "pdf": "PDF", "alphas": r"$\alpha_S$", "psisr": "PS ISR", "psfsr": "PS FSR"}
LINES = {"mN100": r"lightest $m_N$ at each $m_{W_R}$", "mNhalf": r"$m_N \approx m_{W_R} / 2$"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_era_args(parser)
    add_output_args(parser)
    parser.add_argument("-o", "--out", default=None, help="Write plots and the table under this directory instead.")
    return parser.parse_args(argv)


def yield_of(h) -> float:
    return float(h.values(flow=True).sum())


def line_points(points: list[str]) -> dict[str, list[str]]:
    by_wr: dict[int, list[str]] = {}
    for p in sorted(points, key=mass_point):
        by_wr.setdefault(mass_point(p)[0], []).append(p)
    lightest = [min(pts, key=lambda p: mass_point(p)[1]) for pts in by_wr.values()]
    half = [min(pts, key=lambda p: abs(mass_point(p)[1] / mass_point(p)[0] - 0.5)) for pts in by_wr.values()]
    return {"mN100": lightest, "mNhalf": half}


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    info = load_lumi(args.era)
    try:
        (inputs,) = make_inputs(args.era, args.dir, with_dy=False)
    except (FileNotFoundError, ValueError) as exc:
        logging.error(str(exc))
        return 2
    points = inputs.signal_points()
    if not points:
        logging.error("no signal files in %s", ", ".join(str(d) for d in inputs.dirs))
        return 2
    regions = {r.name: r for r in regions_for_era(args.era)}
    load = HistLoader(inputs, Binning())
    com = float(info.get("com", 13.6))

    rows = []
    changes: dict[tuple[str, str, str, str], tuple[float, float] | None] = {}
    for point in points:
        for t in TOPOLOGIES:
            for ch in CHANNELS:
                region = regions[f"wr_{ch}_{t}_sr"]
                nominal, variations = _process_variations(load, [f"signal_{point}"], region, MASS_VARIABLES[t],
                                                          list(SOURCES))
                if nominal is None:
                    continue
                n = yield_of(nominal)
                for source in SOURCES:
                    dirs = variations.get(source)
                    if not dirs or n <= 0:
                        changes[(point, t, ch, source)] = None
                        rows.append({"point": point, "topology": t, "channel": ch, "source": source,
                                     "nominal": f"{n:.6g}", "up_pct": "", "down_pct": ""})
                        continue
                    up = 100 * (yield_of(dirs["up"]) - n) / n
                    down = 100 * (yield_of(dirs["down"]) - n) / n
                    changes[(point, t, ch, source)] = (up, down)
                    rows.append({"point": point, "topology": t, "channel": ch, "source": source,
                                 "nominal": f"{n:.6g}", "up_pct": f"{up:.3f}", "down_pct": f"{down:.3f}"})
    if args.out:
        plots, tables = Path(args.out) / "plots", Path(args.out) / "tables"
    else:
        plots = plots_base(info["run"], info["year"], args.era, args.dir or None, local=args.local_plots) / "signal_theory"
        tables = repo_root() / "tables" / info["run"] / info["year"] / args.era / (args.dir or "") / "signal_theory"
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "theory_uncertainties.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["point", "topology", "channel", "source", "nominal", "up_pct", "down_pct"])
        w.writeheader()
        w.writerows(rows)
    logging.info("table: %s (%d rows)", tables / "theory_uncertainties.csv", len(rows))

    n_saved = 0
    for line, pts in line_points(points).items():
        for t in TOPOLOGIES:
            for ch in CHANNELS:
                hep.style.use("CMS")
                fig, ax = plt.subplots(figsize=(10, 9))
                drawn = False
                for i, (source, label) in enumerate(SOURCES.items()):
                    xs, ups, downs = [], [], []
                    for p in pts:
                        c = changes.get((p, t, ch, source))
                        if c is None or not all(math.isfinite(v) for v in c):
                            continue
                        xs.append(mass_point(p)[0])
                        ups.append(c[0])
                        downs.append(c[1])
                    if not xs:
                        continue
                    drawn = True
                    ax.plot(xs, ups, color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)], markersize=6,
                            lw=1.5, label=label)
                    ax.plot(xs, downs, color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)], markersize=6,
                            lw=1.5, ls="--", mfc="white")
                if not drawn:
                    plt.close(fig)
                    continue
                ax.axhline(0, color="grey", lw=1)
                lo, hi = ax.get_ylim()
                span = max(hi - lo, 1.0)
                ax.set_ylim(lo - 0.1 * span, hi + 0.6 * span)
                ax.set_xlabel(r"$m_{W_R}$ [GeV]")
                ax.set_ylabel("Change of the SR yield [%]")
                ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND - 4, title="solid: up, dashed: down",
                          title_fontsize=FONT_SIZE_LEGEND - 4)
                cms_label(ax, None, data=False, rlabel=f"({com:g} TeV)",
                          text=f"{CHANNEL_LABELS[ch]}, {TOPOLOGY_LABELS[t]}\n{LINES[line]}\n"
                               "Acceptance only (weights normalized per sample)")
                fit_header(ax)
                path = plots / f"{t}_{ch}_{line}.pdf"
                try:
                    save_figure(fig, path)
                    n_saved += 1
                except Exception as exc:  # EOS upload errors must not stop the batch
                    logging.error("  failed to save %s: %s", path, exc)
                finally:
                    plt.close(fig)
    logging.info("%d plots under %s", n_saved, plots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
