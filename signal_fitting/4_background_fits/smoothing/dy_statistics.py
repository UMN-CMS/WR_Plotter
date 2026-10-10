#!/usr/bin/env python3
"""How many MC events does each DY option put in the signal region?

For every DY output file (one per DY option): the generated events of its DY
datasets (fileset metadata), the MC events in the SR (event tree when the file
has one, else the unweighted cutflow), the yield, and
N_eff = (sum w)^2 / sum w^2 -- the number of equal-weight events that would
give the same MC error bar -- in mass ranges.  With an event tree also the
largest single event weight per range.

  PYTHONPATH=/opt/homebrew/lib/root ../../../.venv-fitting/bin/python dy_statistics.py \\
      --dy lo_inc=<WRAnalyzer_DYJets.root> --dy lo_ht=<...> --dy nlo_mll=<...>
  -> run3/dy_options/dy_statistics_<region>.csv
     run3/dy_options/plots/dy_options_<region>.{png,pdf}   (yield and N_eff per 50 GeV)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from smooth_lib import centers_of, rebin                                  # noqa: E402
from smooth_background import _cms, _save                                 # noqa: E402
from wrplotter.cli_utils import setup_logging                            # noqa: E402
from wrplotter.config import load_lumi                                   # noqa: E402

logger = logging.getLogger("dy_statistics")

ERA = "RunIII2024Summer24"
ROOTFILES = HERE.parents[2] / "rootfiles" / "Run3" / "2024" / ERA
FILESETS = HERE.parents[3] / "data" / "filesets" / "Run3" / "2024" / ERA / "skimmed"
LABEL = {"lo_inc": "LO inclusive", "lo_ht": "LO HT-binned", "nlo_inc": "NLO inclusive",
         "nlo_njet": "NLO jet-binned", "nlo_mll": "NLO m(ll)-binned",
         "nlo_ptll": "NLO pT(ll)-binned"}
# fixed per option (CMS palette), so a subset keeps its colors
COLOR = {"lo_inc": "#5790fc", "lo_ht": "#f89c20", "nlo_inc": "#e42536",
         "nlo_njet": "#964a8b", "nlo_mll": "#9c9ca1", "nlo_ptll": "#7a21dd"}
DEFAULT_DY = [
    f"lo_inc={ROOTFILES / '20261008_skims_v2_lo_inc_dy' / 'WRAnalyzer_DYJets.root'}",
    f"lo_ht={ROOTFILES / '20261008_split_events' / 'WRAnalyzer_DYJets.root'}",
    f"nlo_inc={ROOTFILES / '20261008_skims_v2_nlo_inc_dy' / 'WRAnalyzer_DYJets.root'}",
    f"nlo_njet={ROOTFILES / '20261008_skims_v2_nlo_njet_dy' / 'WRAnalyzer_DYJets.root'}",
    f"nlo_mll={ROOTFILES / '20261008_split_events_nlo_mll_dy' / 'WRAnalyzer_DYJets.root'}",
]
RANGES = [(800, 2000), (2000, 3000), (3000, 4000), (4000, 8000)]


def generated_events(tag):
    """Generated (genEventCount) and skimmed events summed over the DY datasets."""
    path = FILESETS / f"{ERA}_mc_dy_{tag}_fileset.json"
    if not path.exists():
        return None, None, 0
    meta = [v["metadata"] for v in json.load(open(path)).values()
            if v["metadata"].get("physics_group") == "DYJets"]
    return (sum(m.get("genEventCount") or 0 for m in meta),
            sum(m.get("nevts") or 0 for m in meta), len(meta))


def read_option(path, region):
    """SR histogram (native bins), cutflow count and, if present, the event tree."""
    f = uproot.open(path)
    h = f[f"{region}/mass_fourobject_{region}"]
    flavour = region.split("_")[1]
    cut = f[f"cutflow_resolved/{flavour}/cumulative_unweighted"]
    n_cutflow = int(round(cut.values()[-1]))
    events = None
    if f"events_{region}" in {k.split(";")[0] for k in f.keys()}:
        t = f[f"events_{region}"].arrays(["mass", "weight"], library="np")
        events = (t["mass"], t["weight"])
    return h.axis().edges(), h.values(), h.variances(), n_cutflow, events


def range_stats(edges, values, variances, events, lo, hi):
    sel = (edges[:-1] >= lo) & (edges[:-1] < hi)
    sw, sw2 = values[sel].sum(), variances[sel].sum()
    row = {"yield": sw, "neff": sw * sw / sw2 if sw2 > 0 else 0.0}
    if events is not None:
        m, w = events
        s = (m >= lo) & (m < hi)
        row["events"] = int(s.sum())
        row["max_w"] = float(w[s].max()) if s.any() else 0.0
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dy", action="append", default=None, metavar="TAG=FILE",
                   help="DY option and its WRAnalyzer_DYJets.root (repeatable); "
                        "default: the 2024 skims_v2 sets of 2026-10-08")
    p.add_argument("--region", default="wr_ee_resolved_sr")
    p.add_argument("--bw", type=float, default=50.0)
    p.add_argument("--output-dir", type=Path, default=HERE / "run3" / "dy_options")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    options = []
    for spec in args.dy or DEFAULT_DY:
        tag, path = spec.split("=", 1)
        if not Path(path).exists():
            logger.warning("skipping %s: %s not found", tag, path)
            continue
        options.append((tag, Path(path)))

    rows, curves = [], []
    for tag, path in options:
        edges, values, variances, n_cutflow, events = read_option(path, args.region)
        n_gen, n_skim, n_datasets = generated_events(tag)
        row = {"option": tag, "file": str(path.relative_to(ROOTFILES)) if ROOTFILES in path.parents
               else str(path), "datasets": n_datasets, "generated": n_gen, "skimmed": n_skim,
               "sr_events": len(events[0]) if events is not None else n_cutflow,
               "sr_events_source": "tree" if events is not None else "cutflow"}
        tot = range_stats(edges, values, variances, events, 800, 8000)
        row.update({"sr_yield": tot["yield"], "sr_neff": tot["neff"]})
        for lo, hi in RANGES:
            r = range_stats(edges, values, variances, events, lo, hi)
            for k, v in r.items():
                row[f"{k}_{lo}_{hi}"] = v
        if events is not None and len(events[0]) != n_cutflow:
            logger.warning("%s: tree has %d events, cutflow %d", tag, len(events[0]), n_cutflow)
        rows.append(row)
        curves.append((tag, edges, values, variances))

    # ---- table
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / f"dy_statistics_{args.region}.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s", out_csv)

    head = f"{'option':18s} {'generated':>10s} {'SR events':>10s} {'SR yield':>9s} {'N_eff':>8s}"
    head += "".join(f"  {'N_eff ' + (f'{lo/1e3:g}-{hi/1e3:g}' if hi < 8000 else f'>{lo/1e3:g}') + ' TeV':>17s}"
                    for lo, hi in RANGES)
    print(head)
    for r in rows:
        line = (f"{LABEL.get(r['option'], r['option']):18s} {r['generated'] / 1e6 if r['generated'] else 0:9.0f}M "
                f"{r['sr_events']:10d} {r['sr_yield']:9.1f} {r['sr_neff']:8.0f}")
        line += "".join(f"  {r[f'neff_{lo}_{hi}']:17.1f}" for lo, hi in RANGES)
        print(line)

    # ---- plot: yield and N_eff per bin
    info = load_lumi(ERA)
    hep.style.use("CMS")
    fig, (ax, axn) = plt.subplots(2, 1, sharex=True, figsize=(10, 10),
                                  gridspec_kw={"height_ratios": [1, 1], "hspace": 0.06})
    f = int(round(args.bw / (curves[0][1][1] - curves[0][1][0])))
    for tag, edges, values, variances in curves:
        e5, v5, var5 = rebin(edges, values, variances, f)
        c5 = centers_of(e5)
        sel = e5[:-1] >= 800
        line = ax.stairs(np.where(sel, v5, np.nan), e5, label=LABEL.get(tag, tag), linewidth=1.6,
                         color=COLOR.get(tag))
        col = line.get_edgecolor()
        err = np.sqrt(var5)
        ax.fill_between(e5, np.append(v5 - err, 0), np.append(v5 + err, 0), step="post",
                        where=np.append(sel, False), color=col, alpha=0.18, linewidth=0)
        neff = np.divide(v5 * v5, var5, out=np.zeros_like(v5), where=var5 > 0)
        axn.stairs(np.where(sel & (neff > 0), neff, np.nan), e5, color=col, linewidth=1.6)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 5e3)
    ax.set_ylabel(f"DY events / {args.bw:.0f} GeV")
    ax.legend(loc="upper right", fontsize=15)
    axn.set_yscale("log")
    axn.set_ylim(0.3, 3e4)
    axn.set_ylabel(f"N_eff / {args.bw:.0f} GeV")
    axn.set_xlim(800, 6000)
    axn.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    flav = args.region.split("_")[1]
    ax.text(0.03, 0.05, f"{flav} resolved SR, DY only\nband = MC stat. error",
            transform=ax.transAxes, fontsize=14)
    axn.text(0.03, 0.06, r"N_eff = $(\Sigma w)^2 / \Sigma w^2$", transform=axn.transAxes, fontsize=14)
    _cms(ax, info["lumi"], info.get("com", 13.6))
    _save(fig, args.output_dir / "plots" / f"dy_options_{args.region}")


if __name__ == "__main__":
    main()
