#!/usr/bin/env python3
"""Overlay the recreated Run2 (2018 ee) expected-limit band on the reference axes.

Reads the Stage-7 run2 xsec-limit table (pb), converts to fb and draws the
Brazil band + theory curve on axes matched to the 2018 reference plot
(run2_results/1D_EE_Combined_HalfN_Limit_vs_WR.pdf): x = 800-6000 GeV,
log y = 1e-4..1e4 fb. The reference is the COMBINED (resolved+boosted) ee
channel; this band is resolved-only ee with the DY(LO HT reshaped)+tt_tW+
Nonprompt+Other MC background, so a gap of a few x is expected -- the point of
the overlay is the shape and mass dependence, judged side by side.

  python compare_run2_reference.py [--function expo]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                        # repo root

from wrplotter.plotting_helpers import custom_log_formatter     # noqa: E402

plt.style.use(hep.style.CMS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--function", default="expo")
    p.add_argument("--table", type=Path,
                   default=HERE / "run2" / "xsec_limit_table_ee_resolved.csv")
    p.add_argument("--output-dir", type=Path, default=HERE / "run2" / "comparison")
    args = p.parse_args()

    rows = [r for r in csv.DictReader(open(args.table))
            if r["function"] == args.function]
    rows.sort(key=lambda r: float(r["mWR"]))
    m = [float(r["mWR"]) for r in rows]
    fb = lambda key: [1000.0 * float(r[key]) for r in rows]   # pb -> fb
    med, m1, p1, m2, p2 = (fb(k) for k in
                           ("sigma_ul_med", "sigma_ul_m1s", "sigma_ul_p1s",
                            "sigma_ul_m2s", "sigma_ul_p2s"))
    theory = fb("xsec_pb")

    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.fill_between(m, m2, p2, color="#ffcc00", label="95% expected")
    ax.fill_between(m, m1, p1, color="#00cc00", label="68% expected")
    ax.plot(m, med, "k--", lw=2, label="Expected limit (this pipeline)")
    ax.plot(m, theory, "r-", lw=1.5, label=r"Theory ($g_R=g_L$, sample $\sigma$)")

    ax.set_yscale("log")
    ax.set_xlim(800, 6000)
    ax.set_ylim(1e-4, 1e4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_xlabel(r"$m_{W_R}$ (GeV)")
    ax.set_ylabel(r"$\sigma(pp \to W_R)\,\mathcal{B}(W_R \to eeq\bar{q}\,')$ (fb)")
    ax.text(0.60, 0.93, r"$\mathbf{m_N = m_{W_R}/2}$", transform=ax.transAxes,
            va="top", fontsize=19)
    ax.text(0.60, 0.87, "ee resolved (recreated)", transform=ax.transAxes,
            va="top", fontsize=19, weight="bold")
    ax.text(0.60, 0.80, f"bkg fit: {args.function}", transform=ax.transAxes,
            va="top", fontsize=14, color="grey")
    ax.legend(loc="lower left", fontsize=16, frameon=False)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi="59.8", com=13, fontsize=18)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"run2_ee_halfN_vs_reference_{args.function}"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {out}.pdf/.png")


if __name__ == "__main__":
    main()
