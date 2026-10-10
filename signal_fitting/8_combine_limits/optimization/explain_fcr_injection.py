#!/usr/bin/env python3
"""Signal-injection linearity of the TWO-REGION fit (fcr_shape), the seminar's
"does the model detect an injected signal?" plot.

For each injected yield r_inj: N Poisson toys of BOTH regions, with truth
  SR: the pseudo-data of explain_fcr --tt-to-cr (simulation, tt scaled to the
      e-mu count) + r_inj * S_i (the signal Gaussian of the fit)
  CR: the e-mu data histogram (as card_model_toys --cr-shape data)
each refit with r floating. Plotted: mean fitted r vs r_inj with the RMS of the
single fits as error bars, the diagonal, and below the bias (mean - injected)
with the error of the mean.

  WR_SEMINAR_ERA=2024 python explain_fcr_injection.py --ntoys 10000
"""
from __future__ import annotations

import argparse
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
from explain_fcr import (ERA, MASS, VAR, INPUTS, OUT_DEFAULT, _cmslabel, _save,  # noqa: E402
                         setup_logging, load_lumi, rebin_snap, CardSB, pseudo_data_sr)

logger = logging.getLogger("explain_fcr_injection")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inject", type=float, nargs="+",
                   default=[0, 10, 20, 32, 50, 75, 100, 150])
    p.add_argument("--ntoys", type=int, default=10000)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--figsize", type=float, nargs=2, default=(10.0, 10.1))
    p.add_argument("--replot", action="store_true",
                   help="redraw from the saved injection_linearity.json in --output-dir (no toys)")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    info = load_lumi(ERA)
    lumi, com = info["lumi"], info.get("com", 13.6)

    out = args.output_dir / "injection_linearity"
    if args.replot:
        saved = json.load(open(out.with_suffix(".json")))
        rows, ntoys = saved["points"], saved["ntoys"]
        window, mu_tt = saved["window"], saved["mu_tt"]
    else:
        meta = json.load(open(INPUTS / "ee_resolved.json"))["masses"][str(MASS)]
        fcr_all = json.load(open(INPUTS / "ee_resolved_fcr.json"))
        f = fcr_all["masses"][str(MASS)]["vars"][VAR]
        v = meta["vars"][VAR]
        edges, mu_sr, mu_tt = pseudo_data_sr(INPUTS, v, f, tt_to_cr=True)
        hcr = uproot.open(INPUTS / "ee_resolved_fcr.root")

        def snap(h):
            _, x = rebin_snap(h.axes[0].edges(), h.values(), v["bw"], v["fit_lo"], v["fit_hi"])
            return np.where(np.isfinite(x) & (x > 0), x, 0.0)
        cr_data, cr_other = snap(hcr["fcr_data_EGamma_native"]), snap(hcr["fcr_rest_native"])
        sb = CardSB(edges, meta["m_c"], meta["sigma"], mode="fcr_shape",
                    fcr=dict(R=fcr_all["R_global"], T_cr=f["T_cr"], b_tt=f["b_tt"],
                             cr_other=cr_other, R_err_rel=f.get("R_err_rel", 0.0)))

        rows = []
        for r_inj in args.inject:
            truth = mu_sr + r_inj * sb.S
            g0 = sb.fit(truth, data_cr=cr_data)
            rng = np.random.default_rng(args.seed + int(round(r_inj)))
            rh = []
            for _ in range(args.ntoys):
                g = sb.fit(rng.poisson(truth).astype(float), data_cr=rng.poisson(cr_data).astype(float))
                if g["status"] == 0:
                    rh.append(g["nsig"])
            rh = np.asarray(rh)
            row = dict(injected=float(r_inj), n_ok=int(len(rh)), mean=float(rh.mean()),
                       rms=float(rh.std()), median=float(np.median(rh)),
                       err_mean=float(rh.std() / np.sqrt(len(rh))), unfluctuated=float(g0["nsig"]))
            rows.append(row)
            logger.info("inject %5.0f: mean %7.2f  RMS %5.2f  bias %+5.2f +- %.2f  (unfluctuated %.2f, %d/%d ok)",
                        r_inj, row["mean"], row["rms"], row["mean"] - r_inj, row["err_mean"],
                        row["unfluctuated"], row["n_ok"], args.ntoys)
        ntoys, window = args.ntoys, [float(edges[0]), float(edges[-1])]

    x = np.array([r["injected"] for r in rows]); m = np.array([r["mean"] for r in rows])
    s = np.array([r["rms"] for r in rows]); em = np.array([r["err_mean"] for r in rows])
    slope, icpt = np.polyfit(x, m, 1)
    logger.info("linear fit of mean vs injected: slope %.3f, intercept %.2f", slope, icpt)

    hep.style.use("CMS")
    fig, (ax, axr) = plt.subplots(2, 1, sharex=True, figsize=tuple(args.figsize),
                                  gridspec_kw=dict(height_ratios=[3, 1], hspace=0.08))
    xmax = max(x) * 1.08
    ax.plot([0, xmax], [0, xmax], color="gray", lw=1.6, ls="--", label="fitted = injected")
    ax.errorbar(x, m, yerr=s, fmt="o", color="#e42536", ms=9, capsize=4, lw=1.8,
                label=fr"mean fitted $\hat r$; bars: spread of the {ntoys:,} toys")
    ax.set_ylabel(r"fitted $\hat r$ [events]")
    ax.set_xlim(-5, xmax)
    ax.set_ylim(min(-25.0, float((m - s).min()) - 5), xmax + 15)
    ax.legend(fontsize=14, loc="upper left")
    ax.text(0.97, 0.05,
            "ee resolved, " fr"$m_{{W_R}} = {MASS}$" "\n"
            "SR: 2024 simulation + injected signal\n"
            r"CR: $e\mu$ data" "\n" "two-region fit, $r$ floating",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=14)
    axr.axhline(0.0, color="gray", lw=1.4, ls="--")
    axr.errorbar(x, m - x, yerr=em, fmt="o", color="#e42536", ms=7, capsize=3)
    axr.set_ylabel("fitted $-$\ninjected", fontsize=16)
    axr.set_xlabel(r"injected signal events")
    lim = max(3.0, float(np.abs(m - x).max() + 2 * em.max()) * 1.25)
    axr.set_ylim(-lim, lim)
    _cmslabel(ax, com, lumi)
    _save(fig, out)
    json.dump(dict(points=rows, slope=float(slope), intercept=float(icpt), ntoys=ntoys,
                   window=window, mu_tt=float(mu_tt)),
              open(out.with_suffix(".json"), "w"), indent=1)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
