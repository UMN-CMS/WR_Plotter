"""Shared paths/loaders for the run2 no-data explainer steps.

Everything is read-only: the steps consume the already-produced run2 inputs
(../baseline/run2/inputs), the Stage-6 run2 window table, and the
Stage-10.8 refined results. LCG_106 environment.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent               # 8_combine_limits/explainer
STAGE9 = HERE.parent / "baseline"                    # config-B combine chain
SIGFIT = HERE.parents[1]                             # signal_fitting
sys.path.insert(0, str(SIGFIT.parent))               # repo root (wrplotter)
sys.path.insert(0, str(SIGFIT / "4_background_fits"))
sys.path.insert(0, str(SIGFIT / "shared"))
sys.path.insert(0, str(SIGFIT / "7_limit_plots"))

ERA = "RunIISummer20UL18"
BKG_DIR = "20260714_run2_bkgs"
SIGNAL_DIR = "20260624_signals"
CHANNEL, TOPOLOGY = "ee", "resolved"
TAG = f"{CHANNEL}_{TOPOLOGY}"
K = 3.0                                              # window half-width [sigma]
BFRAC = 0.5                                          # e-share of the mixed samples

INPUTS = STAGE9 / "run2" / "inputs"                  # Stage-9 run2 combine inputs
STAGE6_TABLE = SIGFIT / "6_spurious_signal_toys" / "run2" / f"spurious_toy_table_{TAG}.csv"
STAGE9_TABLE = STAGE9 / "run2" / f"combine_limit_table_{TAG}.csv"   # data-observed
REFINED = HERE.parent / "production"
REFINED_TABLE = REFINED / f"refined_limit_table_{TAG}.csv"
ANCHORS = REFINED / "cards" / TAG / "anchors.json"

FLOAT_MIN, FLOAT_MAX = 1400.0, 3200.0                # the 10.8 regime split
OPT = HERE.parent / "optimization"
FLOAT_VAR = "k5_bw50"                                # the 10.9 winner window/binning
SIG_ALPHA = 0.30                                     # mu/sigma prior width (x sigma0)
BKG_SAMPLES = ["DYJets", "tt_tW", "Nonprompt", "Other"]
BKG_LABELS = {"DYJets": "DY+jets (LO HT, reshaped)", "tt_tW": r"$t\bar{t}$+tW",
              "Nonprompt": "Nonprompt", "Other": "Other"}


def regime(mass: float) -> str:
    if mass < FLOAT_MIN:
        return "anch_low"
    if mass > FLOAT_MAX:
        return "anch_sparse"
    return "float"


def load_meta():
    """Stage-9 run2 inputs JSON: lumi + per-mass window/eff/rate."""
    with open(INPUTS / f"{TAG}.json") as fh:
        return json.load(fh)


def load_bkg_components(rebin=10):
    """{sample: (edges, values)} + summed, from the converted run2 files."""
    import uproot
    from wrplotter.paths import input_dirs_for_era, repo_root
    dirs, _ = input_dirs_for_era(ERA, repo_root(), BKG_DIR)
    key = f"wr_{CHANNEL}_{TOPOLOGY}_sr/mass_fourobject_wr_{CHANNEL}_{TOPOLOGY}_sr"
    out = {}
    for s in BKG_SAMPLES:
        edges = vals = None
        for d in dirs:
            h = uproot.open(d / f"WRAnalyzer_{s}.root")[key]
            e, v = h.axes[0].edges(), h.values()
            edges, vals = e, (v if vals is None else vals + v)
        n = (len(vals) // rebin) * rebin
        out[s] = (edges[0:n + 1:rebin],
                  vals[:n].reshape(-1, rebin).sum(axis=1))
    return out


def load_bkg_total(rebin=10):
    """(edges, total, stat_err): summed background + combined MC stat error
    (sqrt of the summed sumw2 across all samples and sub-eras)."""
    import numpy as np
    import uproot
    from wrplotter.paths import input_dirs_for_era, repo_root
    dirs, _ = input_dirs_for_era(ERA, repo_root(), BKG_DIR)
    key = f"wr_{CHANNEL}_{TOPOLOGY}_sr/mass_fourobject_wr_{CHANNEL}_{TOPOLOGY}_sr"
    edges = vals = var = None
    for d in dirs:
        for s in BKG_SAMPLES:
            h = uproot.open(d / f"WRAnalyzer_{s}.root")[key]
            e, v, w2 = h.axes[0].edges(), h.values(), h.variances()
            edges = e
            vals = v.copy() if vals is None else vals + v
            var = w2.copy() if var is None else var + w2
    n = (len(vals) // rebin) * rebin
    return (edges[0:n + 1:rebin],
            vals[:n].reshape(-1, rebin).sum(axis=1),
            np.sqrt(var[:n].reshape(-1, rebin).sum(axis=1)))


def load_signal(tag="WR2000_N1000", rebin=10):
    """(edges, values) raw genWeight signal shape in the ee resolved SR."""
    import uproot
    from wrplotter.paths import input_dirs_for_era, repo_root
    dirs, _ = input_dirs_for_era(ERA, repo_root(), SIGNAL_DIR)
    key = f"wr_{CHANNEL}_{TOPOLOGY}_sr/mass_fourobject_wr_{CHANNEL}_{TOPOLOGY}_sr"
    edges = vals = None
    for d in dirs:
        h = uproot.open(d / f"WRAnalyzer_signal_{tag}.root")[key]
        e, v = h.axes[0].edges(), h.values()
        edges, vals = e, (v if vals is None else vals + v)
    n = (len(vals) // rebin) * rebin
    return edges[0:n + 1:rebin], vals[:n].reshape(-1, rebin).sum(axis=1)


def load_stage6_windows():
    """[{mWR, m_c, sigma, fit_lo, fit_hi, B_window}] (expo rows)."""
    rows = []
    with open(STAGE6_TABLE, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["function"] != "expo":
                continue
            rows.append({k: float(r[k]) for k in
                         ("mWR", "m_c", "sigma_win", "fit_lo", "fit_hi",
                          "B_window")})
    return sorted(rows, key=lambda r: r["mWR"])


def load_refined():
    with open(REFINED_TABLE, newline="") as fh:
        return [dict(r, mWR=float(r["mWR"])) for r in csv.DictReader(fh)]


def load_opt_inputs():
    """Stage-10.9 inputs: per-mass variant windows + efficiencies."""
    with open(OPT / "inputs" / f"{TAG}.json") as fh:
        return json.load(fh)["masses"]


def load_opt_table():
    """Stage-10.9 scan results: {variant: {mass: median fb}}."""
    out = {}
    with open(OPT / f"opt_table_{TAG}.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["variant"]] = {int(k.split("_")[1]): float(v)
                                 for k, v in r.items()
                                 if k.startswith("med_") and v}
    return out


def load_stage9(fn="expo"):
    with open(STAGE9_TABLE, newline="") as fh:
        return {float(r["mWR"]): r for r in csv.DictReader(fh)
                if r["function"] == fn}


# Stage-1 (1_signal_widths) plotting conventions, adopted per user request:
# blue stairs for MC shapes, light-red axvspan for the window, red curve for
# the Gaussian (normalized to the in-window count), dotted black mu line.
BLUE = "#3f90da"
RED = "#bd1f01"


def log_yaxis_one_ten(ax):
    """On a log y-axis, label 10^0 as '1' and 10^1 as '10' (CMS style); every
    other decade keeps its power-of-ten mathtext label."""
    import numpy as np
    from matplotlib.ticker import LogFormatterMathtext

    class _Fmt(LogFormatterMathtext):
        def __call__(self, x, pos=None):
            if x > 0:
                e = int(round(np.log10(x)))
                if e == 0:
                    return "$1$"
                if e == 1:
                    return "$10$"
            return super().__call__(x, pos)

    ax.yaxis.set_major_formatter(_Fmt())


def savefig(fig, stem):
    """stem: full path stem (each step dir owns its figures)."""
    from pathlib import Path as _P
    stem = _P(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {stem.name}.pdf/.png")
