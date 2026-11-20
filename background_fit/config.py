"""background_fit configuration: one place for every choice of the data-driven background study.

The study: the background of the S+B fit is an analytic function fitted to
data; the MC, smoothed only to remove spiky fluctuations, is the pseudo-data used to choose
and validate the function. ee and mumu, resolved and boosted SR.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                                   # WR_Plotter
sys.path.insert(0, str(REPO))                        # the plotter package (wrplotter) needs no install

# ---- inputs ------------------------------------------------------------------------------
# The dataset: FIT_DATASET=2024 (default) or run3.  Every era of a dataset enters the spectrum
# through spectrum.load_components as (rootfiles subpath, histogram dir, split-process dir, weight):
# values x weight, variances x weight^2 (a scaled copy of the same MC events adds no statistics).
# FIT_DIR=<name> reads another production directory (the analyzer's --dir), FIT_SPLIT_DIR another split-process one.
DIR = os.environ.get("FIT_DIR", "20261015_skims_v2")   # the production set that holds every DY variant in every era (the
                               # 2024 20261019 rerun, cutflow bookkeeping, agrees with it on the
                               # resolved SR yields to < 0.1 % and has no variants)
_SPLIT_2024 = os.environ.get("FIT_SPLIT_DIR", "20261018_split_bkgs")
_SPLIT_2223 = os.environ.get("FIT_SPLIT_DIR", "20261027_split_bkgs")
_LUMI_2024, _LUMI_2025 = 109.70, 110.63             # fb-1, WR_Plotter/data/lumi.yaml (2025 preliminary)
DATASETS = {
    "2024": dict(label="RunIII2024Summer24", year="2024", lumi=_LUMI_2024, results="results",
                 eras=[("Run3/2024/RunIII2024Summer24", DIR, _SPLIT_2024, 1.0)]),
    # all of Run 3: 2022-2024 as produced; 2025 (data only, MC aliased to 2024) as the 2024 MC scaled to the
    # 2025 luminosity, so 2024 enters with weight 1 + L(2025)/L(2024)
    "run3": dict(label="Run3_2022-2025", year="2022-2025", lumi=8.08 + 26.64 + 18.59 + 9.675 + _LUMI_2024 + _LUMI_2025,
                 results="results_run3",
                 eras=[("Run3/2022/Run3Summer22", DIR, _SPLIT_2223, 1.0),
                       ("Run3/2022/Run3Summer22EE", DIR, _SPLIT_2223, 1.0),
                       ("Run3/2023/Run3Summer23", DIR, _SPLIT_2223, 1.0),
                       ("Run3/2023/Run3Summer23BPix", DIR, _SPLIT_2223, 1.0),
                       ("Run3/2024/RunIII2024Summer24", DIR, _SPLIT_2024, 1.0 + _LUMI_2025 / _LUMI_2024)]),
    # the same luminosity from the 2024 MC alone (x 2.58): separates the statistics from the 2022/2023 shapes
    "run3_scaled": dict(label="RunIII2024Summer24_x2.58", year=r"2024 MC $\times$ 2.58", results="results_run3scaled",
                        lumi=8.08 + 26.64 + 18.59 + 9.675 + _LUMI_2024 + _LUMI_2025,
                        eras=[("Run3/2024/RunIII2024Summer24", DIR, _SPLIT_2024,
                               (8.08 + 26.64 + 18.59 + 9.675 + _LUMI_2024 + _LUMI_2025) / _LUMI_2024)]),
}
_RENAMED = sorted(k for k in os.environ if k.startswith("FITV3_"))
if _RENAMED:                   # the variables' former prefix: refuse it, or the run would silently use the defaults
    raise SystemExit(f"{', '.join(_RENAMED)}: use FIT_DATASET, FIT_TOPOLOGY, FIT_SIGNAL_MAP, FIT_NSIG_LIMIT")
DATASET = os.environ.get("FIT_DATASET", "2024")
if DATASET not in DATASETS:
    raise SystemExit(f"FIT_DATASET={DATASET!r}: choose from {sorted(DATASETS)}")
# The topology: FIT_TOPOLOGY=resolved (default, the study as before) or boosted (2026-11-15: the same chain
# on m_lJ of the boosted SR). Boosted results sit in their own trees, results_boosted*/, so nothing of the
# resolved study is rewritten.
TOPOLOGY = os.environ.get("FIT_TOPOLOGY", "resolved")
if TOPOLOGY not in ("resolved", "boosted"):
    raise SystemExit(f"FIT_TOPOLOGY={TOPOLOGY!r}: choose resolved or boosted")
BOOSTED = TOPOLOGY == "boosted"


def results_name(dataset):
    """The results folder of a dataset in the current topology: results[_run3], results_boosted[_run3]."""
    name = DATASETS[dataset]["results"]
    return name.replace("results", "results_boosted", 1) if BOOSTED else name


_DS = DATASETS[DATASET]
ERA = _DS["label"]             # the dataset's name in summaries
YEAR = _DS["year"]             # the year part of the luminosity label on the figures
LUMI = _DS["lumi"]             # fb-1
ERAS = _DS["eras"]
RESULTS = HERE / results_name(DATASET)
COM = 13.6                     # TeV
SQRTS = 13600.0                # GeV, for the dijet-type variable z = m / sqrt(s)
CHANNELS = ["ee", "mumu"]
VARIABLE = "mass_twoobject" if BOOSTED else "mass_fourobject"      # m_lJ / m_lljj
REGION_TEXT = "Boosted SR" if BOOSTED else "Resolved SR"           # the figures' region line
REGION_NAME = f"{TOPOLOGY} SR"                                     # the summaries' wording
MASS_LABEL = r"$m_{\ell J}$ [GeV]" if BOOSTED else r"$m_{\ell\ell jj}$ [GeV]"
GROUPS = ["tt_tW", "Nonprompt", "Other"]             # the non-DY components (one file each)
# The split-process directories (DATASETS[...]["eras"][i][2]): Nonprompt and Other as their processes (2024:
# --split-groups Nonprompt Other, 2026-10-18, whose DYJets and tt_tW agree with DIR to 0.05 events per 10 GeV
# bin and whose sub-process sums reproduce the groups; 2022/2023: one single-process run each, 2026-10-27).
SPLIT_DIR = ERAS[-1][2]             # the 2024 one, for summaries
COMPONENT_SETS = {                  # the components a truth is built from and summed over
    "groups4": ["DYJets", "tt_tW", "Nonprompt", "Other"],
    "split9": ["DYJets", "tt_tW", "WJets", "TTSemiLep", "tWSemiLep", "SingleTop", "Diboson", "ttV", "Triboson"],
}
SPLIT_PARENT = {"WJets": "Nonprompt", "TTSemiLep": "Nonprompt", "tWSemiLep": "Nonprompt", "SingleTop": "Nonprompt",
                "Diboson": "Other", "ttV": "Other", "Triboson": "Other"}
DY_VARIANTS = {                                      # variant -> file in DIR
    "lo_ht": "WRAnalyzer_DYJets.root",
    "nlo_mll": "WRAnalyzer_DYJets_nlo_mll.root",
    "nlo_njet": "WRAnalyzer_DYJets_nlo_njet.root",
    "nlo_inc": "WRAnalyzer_DYJets_nlo_inc.root",
    "lo_inc": "WRAnalyzer_DYJets_lo_inc.root",
    "lo_ht_rw": "WRAnalyzer_DYJets_lo_ht_zpt_reshape.root",   # LO HT-binned with the Z-pT reweight and the jet-pT reshape
}
DY_LABEL = {"lo_ht": "LO HT-binned (default)", "nlo_mll": "NLO m(ll)-binned",
            "nlo_njet": "NLO n(jet)-binned", "nlo_inc": "NLO inclusive", "lo_inc": "LO inclusive",
            "lo_ht_rw": "LO HT-binned, Z-pT + jet-pT reweighted"}
DY_SHORT = {k: v.replace(" (default)", "") for k, v in DY_LABEL.items()}   # for figure texts

# ---- spectrum (step 1) -------------------------------------------------------------------
THRESHOLD = 800.0              # the m_lljj selection cut: the spectrum is empty below
FIT_LO = 1000.0                # where the spectrum falls smoothly (the 800-1000 GeV region behaves
                               # as a threshold: every 3-parameter form gets 3-10x worse from 800)
BIN_WIDTH = 50.0 if BOOSTED else 100.0   # GeV; about one signal width at 1 TeV (resolved 104-110 GeV; boosted
                               # 43-55 GeV at 1-1.2 TeV, hence 50), finer above (the fits are
                               # unbiased by finer bins; variable resolution bins are a future option)
NATIVE_BW = 10.0               # the histograms' native bin width
TAIL_EDGES = [3000.0, 4000.0, 5000.0]
EXTRAP_SPLIT = 3000.0          # fit below, predict above
LO_VARIANTS = [800.0, 900.0, 1000.0, 1100.0, 1200.0]   # the threshold scan of step 1
PIVOT = 2000.0                 # the fits' pivot m0 (parameter basis only)
PIVOT_AT_SIGNAL = {"expo2n"}   # forms whose pivot is the signal mean of the S+B fit instead

# ---- function menu (step 2) --------------------------------------------------------------
# registry names of lib/published_functions.py
MENU = ["expo", "expo2", "powexp", "dilep8", "powlaw", "atlasemu", "wprime7"]
MENU_LABEL = {
    "expo": "exponential (2)", "expo2": "exp. with curvature (3)", "expo2n": "exp. with curvature, pivot at the signal (3)",
    "powexp": "power x exponential (3)", "dilep8": "power x exp(quadratic) (4)",
    "powlaw": "power law (2)", "atlasemu": "running power law (3)", "wprime7": "W' function (3)",
}
LADDER = [("expo", "powexp"), ("expo", "expo2"), ("powexp", "dilep8"), ("powlaw", "atlasemu")]
F_ALPHA = 0.05                 # Fisher F-test level (CMS dijet convention)
GOF_P_MIN = 0.01               # goodness-of-fit floor (HIG-19-014, EXO-22-019)

# ---- truths (step 4) ---------------------------------------------------------------------
BULKS = ["kde", "kernel"]      # smoother below the trust mass: RooKeysPdf or local-quadratic kernel regression
TAILS = ["expo", "powexp"]     # the fitted tail family above the trust mass (choose unlike the function under test)
TRUST_NEFF = 20.0              # smooth_lib defaults: trust mass = first 50 GeV point where the running N_eff
TRUST_WIDTH = 500.0            # over the next 500 GeV drops below 20
TRUTH_NAMES = ["raw"] + [f"{b}_{t}" for b in BULKS for t in TAILS]

# ---- toys (step 5) -----------------------------------------------------------------------
MASSES = [1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4400, 4800]
CANDIDATES = ["expo", "powexp", "atlasemu", "wprime7", "dilep8"]   # wprime7 added 2026-10-21 (the mumu screening winner)   # the S+B background models tested on toys
NTOYS = 500
INJECT = [0.0, 1.0, 2.0]       # x the expected 95 % CL upper limit (1.96 sigma_fit of the Asimov fit)
SEED = 20261021
BIAS_PASS = 0.5                # |median or mean bias| / sigma_fit (EXO-19-012, EXO-24-005)
BIAS_TARGET = 0.3              # the ATLAS "good compromise"
FLOAT_MAX = 3200.0             # the float regime: m_WR up to here is judged; above it the window is sparse
SIG_CORE_K = 2.0               # the signal core: +-2 sigma of the peak (b_core, and the old N_sig guard below)
# N_sig floats without limits in the S+B fit since 2026-11-16.
# Until then it was held to [-0.8 x the background within the core, max(5 x that + 50,
# 200)]: fits with a low fluctuation under the peak piled up on the lower limit (the spike of the N_sig histograms) and
# were dropped from the pulls. FIT_NSIG_LIMIT=1 restores that guard, to reproduce the toys made with it.
NSIG_LIMITED = os.environ.get("FIT_NSIG_LIMIT", "0") == "1"

# the signal map: straight lines mu(m_WR) and sigma(m_WR) per channel and topology (window_params.json)
# FIT_SIGNAL_MAP=<json of the same layout> replaces it (the 2024-derived boosted map: background_fit/signal_map_2024.json,
# written by 24_boosted_deck.py signal; give such toys their own folder tag, e.g. _100k_sig2024)
WINDOW_JSON = Path(os.environ["FIT_SIGNAL_MAP"]) if os.environ.get("FIT_SIGNAL_MAP") else \
    HERE / "window_params.json"
