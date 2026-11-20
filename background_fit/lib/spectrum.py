"""Step 1 helpers: load the components of one channel and DY variant, rebin, summarise.

Histograms: WR_Plotter/rootfiles/<era subpath>/<dir>/WRAnalyzer_<group>.root for every era of
config.ERAS (the dataset chosen by FIT_DATASET), key
wr_<ch>_resolved_sr/mass_fourobject_wr_<ch>_resolved_sr, 10 GeV bins from 0 to 12 TeV, MC
lumi-scaled, sumw2 stored. The eras are summed with their weights (values x w, variances x w^2).
The DY component comes from the variant's file, the other three groups from their own files.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config as C                                                       # noqa: E402

COMPONENT_LABEL = {"DYJets": "DY+jets", "tt_tW": r"$t\bar{t}$+tW", "Nonprompt": "Nonprompt",
                   "Other": "Other", "sum": "Total background",
                   # the split9 processes (the plotter's names)
                   "WJets": "W+jets", "TTSemiLep": r"$t\bar{t}$ semilep.", "tWSemiLep": "tW semilep.",
                   "SingleTop": "single top", "Diboson": "diboson", "ttV": r"$t\bar{t}$V", "Triboson": "triboson"}
# house colours of the four groups (data/sample_groups.yaml); a split process takes a shade of its parent's
COMPONENT_COLOR = {"DYJets": "#5790fc", "tt_tW": "#f89c20", "Nonprompt": "#e42536",
                   "Other": "#964a8b", "sum": "#000000",
                   "WJets": "#e42536", "TTSemiLep": "#f28e8e", "tWSemiLep": "#c2185b", "SingleTop": "#8c1d1d",
                   "Diboson": "#964a8b", "ttV": "#5a2d57", "Triboson": "#c9a0c4"}
# line styles of overlaid per-process curves: the split processes share their parent's hue
COMPONENT_LS = {"TTSemiLep": ":", "tWSemiLep": "-.", "SingleTop": (0, (1, 1)), "ttV": ":", "Triboson": "-."}


def era_dirs():
    """[(histogram dir, split-process dir, weight)] of the dataset's eras (config.ERAS)."""
    return [(C.REPO / "rootfiles" / sub / d, C.REPO / "rootfiles" / sub / sd, w) for sub, d, sd, w in C.ERAS]


def input_dir():
    """The histogram directory of a single-era dataset (simfit's CR reader needs one directory)."""
    if len(C.ERAS) != 1:
        raise RuntimeError(f"dataset {C.DATASET} sums {len(C.ERAS)} eras: use era_dirs()")
    return era_dirs()[0][0]


def hist_key(channel):
    region = f"wr_{channel}_{C.TOPOLOGY}_sr"
    return f"{region}/{C.VARIABLE}_{region}"


def split_dir():
    """The split-process directory of a single-era dataset."""
    if len(C.ERAS) != 1:
        raise RuntimeError(f"dataset {C.DATASET} sums {len(C.ERAS)} eras: use era_dirs()")
    return era_dirs()[0][1]


def load_components(channel, variant="lo_ht", components="groups4"):
    """{component: (edges, values, variances)} at the native binning, plus 'sum', summed over the
    dataset's eras with their weights.

    components = 'groups4': the four groups of each era's histogram dir (DYJets from the variant's file).
    components = 'split9': Nonprompt and Other as their seven processes from the era's split-process dir;
    DYJets (the variant's file) and tt_tW still from the histogram dir."""
    names = C.COMPONENT_SETS[components]
    out = {}
    for d, sd, w in era_dirs():
        for comp in names:
            if comp == "DYJets":
                path = d / C.DY_VARIANTS[variant]
            elif comp in C.GROUPS:
                path = d / f"WRAnalyzer_{comp}.root"
            else:
                path = sd / f"WRAnalyzer_{comp}.root"
            f = uproot.open(path)
            if hist_key(channel) in f:
                h = f[hist_key(channel)]
                e, v, var = h.axes[0].edges(), h.values().astype(float) * w, h.variances().astype(float) * (w * w)
            else:
                # the analyzer writes only populated region folders: a small process with no event in the
                # region (2022 single top in the ee SR) contributes zero at the binning of the components read so far
                if not out:
                    raise ValueError(f"{path}: no {hist_key(channel)} and no component read before it")
                e = next(iter(out.values()))[0]                  # DYJets is read first in every set
                v, var = np.zeros(len(e) - 1), np.zeros(len(e) - 1)
            if comp in out:
                e0, v0, var0 = out[comp]
                if not np.array_equal(e0, e):
                    raise ValueError(f"{path}: binning differs from the first era's")
                out[comp] = (e0, v0 + v, var0 + var)
            else:
                out[comp] = (e, v, var)
    e = out["DYJets"][0]
    out["sum"] = (e, sum(out[c][1] for c in names), sum(out[c][2] for c in names))
    return out


def components_of(parts):
    """The component names of a loaded set or a truth (everything but the sum)."""
    return [k for k in parts if k != "sum"]


def rebin(edges, values, variances, width=C.BIN_WIDTH):
    """Merge native bins into bins of `width` (a multiple of the native width)."""
    f = int(round(width / (edges[1] - edges[0])))
    n = (len(values) // f) * f
    v = values[:n].reshape(-1, f).sum(axis=1)
    var = variances[:n].reshape(-1, f).sum(axis=1)
    e = edges[::f][: len(v) + 1]
    return e, v, var


def centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def last_populated(edges, values):
    idx = np.nonzero(values != 0)[0]
    return float(edges[idx[-1] + 1]) if len(idx) else float(edges[0])


def neff(values, variances, mask=None):
    v = values if mask is None else values[mask]
    var = variances if mask is None else variances[mask]
    s, s2 = float(v.sum()), float(var.sum())
    return (s * s / s2) if s2 > 0 else 0.0


def tail(values, variances, edges, above):
    c = centers(edges)
    m = c > above
    return float(values[m].sum()), math.sqrt(max(float(variances[m].sum()), 0.0))


def summary_rows(channel, variant, comps):
    """One row per component with yield, N_eff, last event, negative bins, tails."""
    rows = []
    for comp, (e, v, var) in comps.items():
        c = centers(e)
        inside = c > C.THRESHOLD
        row = {"channel": channel, "variant": variant, "component": comp,
               "yield": float(v[inside].sum()), "neff": neff(v, var, inside),
               "last_event": last_populated(e, v), "neg_bins": int((v < 0).sum()),
               "neff_gt3000": neff(v, var, c > 3000.0)}
        for t in C.TAIL_EDGES:
            y, err = tail(v, var, e, t)
            row[f"gt{int(t)}"] = y
            row[f"gt{int(t)}_err"] = err
        rows.append(row)
    return rows
