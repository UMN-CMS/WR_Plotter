"""Step 4 helpers: the smoothed pseudo-data truths, one set per DY variant.

A truth is built per component on the native 10 GeV grid and summed (the components are
whatever spectrum.load_components returned: the four groups or the split9 processes):

  raw            the MC histogram
  kde            the RooKeysPdf alone, its tail as it comes (a stress truth: the adaptive bandwidth
                 leaks events far above the last MC event)
  <bulk>_<tail>  below the component's trust mass m_T the bulk smoother (kde = RooKeysPdf with
                 the production settings; kernel = local-quadratic kernel regression in ln m),
                 above it the Poisson-ML tail fit of family <tail> (expo or powexp) made by
                 smooth_lib.smooth_hybrid, with its 200 GeV linear blend before m_T.

The tail family of the truth must differ from the function under test (ATL-PHYS-PUB-2020-028
Sec. 3.1.1: the same function for smoothing and fitting gives a bias that vanishes by
definition), so every truth set carries both tails and the toy step pairs them accordingly.
RooKeysPdf cannot take negative weights: smooth_lib drops negative bins from its dataset and
keeps them in the normalisation; the kernel regression takes them as they are. A component
whose tail fit fails (for example a DY variant with a negative yield above 3 TeV) makes the
truth unavailable for that variant; the summary says so.

Checks per component and for the sum (the ATLAS leptoquark tail checks): the truth's yield
above 3, 4 and 5 TeV against the MC's with the MC error, and the mean Pearson pull
(MC - truth)/sigma_MC per 100 GeV bin below m_T (a trend means the bulk smoother distorts the
shape).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config as C                                                       # noqa: E402
from util import sl                                                      # noqa: E402
from spectrum import centers, rebin, tail                                # noqa: E402

COMPONENTS = ["DYJets", "tt_tW", "Nonprompt", "Other"]   # the default set; build_truth_set takes what it is given


def smooth_component(e, v, var, bulk, tail_family):
    """(smoothed 10 GeV array, diagnostics) of one component; raises RuntimeError when the
    tail fit is impossible."""
    if bulk == "raw" and tail_family is None:
        return np.array(v, float), {"m_T": None}
    if not np.any((np.asarray(v) > 0) & (centers(e) >= C.THRESHOLD)):     # an empty process (split sets)
        return np.zeros_like(np.asarray(v, float)), {"m_T": None, "empty": True}
    if bulk == "kde" and tail_family is None:                   # the RooKeysPdf alone, its tail as it comes
        k, dk = sl.smooth_kde(e, v, var)
        k = np.asarray(k, float)
        k[centers(e) < C.THRESHOLD] = 0.0
        return k, {"m_T": None, "kde": {kk: dk[kk] for kk in ("rho", "mirror", "var_kind", "n_entries", "n_negative_dropped", "rescale")}}
    # A process whose trust mass sits at the threshold has no bulk for a tail to start from (single top,
    # semileptonic tW in the split sets: 12 and 31 MC events); it keeps its bulk smoother alone.
    # Fitting a "tail" from 800 GeV would put
    # 0.12 events above 5 TeV into the sum, where the whole simulation has 0.18.
    e50, v50, var50 = sl.rebin(e, v, var, int(round(50.0 / C.NATIVE_BW)))
    m_T0 = sl.trust_mass(e50, v50, var50, C.TRUST_NEFF, C.TRUST_WIDTH, step=50.0)
    if m_T0 <= C.THRESHOLD + 1e-6:
        if bulk == "raw":
            return np.array(v, float), {"m_T": float(m_T0), "no_tail": "trust mass at the threshold"}
        k, dk = (sl.smooth_kde if bulk == "kde" else sl.smooth_kernel)(e, v, var)
        k = np.asarray(k, float)
        k[centers(e) < C.THRESHOLD] = 0.0
        return k, {"m_T": float(m_T0), "no_tail": "trust mass at the threshold",
                   bulk: {kk: dk.get(kk) for kk in ("rho", "mirror", "var_kind", "n_entries", "n_negative_dropped",
                                                    "rescale", "degree", "h_min", "h_max", "n_target") if kk in dk}}
    h, dh = sl.smooth_hybrid(e, v, var, family=tail_family, trust_neff=C.TRUST_NEFF,
                             trust_width=C.TRUST_WIDTH)
    diag = {"m_T": float(dh["m_T"]), "m_last": float(dh["m_last"]), "tail_family": dh["family"],
            "tail_params": [float(x) for x in dh["fit"]["params"]],
            "tail_raw": float(dh["tail_raw"]), "tail_fit": float(dh["tail_fit"]),
            "tail_pull": float(dh["tail_pull"]) if math.isfinite(dh["tail_pull"]) else None,
            "seam_pull": float(dh["seam_pull"]) if math.isfinite(dh["seam_pull"]) else None}
    if bulk == "raw":
        return np.asarray(h, float), diag
    if bulk == "kde":
        k, dk = sl.smooth_kde(e, v, var)
        diag["kde"] = {kk: dk[kk] for kk in ("rho", "mirror", "var_kind", "n_entries", "n_negative_dropped", "rescale")}
    elif bulk == "kernel":
        k, dk = sl.smooth_kernel(e, v, var)
        diag["kernel"] = {kk: dk.get(kk) for kk in ("degree", "h_min", "h_max", "n_target", "rescale") if kk in dk}
    else:
        raise KeyError(bulk)
    c = centers(e)
    m_T, l_blend = dh["m_T"], dh["l_blend"]
    t = np.clip((c - (m_T - l_blend)) / l_blend, 0.0, 1.0)      # 0 = bulk smoother, 1 = hybrid (tail fit)
    out = (1.0 - t) * np.asarray(k, float) + t * np.asarray(h, float)
    out[c < C.THRESHOLD] = 0.0
    return out, diag


def check_component(e, v, var, s, m_T):
    """Tail comparisons and the bulk pull trend of one smoothed component."""
    chk = {}
    for thr in C.TAIL_EDGES:
        mc, err = tail(v, var, e, thr)
        tr, _ = tail(s, var, e, thr)
        chk[f"gt{int(thr)}"] = {"mc": mc, "mc_err": err, "truth": tr,
                                "ratio": (tr / mc) if mc > 0 else None,
                                "pull": ((mc - tr) / err) if err > 0 else None}
    e1, v1, var1 = rebin(e, v, var)
    _, s1, _ = rebin(e, s, var)
    c1 = centers(e1)
    lim = m_T if m_T else 3000.0          # the sum has no single m_T: its bulk check stops at 3 TeV
    m = (c1 >= C.FIT_LO) & (c1 < lim) & (var1 > 0)
    pulls = (v1[m] - s1[m]) / np.sqrt(var1[m])
    chk["bulk_pull_mean"] = float(pulls.mean()) if len(pulls) else None
    chk["bulk_pull_rms"] = float(pulls.std()) if len(pulls) else None
    chk["bulk_nbins"] = int(len(pulls))
    return chk


def build_truth_set(comps10, name):
    """Build one truth for all components. name = 'raw' or '<bulk>_<tail>'.
    Returns ({component: array}, {component: diag + checks}, ok, reason)."""
    if "_" not in name:                       # raw, or a bulk smoother alone (kde: the RooKeysPdf with its own tail)
        bulk, tail_family = name, None
    else:
        bulk, tail_family = name.split("_", 1)
    parts, diags = {}, {}
    comps = [k for k in comps10 if k != "sum"]
    for comp in comps:
        e, v, var = comps10[comp]
        try:
            s, d = smooth_component(e, v, var, bulk, tail_family)
        except Exception as exc:                        # noqa: BLE001 - the reason is recorded
            return None, {comp: {"error": str(exc)}}, False, f"{comp}: {exc}"
        if not np.all(np.isfinite(s)):
            return None, {comp: {"error": "non-finite smoothed values"}}, False, f"{comp}: non-finite values"
        parts[comp] = s
        d.update(check_component(e, v, var, s, d.get("m_T")))
        diags[comp] = d
    e, v, var = comps10["sum"]
    parts["sum"] = sum(parts[cmp] for cmp in comps)
    diags["sum"] = check_component(e, v, var, parts["sum"], None)
    return parts, diags, True, ""


def save_truth(path, edges, parts, diags):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), edges=edges, **{k: np.asarray(v) for k, v in parts.items()})
    path.with_suffix(".json").write_text(json.dumps(diags, indent=1, default=float))


def load_truth(path):
    z = np.load(Path(path).with_suffix(".npz"))
    parts = {k: z[k] for k in z.files if k != "edges"}
    diags = json.loads(Path(path).with_suffix(".json").read_text())
    return z["edges"], parts, diags
