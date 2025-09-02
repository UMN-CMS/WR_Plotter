import numpy as np
from hist import Hist
from hist.axis import Variable
from hist.storage import Weight

def rebin_histogram(h: Hist, spec):
    """
    Rebin a 1D hist.Hist either by:
      • an integer spec      → merge‐every‐N bins, or
      • a list of bin edges → exactly those new edges [e0,e1,…,eM].
    Returns a new Hist with Weight() storage so variances are tracked.
    """
    # 1) old edges and bin count
    old_edges = np.array(h.axes[0].edges, dtype=float)
    nbins_old = len(old_edges) - 1

    # 2) build new edges
    if isinstance(spec, int):
        if spec < 1 or spec > nbins_old:
            raise ValueError(f"Invalid integer spec={spec}")
        edges = old_edges[::spec]
        if edges[-1] != old_edges[-1]:
            edges = np.concatenate([edges, [old_edges[-1]]])
    else:
        edges = np.array(spec, dtype=float)

    # 3) sanity checks
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("When spec is not int, it must be ≥2 edges")
    if edges[0] < old_edges[0] or edges[-1] > old_edges[-1]:
        raise ValueError(f"Edges {edges[0]}…{edges[-1]} out of range {old_edges[0]}…{old_edges[-1]}")

    # 4) extract original contents & variances
    vals = h.values(flow=False)
    vars_ = h.variances(flow=False)

    # 5) compute bin centers and new‐bin indices
    centers = 0.5 * (old_edges[:-1] + old_edges[1:])
    inds    = np.digitize(centers, edges) - 1

    # 6) accumulate into new arrays
    new_counts = np.zeros(edges.size - 1, dtype=float)
    new_vars   = np.zeros(edges.size - 1, dtype=float)
    for v, w2, ib in zip(vals, vars_, inds):
        if 0 <= ib < new_counts.size:
            new_counts[ib] += v
            new_vars[ib]   += w2

    # 7) build the new histogram
    new_h = Hist(
        Variable(edges, name=h.axes[0].name),
        storage=Weight(),
        label=h.label,
        name=h.name,
    )

    # 8) inject the results correctly
    arr = new_h.view(flow=False)        # structured array with fields 'value' and 'variance'
    arr['value']    = new_counts
    arr['variance'] = new_vars

    return new_h

def scale_histogram(hist, scale_factor):
    """
    Scale the histogram data by a given factor.
    """
    try:
        # Assuming hist supports element-wise multiplication.
        return hist * scale_factor
    except Exception as e:
        raise RuntimeError(f"Error during scaling: {e}")
