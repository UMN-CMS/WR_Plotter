from pathlib import Path

import numpy as np
from hist import Hist
from hist.axis import Variable
from hist.storage import Weight

def rebin_histogram(h: Hist, spec: int | list[float], *, density: bool = True,
                    xmax: float | None = None) -> Hist:
    """
    Rebin a 1D hist.Hist either by:
      - an integer spec      -> merge every N bins, or
      - a list of bin edges  -> exactly those new edges [e0, e1, ..., eM].
    Returns a new Hist with Weight() storage so variances are tracked.

    Overflow bins are folded into the last visible bin for both modes.
    If *xmax* is given, the new edges are truncated there and all bins
    beyond xmax are folded into the last visible bin as overflow.

    Variable-width rebinning (spec is a list):
        After summing the original bin contents into the new wider bins,
        the result is converted to a differential density so that the
        y-axis has consistent units (events / GeV) regardless of bin width:

            value_i   = sum_of_counts_i / width_i
            variance_i = sum_of_variances_i / width_i**2

        where width_i = edge_{i+1} - edge_i. This is necessary because
        matplotlib/mplhep draws each bin at its face value; without
        normalisation, wider bins would appear artificially tall.
    """
    # 1) old edges and bin count
    old_edges = np.array(h.axes[0].edges, dtype=float)
    nbins_old = len(old_edges) - 1

    # 2) build new edges and track if we're using variable binning
    variable_binning = False
    if isinstance(spec, int):
        if spec < 1 or spec > nbins_old:
            raise ValueError(f"Invalid integer spec={spec}")
        edges = old_edges[::spec]
        if edges[-1] != old_edges[-1]:
            edges = np.concatenate([edges, [old_edges[-1]]])
    else:
        edges = np.array(spec, dtype=float)
        variable_binning = True  # Flag that we're using variable bin widths

    # 3) sanity checks
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("When spec is not int, it must be ≥2 edges")
    if edges[0] < old_edges[0] or edges[-1] > old_edges[-1]:
        raise ValueError(f"Edges {edges[0]}…{edges[-1]} out of range {old_edges[0]}…{old_edges[-1]}")

    # 3b) truncate edges at xmax so bins beyond it become overflow
    if xmax is not None and xmax < edges[-1]:
        # Keep edges up to and including xmax; snap to nearest edge <= xmax
        mask = edges <= xmax
        if mask.sum() < 2:
            raise ValueError(f"xmax={xmax} leaves fewer than 2 edges")
        edges = edges[mask]
        # If xmax exactly matches an edge we're done; otherwise append it
        if edges[-1] < xmax:
            edges = np.append(edges, xmax)

    # 4) extract original contents & variances (including overflow)
    vals = h.values(flow=True)  # Include overflow bin
    vars_ = h.variances(flow=True)  # Include overflow bin

    # 5) compute bin centers and new‐bin indices
    # For flow=True, we get [underflow, bin1, bin2, ..., binN, overflow]
    # We need to handle the overflow specially
    centers = 0.5 * (old_edges[:-1] + old_edges[1:])
    inds    = np.digitize(centers, edges) - 1

    # 6) accumulate into new arrays
    new_counts = np.zeros(edges.size - 1, dtype=float)
    new_vars   = np.zeros(edges.size - 1, dtype=float)

    # Add regular bins (skip underflow at index 0).
    # Bins whose center falls beyond the last new edge are folded into the
    # last visible bin (same treatment as the ROOT-level overflow).
    has_overflow = False
    for v, w2, ib in zip(vals[1:-1], vars_[1:-1], inds):
        if ib >= new_counts.size:
            new_counts[-1] += v
            new_vars[-1]   += w2
            if v > 0:
                has_overflow = True
        elif ib >= 0:
            new_counts[ib] += v
            new_vars[ib]   += w2

    # Add ROOT-level overflow bin to the last bin
    if len(vals) > 1:
        new_counts[-1] += vals[-1]
        new_vars[-1] += vars_[-1]
        if vals[-1] > 0:
            has_overflow = True

    # 6b) Normalize by bin width for variable binning
    if variable_binning and density:
        bin_widths = np.diff(edges)
        new_counts = new_counts / bin_widths
        new_vars = new_vars / (bin_widths**2)  # variance scales as 1/width^2

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

    # If overflow was folded in, set a nonzero sentinel in the histogram's
    # overflow flow bin so that mplhep's flow="hint" draws an arrow marker.
    if has_overflow:
        flow_arr = new_h.view(flow=True)
        flow_arr['value'][-1] = 1.0

    return new_h


def extract_hist_data(h):
    """Extract edges, bin centers, values, and errors from a hist.Hist."""
    edges = h.axes[0].edges
    vals  = h.values()
    errs  = np.sqrt(h.variances()) if h.variances() is not None else np.sqrt(vals)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers, vals, errs


def load_histogram(path: Path, hist_key: str) -> Hist:
    """Load a single histogram from a ROOT file, using the shared LRU cache.

    Uses the same cached file handle as histogram_cache.load_and_rebin() so repeated
    calls to the same file avoid redundant I/O.
    """
    from .histogram_cache import _open_root  # lazy import avoids circular dependency
    return _open_root(str(path))[hist_key].to_hist()
