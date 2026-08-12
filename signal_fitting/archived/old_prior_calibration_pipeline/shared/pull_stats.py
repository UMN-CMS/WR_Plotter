"""Standard HEP pull-distribution metric: a binned Gaussian fit.

The single source of truth for characterizing a pull distribution across the
study. We fit a Gaussian to the binned pulls in [lo, hi] and report:

    mu    (+/- mu_err)    -> the bias       (target 0)
    sigma (+/- sigma_err) -> the pull width (target 1)

This replaces the earlier robust median + half-68% (and MAD) estimators.
The [lo, hi] window (default +/-4) truncates pathological RooFit outliers,
which is the conventional way HEP calibration notes keep the gaus fit stable.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

LO, HI, NBINS = -4.0, 4.0, 32


def gauss_amp(x, amp, mu, sigma):
    """Gaussian with free amplitude (for binned histogram fits)."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_pull_fit(pulls, lo=LO, hi=HI, nbins=NBINS):
    """Binned Gaussian fit to a pull array in [lo, hi].

    Returns a dict: {mu, mu_err, sigma, sigma_err, amp, n, counts, edges}.
    On failure (too few entries or non-converging fit) the fit params are NaN
    but counts/edges are still returned so the caller can still draw the hist.
    """
    pulls = np.asarray(pulls, dtype=float)
    pulls = pulls[np.isfinite(pulls)]
    edges = np.linspace(lo, hi, nbins + 1)
    counts, _ = np.histogram(pulls, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = {"mu": np.nan, "mu_err": np.nan, "sigma": np.nan, "sigma_err": np.nan,
           "amp": np.nan, "n": int(pulls.size),
           "counts": counts, "edges": edges, "centers": centers}
    if pulls.size < 5:
        return out
    p0 = [max(float(counts.max()), 1.0),
          float(np.median(pulls)),
          max(float(np.std(pulls)), 0.1)]
    try:
        popt, pcov = curve_fit(gauss_amp, centers, counts, p0=p0, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        out.update(amp=float(popt[0]), mu=float(popt[1]),
                   sigma=abs(float(popt[2])),
                   mu_err=float(perr[1]), sigma_err=float(perr[2]))
    except Exception:
        pass
    return out
