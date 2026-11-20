"""The pull statistics of a toy ensemble, the CMS way: a Gaussian fitted to the histogram of
(N_fit - N_inj) / sigma_fit, its mean read as the bias and its sigma as the check of the fit error
(CMS-EXO-17-005 Sec. 4.1: the pull distributions are "consistent with a normal distribution with a
mean less than 0.5 and a width consistent with unity"; CMS-EXO-24-005 Sec. 7.3; the Combine
parametric tutorial's bias section, which fits a Gaussian to the pull distribution and defines
the bias as its mean). No robust estimator (MAD) anywhere.

The fit is a binned Poisson maximum-likelihood fit of A Gauss(mu, sigma) to the histogram over
PULL_RANGE (40 bins below 5000 toys, 100 above, the explainer's binning); entries outside the
range do not enter, as in a ROOT histogram fit. The quoted errors are the asymptotic ones,
sigma / sqrt(N) and sigma / sqrt(2N).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

PULL_RANGE = (-5.0, 5.0)
NAN = float("nan")


def pull_bins(n):
    return 100 if n >= 5000 else 40


def gauss_fit(pulls, lo=PULL_RANGE[0], hi=PULL_RANGE[1], nbins=None):
    """Fit a Gaussian to the histogram of `pulls`. Returns a dict with mean, sigma (and their
    asymptotic errors), the sample mean and std, the binning, a Pearson chi2 over the populated
    bins and `ok` (False when fewer than 10 pulls or the fit failed: mean and sigma are then NaN)."""
    p = np.asarray(pulls, float)
    p = p[np.isfinite(p)]
    out = {"n": int(len(p)), "mean": NAN, "mean_err": NAN, "sigma": NAN, "sigma_err": NAN,
           "sample_mean": float(p.mean()) if len(p) else NAN, "sample_std": float(p.std()) if len(p) else NAN,
           "norm": NAN, "bin_width": NAN, "nbins": 0, "n_in": 0, "chi2": NAN, "ndf": 0, "ok": False}
    if len(p) < 10:
        return out
    nb = nbins or pull_bins(len(p))
    q = p[(p >= lo) & (p < hi)]
    if len(q) < 10:
        return out
    counts, edges = np.histogram(q, bins=nb, range=(lo, hi))
    c = 0.5 * (edges[:-1] + edges[1:])
    bw = float(edges[1] - edges[0])

    def expected(theta):
        a, mu, s = theta
        return a * bw * np.exp(-0.5 * ((c - mu) / s) ** 2) / (s * math.sqrt(2.0 * math.pi))

    def nll(theta):
        a, mu, s = theta
        if s <= 0 or a <= 0:
            return 1e300
        lam = np.maximum(expected(theta), 1e-300)
        return float(np.sum(lam - counts * np.log(lam)))

    x0 = [float(len(q)), float(q.mean()), max(float(q.std()), 1e-3)]
    res = minimize(nll, x0, method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 40000, "maxfev": 40000})
    if not res.success or not np.all(np.isfinite(res.x)) or res.x[2] <= 0:
        return out
    a, mu, s = (float(v) for v in res.x)
    lam = expected(res.x)
    m = lam >= 1.0                      # the Pearson chi2 over the bins the fit expects to populate (the far
    n_in = int(len(q))                  # tails of 100k toys would otherwise dominate it with single entries)
    out.update({"mean": mu, "sigma": s, "mean_err": s / math.sqrt(n_in), "sigma_err": s / math.sqrt(2.0 * n_in),
                "norm": a, "bin_width": bw, "nbins": nb, "n_in": n_in,
                "chi2": float(np.sum((counts[m] - lam[m]) ** 2 / lam[m])), "ndf": int(m.sum()) - 3, "ok": True})
    return out


def gauss_curve(fit, x):
    """The fitted Gaussian in counts per histogram bin, evaluated at x."""
    a, mu, s, bw = fit["norm"], fit["mean"], fit["sigma"], fit["bin_width"]
    return a * bw * np.exp(-0.5 * ((np.asarray(x, float) - mu) / s) ** 2) / (s * math.sqrt(2.0 * math.pi))


def pull_stats(pulls):
    """The row fields of a toy ensemble's pulls: sample mean / RMS / median, and the Gaussian fit
    (pull_width = its sigma, pull_gauss_mean = its mean)."""
    g = gauss_fit(pulls)
    p = np.asarray(pulls, float)
    return {"pull_mean": float(p.mean()), "pull_rms": float(p.std()), "pull_median": float(np.median(p)),
            "pull_gauss_mean": g["mean"], "pull_gauss_mean_err": g["mean_err"],
            "pull_gauss_sigma": g["sigma"], "pull_gauss_sigma_err": g["sigma_err"], "pull_width": g["sigma"]}
