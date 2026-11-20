#!/usr/bin/env python3
"""The window-width matrix at 100k toys per cell, every dataset on one page.

Reads 6_windows/k<k>_lo1200<tag>/lo_ht/toys_{raw,kde9}.csv (6_windows.py --tag) of every dataset that has them:
2024 (109.7 fb-1, results/), the 2024 MC x 2.58 (283.3 fb-1, results_run3scaled/) and the summed 2022-2025 MC
(283.3 fb-1, results_run3/); the half-widths 3, 5, 7 and 10 sigma and the full range from 1200 GeV; the
exponential and the power x exponential; the raw MC and the nine-process RooKeysPdf without tails. Writes
background_fit/MATRIX<tag>.md:

- at a glance: the +-5 sigma row of every table;
- per function and truth: rows = widths, columns = dataset x channel, each cell "last passing mass; cells beyond
  0.5 / beyond 0.3; largest |mu| (at mass)" over the float regime; then sigma_fit(width) / sigma_fit(full range)
  at 2, 3 and 3.4 TeV; then the Gaussian sigma of the pulls (min-max);
- the power x exponential's float-regime cells beyond 0.5, and those within 0.1 of the line (0.4 <= |mu| <= 0.6).

Float regime: the cells above the floor cell (1200 GeV is a half window clipped at the floor and does not count,
as in 4_toys.first_sparse and 14_luminosity.py) with >= 90 % of their zero-injection fits passing the checks
(4_toys.pass_fractions, passed / converged), below the first sparse mass. mu, sigma: the Gaussian fitted to the
zero-injection pulls. With 100k toys the error on a cell's mu is 0.003-0.004, so differences above 0.01 between
cells are the cells', not the toys' (500 toys: 0.045).

  cd WR_Plotter && python background_fit/16_matrix.py [--tag _100k]
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))
import config as C                                                       # noqa: E402
from util import read_csv                                                # noqa: E402

T = importlib.import_module("4_toys")                                    # noqa: E402
logger = logging.getLogger("16_matrix")
PASS_FRAC = T.PASS_FRAC
FLOOR = 1200.0
WIDTHS = [(3, "+-3 sigma"), (5, "+-5 sigma"), (7, "+-7 sigma"), (10, "+-10 sigma"), (100, "full range")]
WLAB = dict(WIDTHS)
DATASETS = ["2024", "run3_scaled", "run3"]
DS_SHORT = {"2024": "2024", "run3_scaled": "2024 MC x 2.58", "run3": "summed 2022-2025"}
TRUTHS = ["raw", "kde9"]
TRUTH_LABEL = {"raw": "the raw MC", "kde9": "the nine-process RooKeysPdf (no tails)"}
CANDS = ["expo", "powexp"]
CAND_LABEL = {"expo": "the exponential", "powexp": "the power x exponential"}
REF_MASSES = (2000.0, 3000.0, 3400.0)


def load(ds, channel, k, truth, tag):
    p = HERE / C.DATASETS[ds]["results"] / channel / "6_windows" / f"k{k}_lo1200{tag}" / "lo_ht" / f"toys_{truth}.csv"
    return [T._type_row(r) for r in read_csv(p)] if p.exists() else []


def series(rows, cand, key):
    """(masses, values, >= 90 % passing) of the zero-injection cells, in mass order."""
    x, y = T._series(rows, cand, 0.0, key)
    frac = dict(T.pass_fractions(rows, cand))
    x, y = np.array(x, float), np.array(y, float)
    return x, y, np.array([frac.get(m_, 0.0) >= PASS_FRAC for m_ in x], bool)


class Cell:
    """One configuration (dataset, channel, width, truth, function): its float regime and the numbers quoted."""

    def __init__(self, rows, cand):
        x, mu, good = series(rows, cand, "pull_gauss_mean")
        self.sparse = T.first_sparse(rows, cand, PASS_FRAC)
        hi = self.sparse if self.sparse else 1e9
        self.x, self.mu, self.good = x, mu, good
        self.fl = good & (x < hi) & (x > FLOOR)
        self.last = float(x[self.fl].max()) if self.fl.any() else None
        sx, sy, _ = series(rows, cand, "pull_width")
        self.sig = dict(zip(sx.tolist(), sy.tolist()))
        ax_, ay_, _ = series(rows, cand, "asimov_err")
        self.sfit = dict(zip(ax_.tolist(), ay_.tolist()))
        self.ntoys = rows[0].get("n_toys") if rows else None

    def in_float(self, m_):
        j = np.nonzero(self.x == m_)[0]
        return bool(len(j)) and bool(self.fl[j[0]])

    def beyond(self, thr):
        return [(float(m_), float(v)) for m_, v, g in zip(self.x, self.mu, self.fl) if g and abs(v) > thr]

    def near(self, lo=0.4, hi=0.6):
        return [(float(m_), float(v)) for m_, v, g in zip(self.x, self.mu, self.fl) if g and lo <= abs(v) <= hi]

    def maxmu(self):
        if not self.fl.any():
            return None
        i = int(np.argmax(np.where(self.fl, np.abs(self.mu), -1.0)))
        return float(self.mu[i]), float(self.x[i])

    def sigrange(self):
        v = [self.sig[m_] for m_ in self.x[self.fl].tolist() if m_ in self.sig and np.isfinite(self.sig[m_])]
        return (min(v), max(v)) if v else None

    def summary(self):
        mx = self.maxmu()
        last = "-" if self.last is None else f"{self.last:.0f}"
        return f"{last}; {len(self.beyond(C.BIAS_PASS))} / {len(self.beyond(C.BIAS_TARGET))}; " + (f"{mx[0]:+.2f} ({mx[1]:.0f})" if mx else "-")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="_100k", help="folder suffix of the grids (6_windows.py --tag) and of the markdown file")
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=sorted(C.DATASETS))
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--out", default=None, help="the markdown file; default background_fit/MATRIX<tag>.md")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cells, ntoys, missing = {}, set(), []
    for ds in args.datasets:
        for ch in args.channels:
            for k, _ in WIDTHS:
                for tr in TRUTHS:
                    rows = load(ds, ch, k, tr, args.tag)
                    if not rows:
                        missing.append(f"{DS_SHORT[ds]} {ch} {WLAB[k]} {tr}")
                        continue
                    for cand in CANDS:
                        c = Cell(rows, cand)
                        cells[(ds, ch, k, tr, cand)] = c
                        if c.ntoys:
                            ntoys.add(int(c.ntoys))
    cols = [(ds, ch) for ds in args.datasets for ch in args.channels
            if any((ds, ch, k, tr, cand) in cells for k, _ in WIDTHS for tr in TRUTHS for cand in CANDS)]
    head = "| width | " + " | ".join(f"{DS_SHORT[ds]} {ch}" for ds, ch in cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    sets = "; ".join(f"{DS_SHORT[ds]} = {C.DATASETS[ds]['results']} ({C.DATASETS[ds]['lumi']:.1f} fb-1)" for ds in args.datasets)
    L = [f"# The window-width matrix at {' / '.join(f'{n:,}' for n in sorted(ntoys)) or '?'} toys per cell (6_windows/k<k>_lo1200{args.tag}, lo_ht, the 1200 GeV floor)", "",
         f"Datasets: {sets}. Widths: +-3, 5, 7, 10 sigma around the signal mass and the full range from 1200 GeV. "
         "Truths: the raw MC and the nine-process RooKeysPdf without fitted tails. Functions: the exponential and the "
         "power x exponential. Float regime: the cells above the 1200 GeV cell (a half window clipped at the floor) with "
         ">= 90 % of the zero-injection fits passing the checks (passed / converged), below the first sparse mass; "
         "mu and sigma are the Gaussian fitted to the zero-injection pulls. With 100k toys the error on a cell's mu is "
         "0.003-0.004 (500 toys: 0.045). Each cell below: last passing mass; cells beyond 0.5 / beyond 0.3; largest |mu| (at mass).", ""]
    if missing:
        L += ["Missing grids: " + ", ".join(missing) + ".", ""]
    # at a glance: the +-5 sigma row of every table
    L += ["## At a glance: the +-5 sigma window", "", "| function, truth | " + " | ".join(f"{DS_SHORT[ds]} {ch}" for ds, ch in cols) + " |", sep]
    for cand in CANDS:
        for tr in TRUTHS:
            L.append(f"| {cand} on {tr} | " + " | ".join(cells[(ds, ch, 5, tr, cand)].summary() if (ds, ch, 5, tr, cand) in cells else "-" for ds, ch in cols) + " |")
    L.append("")
    for cand in CANDS:
        for tr in TRUTHS:
            L += [f"## {CAND_LABEL[cand]} on {TRUTH_LABEL[tr]}", "", head, sep]
            for k, lab in WIDTHS:
                L.append(f"| {lab} | " + " | ".join(cells[(ds, ch, k, tr, cand)].summary() if (ds, ch, k, tr, cand) in cells else "-" for ds, ch in cols) + " |")
            L += ["", f"sigma_fit(width) / sigma_fit(full range) at {' / '.join(f'{m_ / 1000:g}' for m_ in REF_MASSES)} TeV, where both cells are in the float regime:", "", head, sep]
            for k, lab in WIDTHS[:-1]:
                row = []
                for ds, ch in cols:
                    c, ref = cells.get((ds, ch, k, tr, cand)), cells.get((ds, ch, 100, tr, cand))
                    if c is None or ref is None:
                        row.append("-")
                        continue
                    vals = []
                    for m_ in REF_MASSES:
                        ok = m_ in c.sfit and m_ in ref.sfit and ref.sfit[m_] > 0 and c.in_float(m_) and ref.in_float(m_)
                        vals.append(f"{c.sfit[m_] / ref.sfit[m_]:.2f}" if ok else "-")
                    row.append(" / ".join(vals))
                L.append(f"| {lab} | " + " | ".join(row) + " |")
            L += ["", "Gaussian sigma of the pulls over the float regime, min-max:", "", head, sep]
            for k, lab in WIDTHS:
                row = []
                for ds, ch in cols:
                    c = cells.get((ds, ch, k, tr, cand))
                    r = c.sigrange() if c is not None else None
                    row.append(f"{r[0]:.2f}-{r[1]:.2f}" if r else "-")
                L.append(f"| {lab} | " + " | ".join(row) + " |")
            L.append("")
    order = {k: i for i, (k, _) in enumerate(WIDTHS)}

    def keyf(item):
        ds, ch, k, tr, cand = item[0]
        return (args.datasets.index(ds), args.channels.index(ch), order[k], TRUTHS.index(tr))

    for title, pick in [("float-regime cells beyond 0.5", lambda c: c.beyond(C.BIAS_PASS)),
                        ("float-regime cells within 0.1 of the line (0.4 <= |mu| <= 0.6)", lambda c: c.near())]:
        L += [f"## The power x exponential: {title}", ""]
        n = 0
        for (ds, ch, k, tr, cand), c in sorted(cells.items(), key=keyf):
            if cand != "powexp":
                continue
            hits = pick(c)
            if hits:
                n += len(hits)
                L.append(f"- {DS_SHORT[ds]}, {ch}, {WLAB[k]}, {tr}: " + ", ".join(f"{m_:.0f} GeV {v:+.2f}" for m_, v in hits))
        L += ["- none" if n == 0 else f"", ""]
    out = Path(args.out) if args.out else HERE / f"MATRIX{args.tag}.md"
    out.write_text("\n".join(L) + "\n")
    logger.info("wrote %s (%d configurations%s)", out, len(cells), f", {len(missing)} grids missing" if missing else "")


if __name__ == "__main__":
    main()
