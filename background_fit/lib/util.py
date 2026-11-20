"""Small shared helpers: CSV writing, number formatting and the smoothing library."""
from __future__ import annotations

import csv
import importlib
import math
import sys
from pathlib import Path


def smooth_lib():
    """The smoothing library (RooKeysPdf, kernel regression, tail fits). It is not part of this folder: taken from
    the import path, or found by its file name under WR_Plotter/. Only building a smoothed background needs it."""
    try:
        return importlib.import_module("smooth_lib")
    except ModuleNotFoundError:
        pass
    hits = sorted(Path(__file__).resolve().parents[2].glob("*/*/smoothing/smooth_lib.py"))
    if not hits:
        raise SystemExit("smooth_lib.py not found: building a smoothed background needs the smoothing library. "
                         "The simulation as it is (--truths raw) and stored backgrounds need none.")
    sys.path.insert(0, str(hits[0].parent))
    return importlib.import_module("smooth_lib")


class _SmoothLib:
    """smooth_lib, imported on first use."""

    def __getattr__(self, name):
        return getattr(smooth_lib(), name)


sl = _SmoothLib()


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, str):
        return x
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "inf" if v > 0 else ("-inf" if v < 0 else "nan")
    if abs(v) >= 1e5:
        return f"{v:.2e}"
    return f"{v:.{nd}f}"
