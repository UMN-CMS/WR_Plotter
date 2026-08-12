#!/usr/bin/env python3
"""Digitize the official 2018 ee expected limit from the reference PDF.

Renders run2_results/1D_EE_Combined_HalfN_Limit_vs_WR.pdf at 300 dpi and
extracts, per pixel column, the GREEN (68%) band edges; the official median
expected is taken as the geometric mean of the two edges (CLs bands are
log-symmetric in +-1 sigma to <1% -- verified on our own tables). Black
dashed/solid curves are deliberately not traced (the observed line is black
too and crosses the band).

Frame calibration: the ROOT frame spans x = [800, 6000] GeV (linear) and
y = [1e-4, 1e4] fb (log). A verification overlay (digitize_check.png) draws
the detected frame, the computed tick positions, the extracted edges and the
derived median ON TOP of the render -- inspect it before trusting the CSV.

Writes official2018_expected_digitized.csv: mass_GeV, med_fb, lo68_fb, hi68_fb.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
PDF = HERE.parent / "run2_results" / "1D_EE_Combined_HalfN_Limit_vs_WR.pdf"
XLO, XHI = 800.0, 6000.0
YLO, YHI = 1e-4, 1e4

png = HERE / "official2018_render.png"
subprocess.run(["pdftocairo", "-png", "-r", "300", "-singlefile",
                str(PDF), str(png.with_suffix(""))], check=True)
img = np.asarray(Image.open(png).convert("RGB")).astype(int)
H, W, _ = img.shape
dark = img.sum(axis=2) < 240

# frame: the longest dark horizontal/vertical lines
row_runs = dark.sum(axis=1)
col_runs = dark.sum(axis=0)
rows = np.where(row_runs > 0.5 * W)[0]
cols = np.where(col_runs > 0.65 * H)[0]
fy0, fy1 = rows.min(), rows.max()          # top, bottom (pixel y grows down)
fx0, fx1 = cols.min(), cols.max()
print(f"frame px: x [{fx0},{fx1}]  y [{fy0},{fy1}]")


def px_to_mass(px):
    return XLO + (px - fx0) / (fx1 - fx0) * (XHI - XLO)


def py_to_sigma(py):
    frac = (fy1 - py) / (fy1 - fy0)
    return 10 ** (np.log10(YLO) + frac * (np.log10(YHI) - np.log10(YLO)))


def mass_to_px(m):
    return fx0 + (m - XLO) / (XHI - XLO) * (fx1 - fx0)


def sigma_to_py(s):
    frac = (np.log10(s) - np.log10(YLO)) / (np.log10(YHI) - np.log10(YLO))
    return fy1 - frac * (fy1 - fy0)


# green 68% band mask (ROOT kGreen-ish; exclude frame/label greys)
r, g, b = img[..., 0], img[..., 1], img[..., 2]
green = (g > 140) & (r < 130) & (b < 130)
green[:fy0 + 2] = green[fy1 - 1:] = False
green[:, :fx0 + 2] = green[:, fx1 - 1:] = False
# the legend's green swatch sits inside the frame at sigma ~ 1e-3..1e-2;
# the real band never drops below ~0.1 fb -> cut everything below 0.03 fb
y_cut = int(sigma_to_py(0.03))
green[y_cut:, :] = False

# yellow (95%) mask -- its inner edges coincide with the green band's outer
# edges, which lets us bridge the OBSERVED black line where it covers one of
# the colours near a boundary: each 68% edge is taken as the MIDPOINT of the
# green->yellow gap (gap ~ 0 normally; ~ the line thickness where the black
# line straddles the boundary, and the midpoint stays on it).
yellow = (r > 200) & (g > 150) & (b < 130)
yellow[:fy0 + 2] = yellow[fy1 - 1:] = False
yellow[:, :fx0 + 2] = yellow[:, fx1 - 1:] = False
yellow[y_cut:, :] = False

masses, med_px_lo, med_px_hi = [], [], []
for px in range(fx0 + 3, fx1 - 2):
    gcol = np.where(green[:, px])[0]
    ycol = np.where(yellow[:, px])[0]
    if len(gcol) < 4 or len(ycol) < 4:
        continue
    g_top, g_bot = gcol.min(), gcol.max()
    y_top_seg = ycol[ycol < g_top]          # upper yellow band
    y_bot_seg = ycol[ycol > g_bot]          # lower yellow band
    hi_edge = 0.5 * (g_top + (y_top_seg.max() if len(y_top_seg) else g_top))
    lo_edge = 0.5 * (g_bot + (y_bot_seg.min() if len(y_bot_seg) else g_bot))
    masses.append(px_to_mass(px))
    med_px_hi.append(hi_edge)
    med_px_lo.append(lo_edge)

masses = np.array(masses)


def _rollmed(a, w=31):
    """Rolling median along mass; true band edges are piecewise-linear, so
    this kills black-line-width notches without touching real structure."""
    a = np.asarray(a, float)
    out = np.empty_like(a)
    h = w // 2
    for i in range(len(a)):
        out[i] = np.median(a[max(0, i - h):i + h + 1])
    return out


hi68 = py_to_sigma(_rollmed(med_px_hi))
lo68 = py_to_sigma(_rollmed(med_px_lo))
med = np.sqrt(lo68 * hi68)
print(f"extracted {len(masses)} columns, mass span "
      f"[{masses.min():.0f}, {masses.max():.0f}]")

out = HERE / "official2018_expected_digitized.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["mass_GeV", "med_fb", "lo68_fb", "hi68_fb"])
    for row in zip(masses, med, lo68, hi68):
        w.writerow([f"{row[0]:.1f}"] + [f"{v:.5g}" for v in row[1:]])
print(f"wrote {out}")
for m in (1400, 2000, 2600, 3200, 4000, 5000):
    i = np.argmin(np.abs(masses - m))
    print(f"  official median @ {m}: {med[i]:.3f} fb  (68%: "
          f"[{lo68[i]:.3f}, {hi68[i]:.3f}])")

# verification overlay
im = Image.open(png).convert("RGB")
dr = ImageDraw.Draw(im)
dr.rectangle([fx0, fy0, fx1, fy1], outline=(255, 0, 255), width=2)
for m in range(1000, 6001, 1000):                      # x ticks
    x = mass_to_px(m)
    dr.line([x, fy1 - 25, x, fy1 + 25], fill=(255, 0, 255), width=3)
for dec in range(-4, 5):                               # y decades
    y = sigma_to_py(10.0 ** dec)
    dr.line([fx0 - 25, y, fx0 + 25, y], fill=(255, 0, 255), width=3)
for xs, ys, color in [(masses, lo68, (0, 0, 255)), (masses, hi68, (0, 0, 255)),
                      (masses, med, (255, 0, 0))]:
    pts = [(mass_to_px(m), sigma_to_py(s)) for m, s in zip(xs, ys)]
    dr.line(pts, fill=color, width=2)
im.save(HERE / "digitize_check.png")
print("wrote digitize_check.png -- inspect tick alignment + curves")
