#!/usr/bin/env python3
"""Build LaTeX summary tables of the Minuit2 fit checks per function.

One table per region (resolved / boosted), each pooling the ee and mumu
channels and restricted to W_R masses below MASS_MAX. Reads the
in_window_table_{channel}_{topology}.csv tables, counts how many fits pass each
of the three checks per function, writes fit_checks/<stem>.tex (article +
booktabs) and compiles to PDF with pdflatex when available.

Usage:
  python make_check_table.py [--run run2|run3]
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_args = argparse.ArgumentParser(description=__doc__)
_args.add_argument("--run", default="run3", choices=["run2", "run3"],
                   help="Which run's in_window_table CSVs to summarize.")
RUN_SUB = _args.parse_args().run
TABLE_DIR = HERE / RUN_SUB
sys.path.insert(0, str(HERE.parents[0]))
from bkg_fit_lib import FUNCS  # noqa: E402

MASS_MAX = 4000.0       # only W_R masses below 4 TeV
REGIONS = [             # (label, pooled categories, output stem)
    ("Resolved", ["ee_resolved", "mumu_resolved"], "fit_checks_resolved"),
    ("Boosted",  ["ee_boosted", "mumu_boosted"],   "fit_checks_boosted"),
]
CHECKS = [
    ("valid_minimum", "valid minimum"),
    ("cov_ok", "covariance accurate"),
    ("no_param_at_limit", "no par.\\ at limit"),
    ("monotonic", "monotonic (falling)"),
]
# LaTeX form of each function (matches bkg_fit_lib.FUNCS display formulas;
# recentered at m_c, slope in (m-m_c)/1000, power laws use m/m_c; spelled out, no u).
TEX_FORMULA = {
    "expo":   r"$e^{a+b(m-m_{c})/1000}$",
    "expo2":  r"$e^{a+b(m-m_{c})/1000+c[(m-m_{c})/1000]^{2}}$",
    "expo3":  r"$e^{a+b(m-m_{c})/1000+c[(m-m_{c})/1000]^{2}+d[(m-m_{c})/1000]^{3}}$",
    "powlaw": r"$e^{a}\,(m/m_{c})^{b}$",
    "powexp": r"$(m/m_{c})^{b}\,e^{a+c(m-m_{c})/1000}$",
    "dexp":   r"$e^{a_1+b_1(m-m_{c})/1000}+e^{a_2+b_2(m-m_{c})/1000}$",
}


def collect(categories, mass_max):
    """Pool the given categories, counting only fits with m_WR < mass_max."""
    counts = {f: {"attempted": 0, "notfit": 0, "all6": 0,
                  **{k: 0 for k, _ in CHECKS}} for f in FUNCS}
    masses, kset = set(), set()
    for cat in categories:
        path = TABLE_DIR / f"in_window_table_{cat}.csv"
        if not path.exists():
            print(f"WARNING: missing {path.name}, skipping")
            continue
        for r in csv.DictReader(open(path)):
            if float(r["mWR"]) >= mass_max:
                continue
            masses.add(int(float(r["mWR"])))
            kset.add(r["k"])
            c = counts[r["function"]]
            if r["fit_ok"] != "True":
                c["notfit"] += 1
                continue
            c["attempted"] += 1
            for key, _ in CHECKS:
                c[key] += r[key] == "True"
            c["all6"] += r["fit_passed"] == "True"
    return counts, masses, kset


def build_lines(counts, masses, kset, region_label):
    m_lo, m_hi = min(masses) / 1000, max(masses) / 1000          # TeV
    kval = f"{float(next(iter(kset))):g}" if len(kset) == 1 else "k"
    region = region_label.lower()

    rot = "".join(rf" & \rotatebox{{60}}{{{lab}}}" for _, lab in CHECKS)
    ncheck = len(CHECKS)
    ncol = 3 + ncheck + 1            # function f(m) pars | checks | all
    clo, chi = 4, 3 + ncheck         # check columns span (for \cmidrule)
    lines = [
        r"\documentclass{article}",   # standalone.cls unavailable on this node
        r"\usepackage[paperwidth=20.6cm,paperheight=10.5cm,margin=6mm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{amsmath}",
        r"\pagestyle{empty}",
        r"\begin{document}",
        r"\noindent",
        rf"{{\large\textbf{{In-window background-fit quality --- {region_label}}}}}\\[2pt]",
        rf"ee \& $\mu\mu$, {region};\ "
        rf"$m_{{W_R}}={m_lo:g}$--${m_hi:g}$\,TeV ($<4$\,TeV), "
        rf"$m_{{c}}\pm{kval}\sigma$ windows\\[6pt]",
        r"\begin{tabular}{l l r " + "c" * ncheck + r" c}",
        r"\toprule",
        rf" & & & \multicolumn{{{ncheck}}}{{c}}{{fits passing each check}} & \\",
        rf"\cmidrule(lr){{{clo}-{chi}}}",
        r"function & $f(m)$ & pars" + rot
        + r" & \rotatebox{60}{\textbf{all}} \\",
        r"\midrule",
    ]
    # rows ordered by number of parameters (stable sort: ties keep FUNCS order)
    for name, (_c, _f, npar) in sorted(FUNCS.items(), key=lambda kv: kv[1][2]):
        c = counts[name]
        if c["attempted"] + c["notfit"] == 0:
            continue
        cells = "".join(f" & {c[k]}" for k, _ in CHECKS)
        lines.append(
            rf"{name} & {TEX_FORMULA[name]} & {npar}" + cells
            + rf" & \textbf{{{c['all6']}}} \\")
    n_win = counts["expo"]["attempted"] + counts["expo"]["notfit"]
    lines += [
        r"\bottomrule",
        rf"\multicolumn{{{ncol}}}{{l}}{{\footnotesize ROOT TF1 + Minuit2 $\chi^2$ "
        rf"fit, slope in $(m-m_{{c}})/1000$; {n_win} windows per function "
        rf"(ee\,$+$\,$\mu\mu$ {region}, $m_{{W_R}}<4$\,TeV).}}\\",
        rf"\multicolumn{{{ncol}}}{{l}}{{\footnotesize Four checks. "
        rf"TFitResult::IsValid(): a converged minimum (EDM, call limit, Hesse); "
        rf"covariance accurate: CovMatrixStatus()$=3$ (not forced pos-def).}}\\",
        rf"\multicolumn{{{ncol}}}{{l}}{{\footnotesize no par.\ at limit: no slope "
        rf"railed against the $\le0$ bound (dexp also fences its norms); "
        rf"monotonic: the fit falls across the window (no local rise). "
        rf"pass $=$ all four hold.}}\\",
        r"\end{tabular}",
        r"\end{document}",
    ]
    return lines


def compile_table(lines, out_stem):
    out_dir = TABLE_DIR / "fit_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    tex = out_dir / f"{out_stem}.tex"
    tex.write_text("\n".join(lines) + "\n")
    print(f"wrote {tex}")

    if not shutil.which("pdflatex"):
        print("pdflatex not found; .tex written only")
        return
    # sanitized env: the LCG view's LD_LIBRARY_PATH breaks system pdflatex
    import os
    clean_env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp")}
    r = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex.name],
        cwd=out_dir, capture_output=True, text=True, env=clean_env)
    if (out_dir / f"{out_stem}.pdf").exists() and r.returncode == 0:
        for ext in (".aux", ".log"):
            (out_dir / f"{out_stem}{ext}").unlink(missing_ok=True)
        print(f"wrote {out_dir / (out_stem + '.pdf')}")
    else:
        print(f"pdflatex failed -- see {out_stem}.log")
        sys.exit(1)


def main():
    for region_label, categories, out_stem in REGIONS:
        counts, masses, kset = collect(categories, MASS_MAX)
        if not masses:
            print(f"WARNING: no masses < {MASS_MAX:g} for {region_label}, skipping")
            continue
        lines = build_lines(counts, masses, kset, region_label)
        compile_table(lines, out_stem)


if __name__ == "__main__":
    main()
