#!/usr/bin/env python3
"""Summarise radar_by_estimator.py output: one row per (gamma_c, radar pass, estimator).

Usage: python radar_by_estimator_summary.py <folder with frames_*.csv>
Writes summary.csv and summary.md into the same folder.
bg_vs_perfect_db: mean over frames of 10 log10(bg / bg with perfect CSI on the same frame); 0 = equally clean map.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

d = sys.argv[1]
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(d, "frames_*.csv")))], ignore_index=True)
ref = df[df.est == "perfect"].set_index(["gamma_db", "frame", "pass"]).bg_lin
df["bg_rel_db"] = 10 * np.log10(df.bg_lin.values / ref.reindex(pd.MultiIndex.from_frame(df[["gamma_db", "frame", "pass"]])).values)


def errs(s):
    return np.array([float(x) for cell in s.dropna().astype(str) for x in cell.split(";") if x not in ("", "nan")])


rows = []
order = ["ls", "rls", "lmmse", "ddls", "sarrn", "perfect"]
for (g, k, e), x in df.groupby(["gamma_db", "pass", "est"]):
    n = len(x); m = x.n_match.sum()
    er, ev = errs(x.err_r), errs(x.err_v)
    rows.append(dict(gamma_db=g, radar_pass=k, est=e, frames=n, Pd=m / (3 * n), false_per_frame=x.n_false.mean(),
                     rmse_r_m=np.sqrt(x.se_r.sum() / max(m, 1)), rmse_v_mps=np.sqrt(x.se_v.sum() / max(m, 1)),
                     median_abs_r_m=np.median(er) if len(er) else np.nan,
                     median_abs_v_mps=np.median(ev) if len(ev) else np.nan,
                     bg_vs_perfect_db=x.bg_rel_db.mean(), bg_vs_perfect_db_max=x.bg_rel_db.max(),
                     residual_com_db=x.res_db.replace(-np.inf, np.nan).mean()))
T = pd.DataFrame(rows)
T["o"] = T.est.map({e: i for i, e in enumerate(order)})
T = T.sort_values(["gamma_db", "radar_pass", "o"]).drop(columns="o")
T.to_csv(os.path.join(d, "summary.csv"), index=False)
with open(os.path.join(d, "summary.md"), "w") as f:
    try:
        f.write(T.to_markdown(index=False, floatfmt=".3f"))   # needs the optional 'tabulate' package
    except ImportError:
        f.write(T.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(T.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
