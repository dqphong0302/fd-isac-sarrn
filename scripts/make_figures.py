#!/usr/bin/env python3
"""
make_figures.py — publication figures/tables for the PLOS revision from results/fig3_final/frames.csv
(+ results/radar/radar_summary.csv). One run, same frames for every method, no smoothing.

NMSE: mean over frames in linear scale, reported in dB, 95 % CI by percentile bootstrap (2000 resamples).
BER : pooled errors / pooled bits over all frames, 95 % Clopper–Pearson interval
      (zero errors -> one-sided upper bound shown as a down-pointing marker).
Outputs (PNG 300 dpi, PDF, TIFF-LZW 600 dpi) + CSV tables + tables.md.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NC, NS = 1024, 512
STY = {  # (np, method): label, color, marker, linestyle
    (1, "ls"): ("LS, 1 pilot", "#7f7f7f", "o", "--"),
    (2, "ls"): ("LS, 2 pilots", "#1f77b4", "s", "--"),
    (2, "rls"): ("RLS, 2 pilots (reset per frame)", "#17becf", "v", "--"),
    (2, "rls_carryP"): ("RLS, 2 pilots (P not reset, as submitted)", "#bcbd22", "1", ":"),
    (1, "lmmse"): ("LMMSE (known PDP), 1 pilot", "#9467bd", "D", "--"),
    (1, "ridge"): ("Ridge anchor (no NN), 1 pilot", "#8c564b", "x", ":"),
    (1, "sarrn_v36"): ("SARRN as submitted, 1 pilot", "#e377c2", "^", "-."),
    (1, "sarrn_etilde"): ("SARRN retrained on residual, 1 pilot", "#ff7f0e", "<", "-."),
    (1, "ddls"): ("DD-LS (semi-blind, no NN), 1 pilot", "#2ca02c", ">", "-"),
    (1, "sarrn_da"): ("SARRN-DA (proposed), 1 pilot", "#d62728", "*", "-"),
    (1, "genie"): ("Perfect CSI", "#000000", "", "-"),
}
FIG_NMSE = [(1, "ls"), (2, "ls"), (2, "rls"), (1, "lmmse"), (1, "sarrn_v36"), (1, "ddls"), (1, "sarrn_da")]
FIG_BER = [(1, "ls"), (2, "ls"), (2, "rls"), (1, "ddls"), (1, "sarrn_da"), (1, "genie")]


def boot_ci(x, n=2000, seed=0):
    r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".tif"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="results/fig3_final/frames.csv")
    ap.add_argument("--radar", default="results/radar/radar_summary.csv")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.concat([pd.read_csv(p) for p in a.frames.split(",")], ignore_index=True)
    df = df[df.fb == df.groupby(["np", "method"]).fb.transform("max")]
    rows = []
    for (Np, m, g), d in df.groupby(["np", "method", "gamma_db"]):
        x = d.nmse.values
        lo, hi = boot_ci(x)
        nbits = (NS - Np) * NC * len(d)
        k = int(round((d.ber.values * (NS - Np) * NC).sum()))
        blo = beta.ppf(0.025, k, nbits - k + 1) if k > 0 else 0.0
        bhi = beta.ppf(0.975, k + 1, nbits - k)
        rows.append(dict(np=Np, method=m, gamma_db=g, frames=len(d), nmse_db=10 * np.log10(x.mean()),
                         nmse_lo_db=10 * np.log10(lo), nmse_hi_db=10 * np.log10(hi),
                         nmse_median_db=10 * np.log10(np.median(x)),
                         bits=nbits, bit_errors=k, ber=k / nbits, ber_lo=blo, ber_hi=bhi))
    T = pd.DataFrame(rows).sort_values(["np", "method", "gamma_db"])
    T.to_csv(os.path.join(a.out, "nmse_ber_table.csv"), index=False)

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    # ---- NMSE figure
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for key in FIG_NMSE:
        d = T[(T.np == key[0]) & (T.method == key[1])]
        if d.empty:
            continue
        lab, c, mk, ls = STY[key]
        ax.plot(d.gamma_db, d.nmse_db, ls=ls, color=c, marker=mk, ms=4, lw=1.2, label=lab)
        ax.fill_between(d.gamma_db, d.nmse_lo_db, d.nmse_hi_db, color=c, alpha=0.18, lw=0)
    ax.set_xlabel(r"Communication-to-noise ratio $\gamma_c$ (dB)")
    ax.set_ylabel("NMSE (dB)"); ax.grid(ls=":", lw=0.5); ax.legend(fontsize=6.5, ncol=1, loc="lower left")
    save(fig, a.out, "Fig3_NMSE")

    # ---- NMSE figure, pilot-matched panels
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.3), sharey=True)
    for ax, Np in zip(axs, (1, 2)):
        for (n, m), (lab, c, mk, ls) in STY.items():
            if n != Np or m == "genie":
                continue
            d = T[(T.np == Np) & (T.method == m)]
            if d.empty:
                continue
            ax.plot(d.gamma_db, d.nmse_db, ls=ls, color=c, marker=mk, ms=3.5, lw=1.1, label=lab.replace(f", {Np} pilots", "").replace(f", {Np} pilot", "").replace("pilots (", "(").replace("RLS, 2 pilots", "RLS"))
            ax.fill_between(d.gamma_db, d.nmse_lo_db, d.nmse_hi_db, color=c, alpha=0.18, lw=0)
        ax.set_title(f"{Np} pilot symbol{'s' if Np > 1 else ''} (all methods)")
        ax.set_xlabel(r"$\gamma_c$ (dB)"); ax.grid(ls=":", lw=0.5); ax.legend(fontsize=6)
    axs[0].set_ylabel("NMSE (dB)")
    save(fig, a.out, "S2_Fig_pilot_matched_NMSE")

    # ---- BER figure
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for key in FIG_BER:
        d = T[(T.np == key[0]) & (T.method == key[1])]
        if d.empty:
            continue
        lab, c, mk, ls = STY[key]
        nz = d[d.bit_errors > 0]
        ax.errorbar(nz.gamma_db, nz.ber, yerr=[nz.ber - nz.ber_lo, nz.ber_hi - nz.ber], ls=ls, color=c,
                    marker=mk or ".", ms=4, lw=1.1, capsize=2, label=lab)
        z = d[d.bit_errors == 0]
        if len(z):
            ax.plot(z.gamma_db, z.ber_hi, ls="none", color=c, marker="v", ms=3, alpha=0.6)
    ax.set_yscale("log"); ax.set_xlabel(r"$\gamma_c$ (dB)"); ax.set_ylabel("BER")
    ax.grid(ls=":", lw=0.5, which="both"); ax.legend(fontsize=6.5)
    ax.text(0.98, 0.98, "$\\triangledown$: zero errors, 95% upper bound", transform=ax.transAxes,
            ha="right", va="top", fontsize=6)
    save(fig, a.out, "Fig6_BER")

    # ---- radar figure (Pd and false detections vs gamma_c)
    if os.path.exists(a.radar):
        R = pd.read_csv(a.radar)
        fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.0))
        for (mode, fb), d in R.groupby(["mode", "fb"]):
            lab = f"{'corrected' if mode == 'fixed' else 'submitted'} periodogram, iteration {fb}"
            c = ("#d62728" if fb == 2 else "#ff9896") if mode == "fixed" else ("#1f77b4" if fb == 2 else "#aec7e8")
            ls = "-" if mode == "fixed" else "--"
            mk = "o" if fb == 2 else "s"
            d = d.sort_values("gamma_db")
            pd_ = d.Pd.values
            n = 3 * d.frames.values
            axs[0].plot(d.gamma_db, pd_, ls=ls, color=c, marker=mk, ms=4, label=lab)
            if n is not None:
                lo = beta.ppf(0.025, pd_ * n, (1 - pd_) * n + 1); hi = beta.ppf(0.975, pd_ * n + 1, (1 - pd_) * n)
                axs[0].fill_between(d.gamma_db, lo, hi, color=c, alpha=0.18, lw=0)
            axs[1].plot(d.gamma_db, d.false_per_frame, ls=ls, color=c, marker=mk, ms=4, label=lab)
        axs[0].set_ylabel("Detection probability $P_d$"); axs[0].set_ylim(0, 1)
        axs[1].set_ylabel("False detections per frame")
        for ax in axs:
            ax.set_xlabel(r"$\gamma_c$ (dB)"); ax.grid(ls=":", lw=0.5)
        axs[0].legend(fontsize=6, loc="center right")
        save(fig, a.out, "Fig4_Radar")

    # ---- markdown tables
    G = [0, 4, 10, 14, 20, 30]
    md = ["## NMSE (dB), mean [95% bootstrap CI], final GSFDIC iteration\n",
          "| Method | " + " | ".join(f"γc={g}" for g in G) + " |", "|---|" + "---|" * len(G)]
    for (n, m), (lab, *_r) in STY.items():
        d = T[(T.np == n) & (T.method == m)].set_index("gamma_db")
        if d.empty:
            continue
        md.append(f"| {lab} | " + " | ".join(
            f"{d.loc[g,'nmse_db']:.1f} [{d.loc[g,'nmse_lo_db']:.1f}, {d.loc[g,'nmse_hi_db']:.1f}]" if g in d.index else "–"
            for g in G) + " |")
    md += ["\n## BER, pooled [95% Clopper–Pearson]; bits per point = frames × (512 − Np) × 1024\n",
           "| Method | " + " | ".join(f"γc={g}" for g in G) + " |", "|---|" + "---|" * len(G)]
    for (n, m), (lab, *_r) in STY.items():
        d = T[(T.np == n) & (T.method == m)].set_index("gamma_db")
        if d.empty:
            continue
        md.append(f"| {lab} | " + " | ".join(
            (f"{d.loc[g,'ber']:.2e} [{d.loc[g,'ber_lo']:.1e}, {d.loc[g,'ber_hi']:.1e}]" if d.loc[g, 'bit_errors'] > 0
             else f"0 (< {d.loc[g,'ber_hi']:.1e})") if g in d.index else "–" for g in G) + " |")
    if os.path.exists(a.radar):
        R = pd.read_csv(a.radar)
        md += ["\n## Radar (random 3-target scenes), gate = one native cell (1.875 m, 4.58 m/s)\n",
               "| Periodogram | γc | iteration | Pd | false det./frame | RMSE r (m) | RMSE v (m/s) |", "|---|---|---|---|---|---|---|"]
        for _, r in R.iterrows():
            md.append(f"| {r['mode']} | {r.gamma_db:.0f} | {r.fb} | {r.Pd:.3f} | {r.false_per_frame:.2f} | "
                      f"{r.rmse_r_m:.3f} | {r.rmse_v_mps:.3f} |")
    open(os.path.join(a.out, "tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
