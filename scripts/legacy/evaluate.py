#!/usr/bin/env python3
"""
evaluate.py — NMSE of LS / Ridge (anchor, no NN) / RLS / LMMSE / SARRN on the
independent test set, per gamma_c bin and per pilot count, with 95 % bootstrap CI.

  python evaluate.py --test data/test.h5 --runs runs --np 1,2,4 --out results

Outputs: results/nmse_by_gamma.csv (long format), results/summary.md,
         results/nmse_vs_gamma_np{k}.png/.pdf
"""
import argparse
import json
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

METHODS = [("ls", "LS"), ("ridge", "Ridge (anchor, no NN)"), ("rls", "RLS (λ=0.999)"),
           ("lmmse", "LMMSE (known PDP)"), ("sarrn", "SARRN (proposed)")]
STYLE = {"ls": ("k", "--", "s"), "ridge": ("tab:orange", "-", "o"), "rls": ("tab:red", ":", "v"),
         "lmmse": ("tab:green", "-.", "d"), "sarrn": ("tab:blue", "-", "^")}


def per_frame_nmse(h, y):
    return np.sum((h - y) ** 2, 1) / np.sum(y ** 2, 1)


def boot_ci(v, rng, n=2000):
    idx = rng.integers(0, len(v), (n, len(v)))
    m = v[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--np", default="1,2,4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)

    with h5py.File(a.test, "r") as f:
        feat, y, gam = f["feat"][:], f["y"][:], np.round(f["gamma_db"][:], 6)
        est = {m: {k: f[f"est/{m}/np{k}"][:] for k in map(int, a.np.split(","))}
               for m in ["ls", "ridge", "rls", "lmmse"]}
        test_seed = int(f.attrs["seed"])
    grid = np.unique(gam)
    rows = []
    for k in map(int, a.np.split(",")):
        meta = json.load(open(os.path.join(a.runs, f"np{k}", "train_meta.json")))
        assert test_seed not in (meta["train_seed"], meta["val_seed"]), "test seed reused!"
        sess = ort.InferenceSession(os.path.join(a.runs, f"np{k}", "sarrn.onnx"),
                                    providers=["CPUExecutionProvider"])
        X = np.concatenate([feat, est["ridge"][k]], 1).astype(np.float32)
        h_dl = sess.run(None, {"x": X})[0].astype(np.float64)
        allh = {m: est[m][k] for m in ["ls", "ridge", "rls", "lmmse"]} | {"sarrn": h_dl}
        for m, _ in METHODS:
            e = per_frame_nmse(allh[m], y)
            for g in grid:
                v = e[gam == g]
                lo, hi = boot_ci(v, rng)
                rows.append(dict(np=k, method=m, gamma_db=float(g), n=len(v), nmse=v.mean(),
                                 nmse_db=10 * np.log10(v.mean()), ci_lo_db=10 * np.log10(lo),
                                 ci_hi_db=10 * np.log10(hi), median_db=10 * np.log10(np.median(v))))
            # paired SARRN-vs-anchor gain per frame
        gain = 10 * np.log10(per_frame_nmse(est["ridge"][k], y) / per_frame_nmse(h_dl, y))
        rows.append(dict(np=k, method="gain_sarrn_vs_ridge", gamma_db=np.nan, n=len(gain),
                         nmse=np.nan, nmse_db=float(np.mean(gain)),
                         ci_lo_db=float(np.percentile(gain, 2.5)), ci_hi_db=float(np.percentile(gain, 97.5)),
                         median_db=float(np.median(gain))))

    import csv
    with open(os.path.join(a.out, "nmse_by_gamma.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    # figures
    for k in map(int, a.np.split(",")):
        fig, ax = plt.subplots(figsize=(6, 4.2), dpi=200)
        for m, lab in METHODS:
            r = [x for x in rows if x["np"] == k and x["method"] == m]
            g = np.array([x["gamma_db"] for x in r]); v = np.array([x["nmse_db"] for x in r])
            lo = np.array([x["ci_lo_db"] for x in r]); hi = np.array([x["ci_hi_db"] for x in r])
            c, ls, mk = STYLE[m]
            ax.plot(g, v, color=c, linestyle=ls, marker=mk, ms=4, label=lab)
            ax.fill_between(g, lo, hi, color=c, alpha=0.15)
        ax.set_xlabel(r"$\gamma_c$ (dB)"); ax.set_ylabel("NMSE (dB)")
        ax.set_title(f"Np = {k} pilot OFDM symbol{'s' if k > 1 else ''} (test set, 95% CI)")
        ax.grid(True, ls=":"); ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(a.out, f"nmse_vs_gamma_np{k}.{ext}"))
        plt.close(fig)

    # summary table
    pick = [0, 10, 20, 30]
    lines = ["| Np | γc (dB) | " + " | ".join(l for _, l in METHODS) + " |",
             "|---|---|" + "---|" * len(METHODS)]
    for k in map(int, a.np.split(",")):
        for g in pick:
            cells = []
            for m, _ in METHODS:
                r = [x for x in rows if x["np"] == k and x["method"] == m and abs(x["gamma_db"] - g) < 1e-6]
                cells.append(f"{r[0]['nmse_db']:.2f}" if r else "–")
            lines.append(f"| {k} | {g} | " + " | ".join(cells) + " |")
    lines.append("")
    for k in map(int, a.np.split(",")):
        r = [x for x in rows if x["np"] == k and x["method"] == "gain_sarrn_vs_ridge"][0]
        lines.append(f"- Np={k}: per-frame NMSE gain SARRN vs Ridge anchor: mean {r['nmse_db']:.2f} dB, "
                     f"median {r['median_db']:.2f} dB, 95% range [{r['ci_lo_db']:.2f}, {r['ci_hi_db']:.2f}] dB")
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
