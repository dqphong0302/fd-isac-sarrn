#!/usr/bin/env python3
"""
eval_radar.py — quantitative sensing metrics for the GSFDIC chain (Reviewer Major 6).

For each frame: 3 random targets (-10 dB, r ~ U[3,20] m, v ~ U[-25,25] m/s, pairwise separation
>= 1 native resolution cell), random SI slow paths, random comm channel phase, gamma_c fixed per run.
The 2-iteration GSFDIC chain is run with a chosen estimator; detections of each iteration are
matched to the truth (Hungarian, gate = one native cell: dR = c/(2 Nc df) = 1.875 m,
dv = c/(2 fc Ns Ts) = 4.58 m/s).

Metrics per (radar mode, gamma_c, iteration): Pd, false detections / frame, range / velocity RMSE
of matched detections, plus the CA-CFAR raw false-alarm rate (threshold crossings outside the
guard region of any true target / number of tested cells).
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.signal import lfilter  # noqa: F401  (used through gsfdic_chain)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sarrn import sim as S  # noqa: E402
from sarrn import receiver as G  # noqa: E402

DR = S.C0 / (2 * S.NC * S.DELTA_F)
DV = S.C0 / (2 * S.FC * S.NS * S.TS)


def draw_targets(rng, k=3):
    while True:
        r = rng.uniform(3, 20, k); v = rng.uniform(-25, 25, k)
        ok = all(abs(r[i] - r[j]) > DR or abs(v[i] - v[j]) > DV for i in range(k) for j in range(i + 1, k))
        if ok:
            return list(r), list(v)


def match(tr, tv, dr, dv):
    if len(dr) == 0:
        return [], 0
    cost = np.abs(np.subtract.outer(tr, dr)) / DR + np.abs(np.subtract.outer(tv, dv)) / DV
    gate = (np.abs(np.subtract.outer(tr, dr)) <= DR) & (np.abs(np.subtract.outer(tv, dv)) <= DV)
    cost = np.where(gate, cost, 1e6)
    ri, ci = linear_sum_assignment(cost)
    pairs = [(i, j) for i, j in zip(ri, ci) if gate[i, j]]
    return pairs, len(dr) - len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--gammas", default="10,20,30")
    ap.add_argument("--seed", type=int, default=9100)
    ap.add_argument("--est", default="ddls", choices=["ls", "ddls"])
    ap.add_argument("--modes", default="fixed,legacy")
    ap.add_argument("--out", default="results/radar")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    mags = 10 ** (S.HCOM_DB / 20)
    rot = complex(math.cos(S.PSK_PHASE), math.sin(S.PSK_PHASE))
    est = G.DDLS() if a.est == "ddls" else (lambda xp, yp, **k: G.ridge_np(xp, yp, 0.0)[0])

    fcsv = open(os.path.join(a.out, "radar_frames.csv"), "w", newline="")
    w = csv.writer(fcsv)
    w.writerow(["mode", "gamma_db", "frame", "fb", "n_true", "n_det", "n_match", "n_false",
                "se_r_sum", "se_v_sum", "true_r", "true_v", "det_r", "det_v"])
    agg = {}
    t0 = time.time()
    for g in map(float, a.gammas.split(",")):
        rng = np.random.default_rng(a.seed + int(g))
        frames = []
        for i in range(a.n):
            tr, tv = draw_targets(rng)
            ph = 2 * np.pi * rng.random(4)
            si_rng = [1.0] + list(rng.uniform(9, 14, 15))
            si_vel = [0.0] + list(rng.uniform(0.5, 1.5, 15) * rng.choice([-1, 1], 15))
            frames.append((tr, tv, ph, si_rng, si_vel, a.seed * 1000 + int(g) * 10000 + i))
        for mode in a.modes.split(","):
            G.RADAR_LEGACY = (mode == "legacy")
            for i, (tr, tv, ph, si_rng, si_vel, seed) in enumerate(frames):
                gen = torch.Generator(device=dev).manual_seed(seed)
                F = (2.0 * torch.randint(0, 2, (S.NC, S.NS), generator=gen, device=dev) - 1.0).to(torch.complex128)
                Ytg = S.generate_echo([-10.0] * 3, tv, tr, F, dev)
                Ysi = S.generate_echo(G.SI_POW, si_vel, si_rng, F, dev)
                Rn = S.generate_noise(0.0, gen, dev)
                bits = torch.randint(0, 2, (S.NC, S.NS), generator=gen, device=dev)
                Fcom = (1.0 - 2.0 * bits.double()) * rot
                x_td = S.ifft_c(Fcom)
                h_true = mags * np.exp(1j * ph) * math.sqrt(10 ** (g / 10))
                y = S.fir_channel(S.col_flatten(x_td), torch.as_tensor(h_true, device=dev))
                A = S.fft_c(Ytg + Ysi + S.col_unflatten(y, S.NC, S.NS)) + Rn
                res = G.gsfdic_chain(A, F, x_td, Fcom, bits, 1, est)
                for fb, (_, ber, nd, dr, dvv) in enumerate(res, 1):
                    pairs, nfa = match(np.array(tr), np.array(tv), dr, dvv)
                    ser = sum((tr[p] - dr[q]) ** 2 for p, q in pairs)
                    sev = sum((tv[p] - dvv[q]) ** 2 for p, q in pairs)
                    w.writerow([mode, g, i, fb, 3, nd, len(pairs), nfa, ser, sev,
                                ";".join(f"{x:.3f}" for x in tr), ";".join(f"{x:.3f}" for x in tv),
                                ";".join(f"{x:.3f}" for x in dr), ";".join(f"{x:.3f}" for x in dvv)])
                    k = (mode, g, fb)
                    A_ = agg.setdefault(k, np.zeros(5))
                    A_ += [3, len(pairs), nfa, ser, sev]
            k1, k2 = agg[(mode, g, 1)], agg[(mode, g, 2)]
            print(f"[{mode:6s} γ={g:4.1f}] {time.time()-t0:6.0f}s  Pd fb1 {k1[1]/k1[0]:.3f} fb2 {k2[1]/k2[0]:.3f}  "
                  f"RMSE_r fb2 {math.sqrt(k2[3]/max(k2[1],1)):.3f} m  RMSE_v fb2 {math.sqrt(k2[4]/max(k2[1],1)):.3f} m/s",
                  flush=True)
    fcsv.close()
    with open(os.path.join(a.out, "radar_summary.csv"), "w", newline="") as f:
        ww = csv.writer(f)
        ww.writerow(["mode", "gamma_db", "fb", "frames", "Pd", "false_per_frame", "rmse_r_m", "rmse_v_mps"])
        lines = ["| radar | γc (dB) | iter | Pd | false det./frame | RMSE range (m) | RMSE vel. (m/s) |",
                 "|---|---|---|---|---|---|---|"]
        for (mode, g, fb), v in sorted(agg.items()):
            n = v[0] / 3
            row = [mode, g, fb, int(n), v[1] / v[0], v[2] / n,
                   math.sqrt(v[3] / max(v[1], 1)), math.sqrt(v[4] / max(v[1], 1))]
            ww.writerow(row)
            lines.append(f"| {mode} | {g:.0f} | {fb} | {row[4]:.3f} | {row[5]:.2f} | {row[6]:.3f} | {row[7]:.3f} |")
    open(os.path.join(a.out, "radar_summary.md"), "w").write(
        f"Gate: dR = {DR:.3f} m, dv = {DV:.3f} m/s; estimator {a.est}; {a.n} frames per γc\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
