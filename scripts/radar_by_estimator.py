#!/usr/bin/env python3
"""
radar_by_estimator.py — does the channel estimator change the sensing result?

Every estimator (LS, RLS, LMMSE, DD-LS, SARRN, perfect CSI; one pilot symbol) is run on exactly the
same random scenes inside the same IFD-DIC receiver with --nfb radar passes. Radar pass k uses the
communication signal reconstructed in iteration k-1, so pass 2 is identical for LS, DD-LS and SARRN
(all three use the one-pilot estimate in iteration 1) and the estimators can only differ from pass 3 on.

Per (estimator, gamma_c, pass):
  Pd, false detections / frame, range / velocity RMSE   (Hungarian matching, gate = one native cell)
  bg_db   : mean power of the range-Doppler map outside +-2 native cells around every true target,
            in dB relative to the same quantity with perfect CSI on the same frame ("map cleanliness")
  res_db  : power of the communication signal left after cancellation in the previous iteration,
            10 log10( mean |Ycom - Ycom_hat|^2 / noise power )
Scenes: 3 targets at -10 dB, R ~ U[3,20] m, v ~ U[-25,25] m/s, random slow SI paths, random channel phases.
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from sarrn import sim as S  # noqa: E402
from sarrn import receiver as G  # noqa: E402
import eval_radar as R  # noqa: E402  (draw_targets, match)

EXCL = 2 * 32  # +-2 native cells in oversampled bins (32x zero padding)


def background(Per, Nper, Mper, tr, tv):
    row, col = Per.shape
    dr = S.C0 / (2 * S.DELTA_F * Nper)
    dv = S.C0 / (2 * S.FC * S.TS * Mper)
    m = np.ones_like(Per, dtype=bool)
    for r, v in zip(tr, tv):
        y0 = int(round(row - 1 - r / dr)); x0 = int(round(v / dv + 1 + col / 2))
        m[max(0, y0 - EXCL): y0 + EXCL + 1, max(0, x0 - EXCL): x0 + EXCL + 1] = False
    return float(Per[m].mean())


def chain(A, F, x_td, Fcom, bits, est, Ycom_true, tr, tv, n_fb):
    """gsfdic_chain with one pilot symbol, plus map background and residual communication power."""
    dev = A.device
    Np = 1
    Yt = torch.zeros_like(A); Yc = torch.zeros_like(A)
    Fcom_vec = S.col_flatten(Fcom).cpu().numpy()
    xp = S.col_flatten(x_td[:, :Np]).cpu().numpy()
    rot = complex(math.cos(S.PSK_PHASE), math.sin(S.PSK_PHASE))
    x_prev, out = None, []
    for fb in range(1, n_fb + 1):
        Yt_prev, Yc_prev = Yt, Yc
        res_db = 10 * math.log10(torch.mean(torch.abs(Ycom_true - Yc_prev) ** 2).item())
        At = A - Yc_prev - Yt_prev
        Hc = None
        if fb > 1:
            Hc = (-1 / (S.NS - 1) * ((At[:, 0] / F[:, 0]) - (At[:, -1] / F[:, -1])))[:, None].expand(-1, S.NS)
        E_si = S.fdic(At, F, Hc)
        Per, Nper, Mper = G.periodogram(E_si + Yt_prev, F)
        dist, vel = G.cfar(Per, 3, Nper, Mper)
        bg = background(Per, Nper, Mper, tr, tv)
        Yt = S.fft_c(S.generate_echo(G.T_POW[: len(dist)], vel, dist, F, dev))
        Et = E_si + Yt_prev - Yt + Yc_prev
        etd = S.ifft_c(Et)
        yp = S.col_flatten(etd[:, :Np]).cpu().numpy()
        e_full = S.col_flatten(etd).cpu().numpy()
        h = est(xp, yp, E=Et, fb=fb, x_prev=x_prev, e_full=e_full)
        ec = lfilter([1.0], h, e_full)
        Ec = S.fft_c(S.col_unflatten(torch.as_tensor(ec, device=dev), S.NC, S.NS))
        bhat = G.demodulate(Ec, rot)
        Fv = S.col_flatten(G.modulate(bhat, rot)).clone()
        Fv[: Np * S.NC] = torch.as_tensor(Fcom_vec[: Np * S.NC], device=dev)
        x_est = S.col_flatten(S.ifft_c(S.col_unflatten(Fv, S.NC, S.NS))).cpu().numpy()
        x_prev = x_est
        Yc = S.fft_c(S.col_unflatten(torch.as_tensor(lfilter(h, [1.0], x_est), device=dev), S.NC, S.NS))
        out.append((np.asarray(dist), np.asarray(vel), bg, res_db))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--gammas", default="10,20,30")
    ap.add_argument("--nfb", type=int, default=3)
    ap.add_argument("--seed", type=int, default=9100)
    ap.add_argument("--onnx", default=os.path.join(ROOT, "models", "sarrn_da_np1.onnx"))
    ap.add_argument("--out", default="results/radar_by_estimator")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mags = 10 ** (S.HCOM_DB / 20)
    rot = complex(math.cos(S.PSK_PHASE), math.sin(S.PSK_PHASE))
    sarrn = G.SARRN_DA(a.onnx)
    tag = a.gammas.replace(",", "_")
    f = open(os.path.join(a.out, f"frames_{tag}.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["est", "gamma_db", "frame", "pass", "n_match", "n_false", "se_r", "se_v", "bg_lin", "res_db", "err_r", "err_v"])
    t0 = time.time()
    for g in map(float, a.gammas.split(",")):
        rng = np.random.default_rng(a.seed + int(g))
        for i in range(a.n):
            tr, tv = R.draw_targets(rng)
            ph = 2 * np.pi * rng.random(4)
            si_rng = [1.0] + list(rng.uniform(9, 14, 15))
            si_vel = [0.0] + list(rng.uniform(0.5, 1.5, 15) * rng.choice([-1, 1], 15))
            gen = torch.Generator(device=dev).manual_seed(a.seed * 1000 + int(g) * 10000 + i)
            F = (2.0 * torch.randint(0, 2, (S.NC, S.NS), generator=gen, device=dev) - 1.0).to(torch.complex128)
            Ytg = S.generate_echo([-10.0] * 3, tv, tr, F, dev)
            Ysi = S.generate_echo(G.SI_POW, si_vel, si_rng, F, dev)
            Rn = S.generate_noise(0.0, gen, dev)
            bits = torch.randint(0, 2, (S.NC, S.NS), generator=gen, device=dev)
            Fcom = (1.0 - 2.0 * bits.double()) * rot
            x_td = S.ifft_c(Fcom)
            h_true = mags * np.exp(1j * ph) * math.sqrt(10 ** (g / 10))
            y = S.fir_channel(S.col_flatten(x_td), torch.as_tensor(h_true, device=dev))
            Ycom = S.fft_c(S.col_unflatten(y, S.NC, S.NS))
            A = S.fft_c(Ytg + Ysi) + Ycom + Rn
            Rh = mags ** 2 * 10 ** (g / 10)
            rls = G.RLS(carry=False)
            ests = {
                "perfect": lambda xp, yp, **k: h_true.copy(),
                "ls": lambda xp, yp, **k: G.ridge_np(xp, yp, 0.0)[0],
                "rls": rls,
                "lmmse": lambda xp, yp, **k: G.lmmse_np(xp, yp, Rh),
                "ddls": G.DDLS(),
                "sarrn": sarrn,
            }
            for name, est in ests.items():
                if name == "rls":
                    rls.reset_frame()
                res = chain(A, F, x_td, Fcom, bits[..., None], est, Ycom, tr, tv, a.nfb)
                for k, (dr, dvv, bg, res_db) in enumerate(res, 1):
                    pairs, nfa = R.match(np.array(tr), np.array(tv), dr, dvv)
                    ser = sum((tr[p] - dr[q]) ** 2 for p, q in pairs)
                    sev = sum((tv[p] - dvv[q]) ** 2 for p, q in pairs)
                    w.writerow([name, g, i, k, len(pairs), nfa, ser, sev, bg, res_db,
                                ";".join(f"{abs(tr[p] - dr[q]):.4f}" for p, q in pairs),
                                ";".join(f"{abs(tv[p] - dvv[q]):.4f}" for p, q in pairs)])
            f.flush()
            if i % 10 == 0:
                print(f"gamma {g:.0f} frame {i} {time.time()-t0:.0f}s", flush=True)
    f.close()


if __name__ == "__main__":
    main()
