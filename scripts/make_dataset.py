#!/usr/bin/env python3
"""
make_dataset.py — training data for SARRN applied INSIDE the GSFDIC loop.

Each frame runs the 2-iteration GSFDIC chain of sarrn.receiver with the Ridge anchor as
the estimator (SARRN ~ Ridge, so the chain statistics match test time closely) and stores
one sample per iteration: features + Ridge anchor computed from E_tilde, target = true h.

Scenario family (randomised; the Table III test scenario is one fixed member of it):
  gamma_c ~ U[0,30] dB, noise 0 dB, channel PDP fixed, phases ~ U[0,2pi)
  3 targets (-10 dB), range ~ U[3,20] m, velocity ~ U[-25,25] m/s
  SI: direct path 100 dB @1 m + 15 slow paths -10 dB, range ~ U[9,14] m, |v| ~ U[0.5,1.5] m/s

Output layout is compatible with train.py: /feat /y /est/ridge/np{k} /gamma_db /fb, attrs seed.
"""
import argparse
import json
import math
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sarrn import sim as S  # noqa: E402
from sarrn import receiver as G  # noqa: E402
from scipy.signal import lfilter  # noqa: E402


def chain_samples(A, F, x_td, Fcom, Np, t_pow):
    dev = A.device
    Yt = torch.zeros_like(A); Yc = torch.zeros_like(A)
    Fcom_vec = S.col_flatten(Fcom)
    xp = S.col_flatten(x_td[:, :Np]).cpu().numpy()
    rot = complex(math.cos(S.PSK_PHASE), math.sin(S.PSK_PHASE))
    out = []
    for fb in (1, 2):
        Yt_prev, Yc_prev = Yt, Yc
        At = A - Yc_prev - Yt_prev
        Hc = None
        if fb > 1:
            Hc = (-1 / (S.NS - 1) * ((At[:, 0] / F[:, 0]) - (At[:, -1] / F[:, -1])))[:, None].expand(-1, S.NS)
        E_si = S.fdic(At, F, Hc)
        Per, Nper, Mper = G.periodogram(E_si + Yt_prev, F)
        dist, vel = G.cfar(Per, len(t_pow), Nper, Mper)
        Yt = S.fft_c(S.generate_echo(t_pow[: len(dist)], vel, dist, F, dev))
        Et = E_si + Yt_prev - Yt + Yc_prev
        etd = S.ifft_c(Et)
        yp = S.col_flatten(etd[:, :Np]).cpu().numpy()
        h, _, _ = G.ridge_np(xp, yp, S.RIDGE_LAMBDA)
        feat, _ = S.spectral_features(Et)
        e_full = S.col_flatten(etd).cpu().numpy()
        hd = G.ddls_np(x_est, e_full) if fb == 2 else np.full(4, np.nan + 0j)
        out.append((fb, feat, np.concatenate([h.real, h.imag]), np.concatenate([hd.real, hd.imag])))
        if fb == 2:
            break
        ec = lfilter([1.0], h, e_full)
        Ec = S.fft_c(S.col_unflatten(torch.as_tensor(ec, device=dev), S.NC, S.NS))
        bhat = torch.real(Ec * np.conj(rot)) < 0
        Fv = S.col_flatten((1.0 - 2.0 * bhat.double()) * rot).clone()
        Fv[: Np * S.NC] = Fcom_vec[: Np * S.NC]
        x_est = S.col_flatten(S.ifft_c(S.col_unflatten(Fv, S.NC, S.NS))).cpu().numpy()
        Yc = S.fft_c(S.col_unflatten(torch.as_tensor(lfilter(h, [1.0], x_est), device=dev), S.NC, S.NS))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="frames (2 samples per frame)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--np", type=int, default=1)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    mags = 10 ** (S.HCOM_DB / 20)
    rot = complex(math.cos(S.PSK_PHASE), math.sin(S.PSK_PHASE))

    n = 2 * a.n
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".partial"
    f = h5py.File(tmp, "w")
    d_feat = f.create_dataset("feat", (n, S.FEAT_DIM), "f8")
    d_y = f.create_dataset("y", (n, 8), "f8")
    d_a = f.create_dataset(f"est/ridge/np{a.np}", (n, 8), "f8")
    d_dd = f.create_dataset(f"est/ddls/np{a.np}", (n, 8), "f8")
    d_g = f.create_dataset("gamma_db", (n,), "f8")
    d_fb = f.create_dataset("fb", (n,), "i4")
    f.attrs.update(seed=a.seed, split=a.split, np_list=[a.np], source="E_tilde (GSFDIC, ridge chain); est/ddls only on fb==2 rows",
                   cmdline=" ".join(sys.argv),
                   config=json.dumps(dict(gamma=[0, 30], targets="3x -10dB r~U[3,20] v~U[-25,25]",
                                          si="100dB@1m + 15x -10dB r~U[9,14] |v|~U[0.5,1.5]")))
    t0 = time.time()
    for i in range(a.n):
        g = float(rng.uniform(0, 30))
        h_true = mags * np.exp(2j * np.pi * rng.random(4)) * math.sqrt(10 ** (g / 10))
        t_pow = [-10.0] * 3
        t_rng = list(rng.uniform(3, 20, 3)); t_vel = list(rng.uniform(-25, 25, 3))
        si_rng = [1.0] + list(rng.uniform(9, 14, 15))
        si_vel = [0.0] + list(rng.uniform(0.5, 1.5, 15) * rng.choice([-1, 1], 15))
        F = (2.0 * torch.randint(0, 2, (S.NC, S.NS), generator=gen, device=dev) - 1.0).to(torch.complex128)
        Ytg = S.generate_echo(t_pow, t_vel, t_rng, F, dev)
        Ysi = S.generate_echo(G.SI_POW, si_vel, si_rng, F, dev)
        Rn = S.generate_noise(0.0, gen, dev)
        bits = torch.randint(0, 2, (S.NC, S.NS), generator=gen, device=dev)
        Fcom = (1.0 - 2.0 * bits.double()) * rot
        x_td = S.ifft_c(Fcom)
        y = S.fir_channel(S.col_flatten(x_td), torch.as_tensor(h_true, device=dev))
        A = S.fft_c(Ytg + Ysi + S.col_unflatten(y, S.NC, S.NS)) + Rn
        for fb, feat, anc, dd in chain_samples(A, F, x_td, Fcom, a.np, t_pow):
            j = 2 * i + fb - 1
            d_feat[j] = feat; d_a[j] = anc; d_dd[j] = dd; d_g[j] = g; d_fb[j] = fb
            d_y[j] = np.concatenate([h_true.real, h_true.imag])
        if (i + 1) % 250 == 0 or i == 0:
            el = time.time() - t0
            print(f"[{a.split}] {i+1}/{a.n}  {el:.0f}s  eta {el/(i+1)*(a.n-i-1):.0f}s", flush=True)
    f.close()
    os.replace(tmp, a.out)
    print(f"done: {a.out} ({n} samples)")


if __name__ == "__main__":
    main()
