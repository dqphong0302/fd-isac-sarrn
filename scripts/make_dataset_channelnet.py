#!/usr/bin/env python3
"""
make_dataset_channelnet.py — training / validation data for the ChannelNet baseline (Major comment 4).

Same frames as scripts/make_dataset.py for the same --seed and --n (identical random draws and the same
2-iteration IFD-DIC chain with the ridge estimator), but stores, per iteration, the ChannelNet input
H_in = FFT_1024(64-tap LS fit on the pilot) and the target H = FFT_1024(true 4-tap channel).

Output: /H_in (n, 2, 1024) float32 [real, imag], /H (n, 2, 1024) float32, /gamma_db (n,), /fb (n,).
"""
import argparse
import math
import os
import sys
import time

import h5py
import numpy as np
import torch
from scipy.signal import lfilter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sarrn import sim as S  # noqa: E402
from sarrn import receiver as G  # noqa: E402
from sarrn import channelnet as CN  # noqa: E402


def chain_pilots(A, F, x_td, Fcom, Np, t_pow):
    """Same chain as make_dataset.chain_samples; returns the pilot observation of each iteration."""
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
        out.append((fb, xp, yp))
        if fb == 2:
            break
        h, _, _ = G.ridge_np(xp, yp, S.RIDGE_LAMBDA)
        e_full = S.col_flatten(etd).cpu().numpy()
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
    d_in = f.create_dataset("H_in", (n, 2, CN.NFFT), "f4")
    d_h = f.create_dataset("H", (n, 2, CN.NFFT), "f4")
    d_g = f.create_dataset("gamma_db", (n,), "f8")
    d_fb = f.create_dataset("fb", (n,), "i4")
    f.attrs.update(seed=a.seed, split=a.split, np=a.np, ls_taps=CN.LS_TAPS, cmdline=" ".join(sys.argv),
                   note="same frames as make_dataset.py with the same seed and n")
    t0 = time.time()
    for i in range(a.n):
        # identical random draws, in the same order, as scripts/make_dataset.py
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
        Ht = CN.true_response(h_true)
        for fb, xp, yp in chain_pilots(A, F, x_td, Fcom, a.np, t_pow):
            j = 2 * i + fb - 1
            Hin = CN.input_response(xp, yp)
            d_in[j] = np.stack([Hin.real, Hin.imag]); d_h[j] = np.stack([Ht.real, Ht.imag])
            d_g[j] = g; d_fb[j] = fb
        if (i + 1) % 250 == 0 or i == 0:
            el = time.time() - t0
            print(f"[{a.split}] {i+1}/{a.n}  {el:.0f}s  eta {el/(i+1)*(a.n-i-1):.0f}s", flush=True)
    f.close()
    os.replace(tmp, a.out)
    print(f"done: {a.out} ({n} samples)")


if __name__ == "__main__":
    main()
