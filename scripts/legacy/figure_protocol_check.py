#!/usr/bin/env python3
"""
figure_protocol_check.py — re-run the NMSE part of generate_paper_figures.m in Python.

Protocol of the submitted Fig. 3: fixed channel phase [0 -.8 1.6 -2.6], noise 0 dB,
Pcom = SNR - h_gain_dB, fixed SI (16 paths, direct path 100 dB) and 3 fixed targets,
Np = 4 for every method; "RLS" in that script is Ridge(lambda=0.1).
Prints LS / Ridge / true RLS / LMMSE / SARRN(onnx) mean NMSE for each SNR.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from sarrn import sim as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", default="models/sarrn_submitted_v36.onnx")
ap.add_argument("--snr", default="0,6,12,18,24,30")
ap.add_argument("--mc", type=int, default=20)
ap.add_argument("--np", type=int, default=4)
a = ap.parse_args()

import onnxruntime as ort  # noqa: E402
sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
dev = torch.device("cuda")
gen = torch.Generator(device=dev).manual_seed(42)

si_power = [100] + [-10] * 15
si_velocity = [0] + [1] * 15
si_range = [1, 10, 10, 10, 10, 10, 11, 11, 12, 12, 13, 11, 11, 12, 12, 13]
t_pow, t_vel, t_rng = [-10, -10, -10], [12, -17, -15], [5, 10, 15]
mags = 10 ** (S.HCOM_DB / 20)
phase = np.array([0, -.8, 1.6, -2.6])
h_gain_db = 10 * math.log10(np.sum(mags ** 2))
F = (2.0 * torch.randint(0, 2, (S.NC, S.NS), generator=gen, device=dev) - 1.0).to(torch.complex128)

def nm(hri, h):
    hh = hri[:4] + 1j * hri[4:]
    return np.sum(np.abs(hh - h) ** 2) / np.sum(np.abs(h) ** 2)

print(f"Np={a.np}  N_mc={a.mc}")
print(" SNR |    LS     Ridge(\"RLS\")   RLS(true)   LMMSE    SARRN   (mean NMSE, dB)")
for snr in [float(v) for v in a.snr.split(",")]:
    com_db = snr - h_gain_db
    h = mags * np.exp(1j * phase) * math.sqrt(10 ** (com_db / 10))
    ht = torch.as_tensor(h, device=dev)
    acc = {k: [] for k in ["ls", "ridge", "rls", "lmmse", "dl"]}
    for _ in range(a.mc):
        Yt = S.generate_echo(t_pow, t_vel, t_rng, F, dev)
        Ysi = S.generate_echo(si_power, si_velocity, si_range, F, dev)
        Rn = S.generate_noise(0.0, gen, dev)
        bits = torch.randint(0, 2, (S.NC, S.NS), generator=gen, device=dev)
        Fc = (1.0 - 2.0 * bits.double()) * complex(math.cos(S.PSK_PHASE), math.sin(S.PSK_PHASE))
        xtd = S.ifft_c(Fc)
        y = S.fir_channel(S.col_flatten(xtd), ht)
        A = S.fft_c(Yt + Ysi + S.col_unflatten(y, S.NC, S.NS)) + Rn
        E = S.fdic(A, F)
        etd = S.ifft_c(E)
        xp = S.col_flatten(xtd[:, : a.np]); yp = S.col_flatten(etd[:, : a.np])
        h_r, Xc, yv = S.ridge_fir(xp, yp, 4, S.RIDGE_LAMBDA)
        h_ls, _, _ = S.ridge_fir(xp, yp, 4, 0.0)
        Rh = torch.as_tensor((mags ** 2) * 10 ** (com_db / 10), device=dev)
        h_mm = S.lmmse_fir(Xc, yv, Rh, 4)
        h_rls = S.rls_fir_batch(xp[None], yp[None], 4, [xp.numel()])[xp.numel()][0]
        feat, _ = S.spectral_features(E)
        x = np.concatenate([feat, S.c2ri(h_r)])[None].astype(np.float32)
        h_dl = sess.run(None, {"x": x})[0][0].astype(np.float64)
        acc["ls"].append(nm(S.c2ri(h_ls), h)); acc["ridge"].append(nm(S.c2ri(h_r), h))
        acc["rls"].append(nm(S.c2ri(h_rls), h)); acc["lmmse"].append(nm(S.c2ri(h_mm), h))
        acc["dl"].append(nm(h_dl, h))
    db = {k: 10 * np.log10(np.mean(v)) for k, v in acc.items()}
    print(f"{snr:4.0f} | {db['ls']:7.2f}   {db['ridge']:7.2f}       {db['rls']:7.2f}   {db['lmmse']:7.2f}  {db['dl']:7.2f}")
