#!/usr/bin/env python3
"""
rdm_figure.py — range–Doppler maps for the revised Figs 4–5 (Table III scenario, gamma_c = 30 dB,
noise 0 dB, one fixed frame). Panel (a): before iteration (GSFDIC iteration 1 input);
panels (b)–(d): after two iterations with LS (1 pilot), DD-LS (1 pilot) and SARRN (1 pilot).
True target positions are overlaid (+), CFAR detections (o). Uses the corrected periodogram;
--legacy-radar reproduces the submitted mapping for comparison.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sarrn import sim as S  # noqa: E402
from sarrn import receiver as G  # noqa: E402


def axes(Per, Nper, Mper):
    row, col = Per.shape
    rng = (row - np.arange(1, row + 1)) * S.C0 / (2 * S.DELTA_F * Nper)          # n_hat-1 per row
    vel = (np.arange(1, col + 1) - (col / 2 + 1) - 1) * S.C0 / (2 * S.FC * S.TS * Mper)
    return rng, vel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=1601)
    ap.add_argument("--da-onnx", default="models/sarrn_da_np1.onnx")
    ap.add_argument("--legacy-radar", action="store_true")
    ap.add_argument("--out", default="results/rdm")
    a = ap.parse_args()
    G.RADAR_LEGACY = a.legacy_radar
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    mags = 10 ** (S.HCOM_DB / 20)
    h_true = mags * np.exp(1j * G.H_PH) * math.sqrt(10 ** (a.gamma / 10))
    rot = complex(math.cos(S.PSK_PHASE), math.sin(S.PSK_PHASE))
    F = (2.0 * torch.randint(0, 2, (S.NC, S.NS), generator=gen, device=dev) - 1.0).to(torch.complex128)
    Ytg = S.generate_echo(G.T_POW, G.T_VEL, G.T_RNG, F, dev)
    Ysi = S.generate_echo(G.SI_POW, G.SI_VEL, G.SI_RNG, F, dev)
    Rn = S.generate_noise(0.0, gen, dev)
    bits = torch.randint(0, 2, (S.NC, S.NS), generator=gen, device=dev)
    Fcom = (1.0 - 2.0 * bits.double()) * rot
    x_td = S.ifft_c(Fcom)
    y = S.fir_channel(S.col_flatten(x_td), torch.as_tensor(h_true, device=dev))
    A = S.fft_c(Ytg + Ysi + S.col_unflatten(y, S.NC, S.NS)) + Rn

    ests = {"LS (1 pilot)": lambda xp, yp, **k: G.ridge_np(xp, yp, 0.0)[0],
            "DD-LS (1 pilot)": G.DDLS()}
    if os.path.exists(a.da_onnx):
        ests["SARRN (1 pilot)"] = G.SARRN_DA(a.da_onnx)
    pers, dets = {}, {}
    for name, est in ests.items():
        kp = []
        res = G.gsfdic_chain(A, F, x_td, Fcom, bits, 1, est, keep_per=kp)
        pers[name] = kp
        dets[name] = res
    first = next(iter(pers))
    panels = [("Before iteration", pers[first][0], dets[first][0])] + \
             [(f"After 2 iterations, {n}", pers[n][1], dets[n][1]) for n in ests]
    Nper, Mper = 32 * S.NC, 32 * S.NS
    rr, vv = axes(panels[0][1], Nper, Mper)
    fig, axs = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.3), dpi=300, sharey=True)
    lines = []
    for ax, (title, P, d) in zip(axs, panels):
        Pn = 10 * np.log10(P / P.max() + 1e-12)
        im = ax.imshow(Pn, extent=[vv[0], vv[-1], rr[-1], rr[0]], aspect="auto", cmap="viridis", vmin=-40, vmax=0)
        ax.plot(G.T_VEL, G.T_RNG, "r+", ms=9, mew=1.5, label="true")
        ax.plot(d[4], d[3], "wo", mfc="none", ms=7, mew=1.0, label="CFAR")
        ax.set_ylim(0, rr[0]); ax.set_title(title, fontsize=8)
        ax.set_xlabel("Velocity (m/s)", fontsize=8); ax.tick_params(labelsize=7)
        lines.append(f"{title}: det r={np.round(d[3],2).tolist()} v={np.round(d[4],2).tolist()}")
    axs[0].set_ylabel("Range (m)", fontsize=8)
    axs[0].legend(fontsize=6, loc="lower left")
    cb = fig.colorbar(im, ax=axs, shrink=0.9, pad=0.01); cb.set_label("Normalised power (dB)", fontsize=8)
    tag = "legacy" if a.legacy_radar else "fixed"
    for ext in ("png", "pdf", "tif"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
        fig.savefig(os.path.join(a.out, f"rdm_{tag}.{ext}"), bbox_inches="tight", **kw)
    open(os.path.join(a.out, f"rdm_{tag}.txt"), "w").write(
        f"true r={G.T_RNG} v={G.T_VEL}\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
