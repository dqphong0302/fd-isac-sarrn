#!/usr/bin/env python3
"""
timing.py — per-stage cost of one GSFDIC iteration (Reviewer Major 8).
Wall-clock medians over --reps runs (GPU stages synchronised), CPU = Intel i5-12400F, GPU = RTX 3060,
plus analytic operation counts. Run on an otherwise idle machine.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import onnxruntime as ort
from scipy.signal import lfilter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sarrn import sim as S  # noqa: E402
from sarrn import receiver as G  # noqa: E402


def tm(fn, reps, gpu=False):
    ts = []
    for _ in range(reps):
        if gpu:
            torch.cuda.synchronize()
        t = time.perf_counter(); fn()
        if gpu:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return 1e3 * float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--onnx", default="models/sarrn_da_np1.onnx")
    ap.add_argument("--out", default="results/timing.md")
    a = ap.parse_args()
    torch.set_num_threads(1)
    rows = []
    for devname in ("cuda", "cpu"):
        dev = torch.device(devname); gpu = devname == "cuda"
        gen = torch.Generator(device=dev).manual_seed(1)
        F = (2.0 * torch.randint(0, 2, (S.NC, S.NS), generator=gen, device=dev) - 1.0).to(torch.complex128)
        A = S.fft_c(S.generate_echo(G.SI_POW, G.SI_VEL, G.SI_RNG, F, dev) +
                    S.generate_echo(G.T_POW, G.T_VEL, G.T_RNG, F, dev)) + S.generate_noise(0.0, gen, dev)
        E = S.fdic(A, F)
        Per, Np_, Mp = G.periodogram(E, F)
        d, v = G.cfar(Per, 3, Np_, Mp)
        e = S.col_flatten(S.ifft_c(E)).cpu().numpy()
        x = S.col_flatten(S.ifft_c(F)).cpu().numpy()
        h = np.array([1, .3, .1, .05], complex)
        st = {
            "FDIC (SI suppression)": lambda: S.fdic(A, F),
            "Periodogram (2-D FFT, 32x zero-pad, crop)": lambda: G.periodogram(E, F),
            "Target echo reconstruction (3 targets)": lambda: S.fft_c(S.generate_echo(G.T_POW, list(v), list(d), F, dev)),
            "Spectral features (202-dim)": lambda: S.spectral_features(E),
            "OFDM demod. of E (IFFT)": lambda: S.ifft_c(E),
        }
        for k, fn in st.items():
            rows.append((k, devname, tm(fn, a.reps, gpu)))
    # CPU-only (numpy/scipy) stages
    rows.append(("CA-CFAR on 410x410 map", "cpu", tm(lambda: G.cfar(Per, 3, Np_, Mp), a.reps)))
    rows.append(("Ridge/LS anchor, 1 pilot (1024 samples)", "cpu", tm(lambda: G.ridge_np(x[:1024], e[:1024], 0.1), a.reps)))
    rows.append(("DD-LS on full frame (524,288 samples)", "cpu", tm(lambda: G.ddls_np(x, e), a.reps)))
    rows.append(("Equalisation (IIR 1/h) of full frame", "cpu", tm(lambda: lfilter([1.0], h, e), a.reps)))
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0]
    z = np.zeros((1, inp.shape[1]), np.float32)
    rows.append(("SARRN-DA ResMLP inference (ONNX, batch 1)", "cpu-1thr", tm(lambda: s.run(None, {inp.name: z}), 200)))
    K = S.NC * S.NS; L = 4
    ops = [
        ("ResMLP", f"{(inp.shape[1]*512 + 3*512*512 + 512*8)/1e6:.2f} M real MAC"),
        ("DD-LS", f"Gram L^2 K + L K = {(L*L*K + L*K)/1e6:.1f} M complex MAC, 4x4 solve negligible"),
        ("Ridge 1 pilot", f"{(L*L*S.NC + L*S.NC)/1e3:.1f} k complex MAC"),
        ("Periodogram", f"Nc x {32*S.NS}-pt FFTs + {2*205} x {32*S.NC}-pt FFTs ≈ "
                        f"{(S.NC*32*S.NS*math.log2(32*S.NS) + 410*32*S.NC*math.log2(32*S.NC))*5/1e9:.2f} GFLOP"),
        ("Frame duration", f"{S.NS} x {S.TS*1e6:.1f} us = {S.NS*S.TS*1e3:.2f} ms"),
    ]
    md = ["| Stage (one GSFDIC iteration) | Device | Median time (ms) |", "|---|---|---|"]
    md += [f"| {k} | {d} | {t:.3f} |" for k, d, t in rows]
    md += ["", "| Item | Operation count |", "|---|---|"] + [f"| {k} | {v} |" for k, v in ops]
    open(a.out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
