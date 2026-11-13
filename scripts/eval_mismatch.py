#!/usr/bin/env python3
"""
eval_mismatch.py — robustness of the estimators to channel-model mismatch (Reviewer Minor 4).
No retraining: every trained model (SARRN v36, ChannelNet) is used as trained on the nominal 4-tap
quasi-static channel; LMMSE keeps assuming the nominal power-delay profile.

Variants (Table III radar scene, noise 0 dB, corrected radar):
  nominal      4 taps, PDP [0 -9.7 -19.2 -22.8] dB, static over the frame
  doppler_X    per-tap Doppler nu_p = X cos(theta_p), theta_p ~ U[0, 2pi)   (X = 100, 500, 1000 Hz)
  taps6        two extra taps at delays 4, 5 with -25, -28 dB (estimator still assumes L = 4)
  pdp_flat     PDP [0 -3 -6 -9] dB
  si_fast      the 15 slow SI paths move at |v| = 3 m/s instead of 1 m/s
NMSE is computed against the time-averaged taps (Doppler) / the first 4 taps with the energy
of unmodelled taps counted as error (taps6).
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sarrn import sim as S  # noqa: E402
from sarrn import receiver as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=20)
    ap.add_argument("--gammas", default="10,20,30")
    ap.add_argument("--v36-onnx", default="models/sarrn_submitted_v36.onnx")
    ap.add_argument("--channelnet", default="", help="checkpoint channelnet.pt (optional)")
    ap.add_argument("--out", default="results/mismatch")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    rot = complex(math.cos(S.PSK_PHASE), math.sin(S.PSK_PHASE))
    variants = ["nominal", "doppler_100", "doppler_500", "doppler_1000", "taps6", "pdp_flat", "si_fast"]
    rls = G.RLS(carry=False)
    ests = {"ls_1p": (1, lambda xp, yp, **k: G.ridge_np(xp, yp, 0.0)[0]),
            "rls_1p": (1, rls),
            "lmmse_1p": (1, None),     # set per gamma (nominal PDP assumed)
            "ridge_1p": (1, lambda xp, yp, **k: G.ridge_np(xp, yp, S.RIDGE_LAMBDA)[0]),
            "sarrn_v36_1p": (1, G.SARRN(a.v36_onnx))}
    if a.channelnet:
        from sarrn.channelnet import ChannelNetEst
        ests["channelnet_1p"] = (1, ChannelNetEst(a.channelnet))
    f = open(os.path.join(a.out, "mismatch_frames.csv"), "w", newline="")
    w = csv.writer(f); w.writerow(["variant", "method", "gamma_db", "mc", "nmse", "ber"])
    agg = {}
    for v in variants:
        for g in map(float, a.gammas.split(",")):
            Rh = 10 ** (S.HCOM_DB / 10) * 10 ** (g / 10)
            ests["lmmse_1p"] = (1, lambda xp, yp, Rh=Rh, **k: G.lmmse_np(xp, yp, Rh))
            for mc in range(a.mc):
                rng = np.random.default_rng(7000 + mc)
                seed = 50000 + int(g) * 100 + mc
                gen = torch.Generator(device=dev).manual_seed(seed)
                db = np.array([0, -3, -6, -9.0]) if v == "pdp_flat" else S.HCOM_DB
                ph = 2 * np.pi * rng.random(6)
                h = 10 ** (db / 20) * np.exp(1j * ph[:4])
                if v == "taps6":
                    h = np.concatenate([h, 10 ** (np.array([-25, -28.0]) / 20) * np.exp(1j * ph[4:])])
                h = h * math.sqrt(10 ** (g / 10))
                nu = None
                h_ref = h[:4].copy()
                if v.startswith("doppler"):
                    fd = float(v.split("_")[1])
                    nu = fd * np.cos(2 * np.pi * rng.random(4))
                    n = np.arange(S.NC * S.NS) * S.TE / S.NC
                    h_ref = h * np.array([np.mean(np.exp(1j * 2 * np.pi * nu_p * n)) for nu_p in nu])
                si_vel = [0] + ([3] * 15 if v == "si_fast" else [1] * 15)
                F = (2.0 * torch.randint(0, 2, (S.NC, S.NS), generator=gen, device=dev) - 1.0).to(torch.complex128)
                Ytg = S.generate_echo(G.T_POW, G.T_VEL, G.T_RNG, F, dev)
                Ysi = S.generate_echo(G.SI_POW, si_vel, G.SI_RNG, F, dev)
                Rn = S.generate_noise(0.0, gen, dev)
                bits = torch.randint(0, 2, (S.NC, S.NS), generator=gen, device=dev)
                Fcom = (1.0 - 2.0 * bits.double()) * rot
                x_td = S.ifft_c(Fcom)
                y = S.fir_channel(S.col_flatten(x_td), torch.as_tensor(h, device=dev), nu)
                A = S.fft_c(Ytg + Ysi + S.col_unflatten(y, S.NC, S.NS)) + Rn
                for name, (Np, est) in ests.items():
                    if hasattr(est, "reset_frame"):
                        est.reset_frame()
                    hh, ber, *_ = G.gsfdic_chain(A, F, x_td, Fcom, bits, Np, est)[-1]
                    err = np.sum(np.abs(hh - h_ref) ** 2) + np.sum(np.abs(h[4:]) ** 2)
                    nm = err / np.sum(np.abs(h) ** 2)
                    w.writerow([v, name, g, mc, nm, ber])
                    agg.setdefault((v, name, g), []).append((nm, ber))
            print(v, g, " ".join(f"{m}={10*np.log10(np.mean([x[0] for x in agg[(v,m,g)]])):.1f}dB/"
                                  f"{np.mean([x[1] for x in agg[(v,m,g)]]):.1e}" for m in ests), flush=True)
    f.close()
    gs = list(map(float, a.gammas.split(",")))
    md = ["| Variant | Method | " + " | ".join(f"γc={g:.0f}: NMSE dB / BER" for g in gs) + " |",
          "|---|---|" + "---|" * len(gs)]
    for v in variants:
        for m in ests:
            md.append(f"| {v} | {m} | " + " | ".join(
                f"{10*np.log10(np.mean([x[0] for x in agg[(v,m,g)]])):.1f} / {np.mean([x[1] for x in agg[(v,m,g)]]):.1e}"
                for g in gs) + " |")
    open(os.path.join(a.out, "mismatch_summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
