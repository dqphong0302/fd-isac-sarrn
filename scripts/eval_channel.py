#!/usr/bin/env python3
"""
eval_channel.py — fair, single-script comparison of channel estimators inside the
GSFDIC loop used for Fig. 3 (port of final/NMSE_{LS,RLS,DL}.m).

Flow per frame (identical for every method, common random numbers):
  for fb = 1..2:
     A~ = A - Y_com_rec - Y_tgt_rec ; H_comp (fb>1) ; E_si = FDIC(A~)
     RDM = periodogram(E_si + Y_tgt_rec_prev) ; CFAR -> Y_tgt_rec
     E~ = E_si + Y_tgt_rec_prev - Y_tgt_rec + Y_com_rec_prev
     h  = estimator(pilots of E~)            <- the only thing that differs
     equalise with h, hard decisions, pilots forced, rebuild Y_com_rec

Fixes w.r.t. the MATLAB scripts: same noise / targets / MC for all methods, RLS state
(h, P) reset every frame, equaliser uses h (not conj(h) as `filter(1, h', .)`).
Two reproduction variants are kept to quantify the original issues:
  rls_carryP      : P1 carried across frames and SNR points (NMSE_RLS.m line 57)
  sarrn_v36_conj  : submitted model + conjugate equaliser (NMSE_DL.m line 187)
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sarrn import sim as S  # noqa: E402
from sarrn import receiver as G  # noqa: E402
from sarrn.receiver import (DDLS, H_DB, H_PH, RLS, SARRN, SARRN_DA, SI_POW, SI_RNG, SI_VEL,  # noqa: E402
                            T_POW, T_RNG, T_VEL, gsfdic_chain, lmmse_np, modulate, ridge_np)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", default="0:2:30")
    ap.add_argument("--mc", type=int, default=20)
    ap.add_argument("--np", default="1,2")
    ap.add_argument("--noise-db", type=float, default=0.0)
    ap.add_argument("--v36-onnx", default="models/sarrn_submitted_v36.onnx")
    ap.add_argument("--runs", default="", help="dir with np{k}/sarrn.onnx (raw-FDIC model, S1 Table)")
    ap.add_argument("--out", default="results/gsfdic")
    ap.add_argument("--etilde-runs", default="", help="dir with np{k}/sarrn.onnx trained on E_tilde")
    ap.add_argument("--methods", default="", help="comma subset of method names")
    ap.add_argument("--da-runs", default="", help="dir with np{k}/sarrn.onnx (data-aided model)")
    ap.add_argument("--mod", default="bpsk", choices=["bpsk", "qpsk", "16qam"], help="data modulation (pilots: BPSK)")
    ap.add_argument("--legacy-radar", action="store_true", help="reproduce calc_periodogram.m mapping bug")
    a = ap.parse_args()
    G.RADAR_LEGACY = a.legacy_radar
    G.MOD = a.mod
    os.makedirs(a.out, exist_ok=True)
    g0, gs, g1 = (float(v) for v in a.gamma.split(":"))
    gammas = np.arange(g0, g1 + 1e-9, gs)
    dev = torch.device("cuda")
    mags = 10 ** (H_DB / 20)

    rows = []
    fcsv = open(os.path.join(a.out, "frames.csv"), "w", newline="")
    w = csv.writer(fcsv)
    w.writerow(["np", "method", "gamma_db", "mc", "fb", "nmse", "ber", "n_det"])
    t0 = time.time()
    for Np in map(int, a.np.split(",")):
        rls_carry = RLS(carry=True)
        sarrn_v36 = SARRN(a.v36_onnx)
        p_new = os.path.join(a.runs, f"np{Np}", "sarrn.onnx") if a.runs else ""
        sarrn_new = SARRN(p_new) if p_new and os.path.exists(p_new) else None
        p_et = os.path.join(a.etilde_runs, f"np{Np}", "sarrn.onnx") if a.etilde_runs else ""
        sarrn_et = SARRN(p_et) if p_et and os.path.exists(p_et) else None
        p_da = os.path.join(a.da_runs, f"np{Np}", "sarrn.onnx") if a.da_runs else ""
        sarrn_da = SARRN_DA(p_da) if p_da and os.path.exists(p_da) else None
        for gi, g in enumerate(gammas):
            com_db = g + a.noise_db
            h_true = mags * np.exp(1j * H_PH) * math.sqrt(10 ** (com_db / 10))
            Rh = mags ** 2 * 10 ** (com_db / 10)
            rls = RLS(carry=False)
            methods = {
                "genie": (lambda xp, yp, **k: h_true.copy(), False),
                "ls": (lambda xp, yp, **k: ridge_np(xp, yp, 0.0)[0], False),
                "ridge": (lambda xp, yp, **k: ridge_np(xp, yp, S.RIDGE_LAMBDA)[0], False),
                "lmmse": (lambda xp, yp, **k: lmmse_np(xp, yp, Rh), False),
                "rls": (rls, False),
                "rls_carryP": (rls_carry, False),
                "sarrn_v36": (sarrn_v36, False),
                "sarrn_v36_conj": (sarrn_v36, True),
            }
            if sarrn_new is not None:
                methods[f"sarrn_np{Np}"] = (sarrn_new, False)
            if sarrn_et is not None:
                methods["sarrn_etilde"] = (sarrn_et, False)
            methods["ddls"] = (DDLS(), False)
            if sarrn_da is not None:
                methods["sarrn_da"] = (sarrn_da, False)
            if a.methods:
                keep = a.methods.split(",")
                methods = {k: v for k, v in methods.items() if k in keep}
            for mc in range(a.mc):
                seed = (gi + 1) * 100 + mc + 1
                gen = torch.Generator(device=dev).manual_seed(seed)
                F = (2.0 * torch.randint(0, 2, (S.NC, S.NS), generator=gen, device=dev) - 1.0).to(torch.complex128)
                Ytg = S.generate_echo(T_POW, T_VEL, T_RNG, F, dev)
                Ysi = S.generate_echo(SI_POW, SI_VEL, SI_RNG, F, dev)
                Rn = S.generate_noise(a.noise_db, gen, dev)
                rot = complex(math.cos(S.PSK_PHASE), math.sin(S.PSK_PHASE))
                if G.MOD == "bpsk":   # identical random stream to the BPSK results
                    bits = torch.randint(0, 2, (S.NC, S.NS), generator=gen, device=dev)[..., None]
                else:
                    bits = torch.randint(0, 2, (S.NC, S.NS, G.BITS_PER_SYM[G.MOD]), generator=gen, device=dev)
                    bits[:, :max(map(int, a.np.split(","))), 1:] = 0   # pilot columns carry 1 BPSK bit
                Fcom = modulate(bits, rot)
                if G.MOD != "bpsk":
                    Fcom[:, :Np] = (1.0 - 2.0 * bits[:, :Np, 0].double()) * rot
                x_td = S.ifft_c(Fcom)
                y = S.fir_channel(S.col_flatten(x_td), torch.as_tensor(h_true, device=dev))
                A = S.fft_c(Ytg + Ysi + S.col_unflatten(y, S.NC, S.NS)) + Rn
                for name, (est, conj_eq) in methods.items():
                    if isinstance(est, RLS):
                        est.reset_frame()
                    res = gsfdic_chain(A, F, x_td, Fcom, bits, Np, est, conj_eq)
                    for fb, (h, ber, nd, *_) in enumerate(res, 1):
                        nm = np.sum(np.abs(h - h_true) ** 2) / np.sum(np.abs(h_true) ** 2)
                        w.writerow([Np, name, g, mc, fb, nm, ber, nd])
                        if fb == len(res):
                            rows.append((Np, name, g, nm, ber))
            el = time.time() - t0
            line = " ".join(f"{m}={10*np.log10(np.mean([r[3] for r in rows if r[0]==Np and r[1]==m and r[2]==g])):.1f}"
                            for m in methods)
            print(f"[Np={Np} γ={g:4.1f}] {el:6.0f}s  {line}", flush=True)
    fcsv.close()

    # summary
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab = {"genie": "Perfect CSI (genie)", "ls": "LS", "ridge": "Ridge (no NN)", "lmmse": "LMMSE (known PDP)", "rls": "RLS (reset per frame)",
           "rls_carryP": "RLS (P carried, as submitted)", "sarrn_v36": "SARRN v36 (submitted model)",
           "sarrn_v36_conj": "SARRN v36 + conj equaliser (as submitted)",
           "sarrn_etilde": "SARRN trained on E_tilde",
           "ddls": "DD-LS (pilot + decided data, no NN)",
           "sarrn_da": "SARRN (proposed: DD-LS anchor + ResMLP)"}
    lines = []
    for Np in map(int, a.np.split(",")):
        names = sorted({r[1] for r in rows if r[0] == Np}, key=lambda m: list(lab).index(m) if m in lab else 99)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
        lines += [f"\n### Np = {Np}  (mean NMSE dB / mean BER over {a.mc} frames)",
                  "| method | " + " | ".join(f"{g:.0f}" for g in gammas[::5]) + " |",
                  "|---|" + "---|" * len(gammas[::5])]
        for m in names:
            nm = [np.mean([r[3] for r in rows if r[0] == Np and r[1] == m and r[2] == g]) for g in gammas]
            be = [np.mean([r[4] for r in rows if r[0] == Np and r[1] == m and r[2] == g]) for g in gammas]
            ax[0].plot(gammas, 10 * np.log10(nm), marker="o", ms=3, label=lab.get(m, f"SARRN retrained Np={Np}"))
            ax[1].semilogy(gammas, np.maximum(be, 1e-7), marker="o", ms=3, label=lab.get(m, f"SARRN retrained Np={Np}"))
            lines.append(f"| {lab.get(m, m)} | " + " | ".join(
                f"{10*np.log10(nm[i]):.1f} / {be[i]:.1e}" for i in range(0, len(gammas), 5)) + " |")
        ax[0].set_xlabel(r"$\gamma_c$ (dB)"); ax[0].set_ylabel("NMSE (dB)"); ax[0].grid(ls=":")
        ax[1].set_xlabel(r"$\gamma_c$ (dB)"); ax[1].set_ylabel("BER"); ax[1].grid(ls=":")
        ax[0].legend(fontsize=6.5); fig.suptitle(f"GSFDIC (2 iterations), same scenario for all methods, Np = {Np}")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(a.out, f"gsfdic_np{Np}.{ext}"))
        plt.close(fig)
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
