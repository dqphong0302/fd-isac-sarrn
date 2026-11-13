#!/usr/bin/env python3
"""
complexity.py — size, operation count and CPU inference time of the learned estimators (Major comment 8):
SARRN as submitted (ResMLP, ONNX) and the ChannelNet baseline (SRCNN + DnCNN).
Times: median over --reps calls, batch 1, one CPU thread. The receiver stages shared by every estimator
(FDIC, periodogram, CA-CFAR, equaliser) are timed by scripts/timing.py (results/timing.md).

Usage: complexity.py --v36-onnx models/sarrn_submitted_v36.onnx --channelnet models/channelnet_np1.pt --out results/complexity.md
"""
import argparse
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sarrn import channelnet as CN  # noqa: E402
from sarrn.receiver import ridge_np  # noqa: E402


def med_ms(fn, reps, warm=20):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v36-onnx", default="models/sarrn_submitted_v36.onnx")
    ap.add_argument("--channelnet", default="models/channelnet_np1.pt")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default="results/complexity.md")
    a = ap.parse_args()
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)

    # SARRN (ResMLP)
    W = [numpy_helper.to_array(t) for t in onnx.load(a.v36_onnx).graph.initializer]
    p_v36 = int(sum(w.size for w in W))
    mac_v36 = int(sum(w.size for w in W if w.ndim == 2))          # Gemm weights = multiply-accumulates
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s = ort.InferenceSession(a.v36_onnx, so, providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0]
    x = rng.standard_normal((1, inp.shape[1])).astype(np.float32)
    t_v36 = med_ms(lambda: s.run(None, {inp.name: x}), a.reps)

    # ridge anchor (1 pilot symbol = 1024 samples)
    xp = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
    yp = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
    t_ridge = med_ms(lambda: ridge_np(xp, yp, 0.1), a.reps)

    # ChannelNet
    st = torch.load(a.channelnet, map_location="cpu")
    sr, dn = CN.SRCNN1D(), CN.DnCNN1D()
    sr.load_state_dict(st["srcnn"]); dn.load_state_dict(st["dncnn"]); sr.eval(); dn.eval()
    p_cn = int(sum(p.numel() for m in (sr, dn) for p in m.parameters()))
    mac_cn = 2 * sum(m.in_channels * m.out_channels * m.kernel_size[0] * CN.NFFT
                     for net in (sr, dn) for m in net.modules() if isinstance(m, torch.nn.Conv1d))
    xi = torch.randn(2, 1, CN.NFFT)                                 # real and imaginary image
    with torch.no_grad():
        t_cn = med_ms(lambda: dn(sr(xi)), max(a.reps // 10, 50))
    t_cn_in = med_ms(lambda: CN.input_response(xp, yp), max(a.reps // 10, 50))

    rows = [
        "| Estimator | Trainable parameters | Real MAC per frame | Input preparation (ms) | Network inference (ms) |",
        "|---|---|---|---|---|",
        f"| Ridge (no NN), 1 pilot | 0 | – | {t_ridge:.3f} (4-tap ridge fit) | – |",
        f"| SARRN (ResMLP, as submitted) | {p_v36:,} | {mac_v36/1e6:.2f} M | {t_ridge:.3f} + spectral features "
        f"(see timing.md) | {t_v36:.3f} |",
        f"| ChannelNet (SRCNN + DnCNN) | {p_cn:,} | {mac_cn/1e6:.0f} M | {t_cn_in:.3f} (64-tap LS + FFT) | {t_cn:.3f} |",
        "",
        f"CPU, one thread, batch 1, median of {a.reps} calls (ChannelNet: {max(a.reps // 10, 50)}).",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
