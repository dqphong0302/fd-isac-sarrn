#!/usr/bin/env python3
"""
validate_port.py — check that the Python port reproduces the MATLAB v36 dataset.

Compares a freshly generated (profile v36, Np=4) file with dataset_v36_optimized.h5:
  1. distribution of the Ridge-anchor NMSE (dB)
  2. per-feature mean / std of the 210-dim input
  3. NMSE of the ORIGINAL v36_opt.onnx model on the new samples
     (should match its validation NMSE if the port is faithful)
Usage: python validate_port.py --new data/port_check.h5 --old ../dataset_v36_optimized.h5 --onnx ../v36_opt.onnx
"""
import argparse

import h5py
import numpy as np


def nmse_db(yhat, y):
    return 10 * np.log10(np.sum((yhat - y) ** 2, 1) / np.sum(y ** 2, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--old-n", type=int, default=4000)
    a = ap.parse_args()

    with h5py.File(a.old, "r") as f:
        Xo = f["/X"][:]; Yo = f["/Y"][:]
    if Xo.shape[0] < Xo.shape[1]:
        Xo, Yo = Xo.T, Yo.T
    Xo, Yo = Xo[: a.old_n].astype(np.float64), Yo[: a.old_n].astype(np.float64)
    with h5py.File(a.new, "r") as f:
        Xn = np.concatenate([f["feat"][:], f["est/ridge/np4"][:]], 1)
        Yn = f["y"][:]

    print(f"old n={len(Xo)}  new n={len(Xn)}  dims {Xo.shape[1]} vs {Xn.shape[1]}")
    eo, en = nmse_db(Xo[:, -8:], Yo), nmse_db(Xn[:, -8:], Yn)
    q = [5, 25, 50, 75, 95]
    print("Ridge-anchor NMSE dB percentiles", q)
    print("  old:", np.round(np.percentile(eo, q), 2))
    print("  new:", np.round(np.percentile(en, q), 2))

    names = ([f"band_mean{i}" for i in range(64)] + [f"band_std{i}" for i in range(64)] +
             [f"band_max{i}" for i in range(64)] + ["g_mean", "g_std", "g_max", "g_min",
             "p10", "p25", "p50", "p75", "p90", "log_nf"] + [f"h{i}" for i in range(8)])
    mo, so = Xo.mean(0), Xo.std(0) + 1e-12
    mn = Xn.mean(0)
    z = np.abs(mn - mo) / so * np.sqrt(len(Xn))      # z-score of the new mean
    worst = np.argsort(-z)[:8]
    print("feature mean shift (in old-std units), worst 8:")
    for i in worst:
        print(f"  {names[i]:>12s}  old {mo[i]:+.4f}±{so[i]:.4f}  new {mn[i]:+.4f}  |d|/sd={abs(mn[i]-mo[i])/so[i]:.3f}")
    print(f"median |d|/sd over all features: {np.median(np.abs(mn-mo)/so):.4f}")

    import onnxruntime as ort
    s = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    names_in = [i.name for i in s.get_inputs()]
    print("onnx inputs:", [(i.name, i.shape) for i in s.get_inputs()])

    def run(X):
        feeds = {names_in[0]: X.astype(np.float32)}
        if len(names_in) > 1:
            feeds[names_in[1]] = X[:, -8:].astype(np.float32)
        return s.run(None, feeds)[0].astype(np.float64)

    for tag, X, Y in [("old", Xo, Yo), ("new", Xn, Yn)]:
        e = nmse_db(run(X), Y)
        lin = 10 * np.log10(np.mean(10 ** (e / 10)))
        print(f"v36_opt.onnx on {tag}: mean-NMSE {lin:.2f} dB | median {np.median(e):.2f} dB | anchor {10*np.log10(np.mean(10**(nmse_db(X[:,-8:],Y)/10))):.2f} dB")


if __name__ == "__main__":
    main()
