#!/usr/bin/env python3
"""
train.py — train SARRN (ResMLP, identical architecture/hyper-parameters to v36)
for one pilot count, on separate train / val files. The test file is never read.

  python scripts/train.py --np 1 --train data/train.h5 --val data/val.h5 --out runs/np1

Outputs in --out: sarrn.pt (best on val NMSE), sarrn.onnx, history.csv, train_meta.json
"""
import argparse
import csv
import json
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sarrn.model import ResidualMLP  # noqa: E402


def load(path, npil, anchor_key=None, extra_key=None, fb=None):
    """X = [feat, (extra), anchor]; the LAST 8 columns are the residual anchor."""
    anchor_key = anchor_key or f"est/ridge/np{npil}"
    with h5py.File(path, "r") as f:
        parts = [f["feat"][:]]
        if extra_key:
            parts.append(f[extra_key][:])
        parts.append(f[anchor_key][:])
        X = np.concatenate(parts, 1).astype(np.float32)
        Y = f["y"][:].astype(np.float32)
        if fb is not None:
            m = f["fb"][:] == fb
            X, Y = X[m], Y[m]
        meta = dict(f.attrs)
    return X, Y, meta


LOSS_MODE = "linear"


def nmse(yhat, y):
    return torch.mean(torch.sum((yhat - y) ** 2, 1) / (torch.sum(y ** 2, 1) + 1e-12))


def nmse_loss(yhat, y):
    """linear: mean NMSE (dominated by low-SNR frames); logdb: mean of 10log10 NMSE
    (every SNR weighted equally, needed when the anchor is already very accurate)."""
    if LOSS_MODE == "logdb":
        r = torch.sum((yhat - y) ** 2, 1) / (torch.sum(y ** 2, 1) + 1e-12)
        return torch.mean(10 * torch.log10(r + 1e-14))
    return nmse(yhat, y)


def mag_loss(yhat, y):
    m1 = torch.sqrt(yhat[:, :4] ** 2 + yhat[:, 4:] ** 2 + 1e-8)
    m2 = torch.sqrt(y[:, :4] ** 2 + y[:, 4:] ** 2 + 1e-8)
    return Fnn.mse_loss(m1, m2)


def run_epoch(model, loader, dev, alpha, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    tot = n_l = n_n = 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                yh = model(x)
                l_n = nmse(yh.float(), y)
                loss = nmse_loss(yh.float(), y) + alpha * mag_loss(yh.float(), y)
            if train:
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            b = x.shape[0]
            tot += b; n_l += loss.item() * b; n_n += l_n.item() * b
    return n_l / tot, n_n / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--np", type=int, required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--anchor-key", default=None, help="HDF5 key of the residual anchor (default est/ridge/np{np})")
    ap.add_argument("--extra-key", default=None, help="extra 8-dim input inserted before the anchor")
    ap.add_argument("--fb", type=int, default=None, help="keep only rows of this GSFDIC iteration")
    ap.add_argument("--rel-residual", action="store_true", help="residual scaled by ||anchor||, zero-init output")
    ap.add_argument("--loss", default="linear", choices=["linear", "logdb"])
    a = ap.parse_args()
    global LOSS_MODE
    LOSS_MODE = a.loss

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)

    Xtr, Ytr, mtr = load(a.train, a.np, a.anchor_key, a.extra_key, a.fb)
    Xva, Yva, mva = load(a.val, a.np, a.anchor_key, a.extra_key, a.fb)
    assert mtr["seed"] != mva["seed"], "train and val must come from different seeds"
    tl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)), a.batch, shuffle=True,
                    generator=torch.Generator().manual_seed(a.seed))
    vl = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)), 1024)

    model = ResidualMLP(Xtr.shape[1], dropout=a.dropout, rel=a.rel_residual).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")

    anchor_va = 10 * np.log10(np.mean(np.sum((Xva[:, -8:] - Yva) ** 2, 1) / np.sum(Yva ** 2, 1)))
    anchor_va_med = 10 * np.log10(np.median(np.sum((Xva[:, -8:] - Yva) ** 2, 1) / np.sum(Yva ** 2, 1)))
    print(f"Np={a.np}  train {len(Xtr)} (seed {mtr['seed']})  val {len(Xva)} (seed {mva['seed']})  "
          f"params {n_par:,}  val anchor NMSE {anchor_va:.2f} dB (median {anchor_va_med:.2f} dB)")

    hist, best, t0 = [], float("inf"), time.time()
    for ep in range(1, a.epochs + 1):
        lr = sch.get_last_lr()[0]
        trl, trn = run_epoch(model, tl, dev, a.alpha, opt, scaler)
        val, van = run_epoch(model, vl, dev, a.alpha)
        sch.step()
        row = dict(epoch=ep, lr=lr, train_loss=trl, train_nmse_db=10 * np.log10(trn),
                   val_loss=val, val_nmse_db=10 * np.log10(van))
        hist.append(row)
        sel = val if a.loss == "logdb" else van   # logdb: select on mean dB-NMSE
        if sel < best:
            best = sel
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "np": a.np,
                        "in_dim": Xtr.shape[1], "val_nmse_db": float(row["val_nmse_db"])},
                       os.path.join(a.out, "sarrn.pt"))
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:3d}  train {row['train_nmse_db']:7.2f} dB  val {row['val_nmse_db']:7.2f} dB  val_loss {val:.3f}")

    with open(os.path.join(a.out, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hist[0].keys()); w.writeheader(); w.writerows(hist)

    ck = torch.load(os.path.join(a.out, "sarrn.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); model.eval().cpu()
    torch.onnx.export(model, torch.zeros(1, Xtr.shape[1]), os.path.join(a.out, "sarrn.onnx"),
                      input_names=["x"], output_names=["h8"], dynamic_axes={"x": {0: "batch"}},
                      opset_version=17, dynamo=False)
    meta = dict(np=a.np, params=n_par, best_epoch=ck["epoch"], best_val_nmse_db=ck["val_nmse_db"],
                val_anchor_nmse_db=float(anchor_va), train_file=a.train, val_file=a.val,
                train_seed=int(mtr["seed"]), val_seed=int(mva["seed"]), train_time_s=time.time() - t0,
                hyper=vars(a))
    json.dump(meta, open(os.path.join(a.out, "train_meta.json"), "w"), indent=2)
    print(f"best epoch {ck['epoch']}  val NMSE {ck['val_nmse_db']:.2f} dB (anchor {anchor_va:.2f} dB)")


if __name__ == "__main__":
    main()
