#!/usr/bin/env python3
"""
train_channelnet.py — train the ChannelNet baseline (sarrn/channelnet.py) as in the original work:
SRCNN first (300 epochs), then DnCNN on the SRCNN output (200 epochs); Adam(lr=1e-3), batch 128,
MSE loss, the weights with the lowest validation loss are kept (ModelCheckpoint save_best_only).

Usage: train_channelnet.py --train data/cn_train_np1.h5 --val data/cn_val_np1.h5 --out runs/channelnet/np1
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sarrn import channelnet as CN  # noqa: E402


def load(path):
    with h5py.File(path, "r") as f:
        Hin = f["H_in"][:, 0] + 1j * f["H_in"][:, 1]
        H = f["H"][:, 0] + 1j * f["H"][:, 1]
    s = np.sqrt(np.mean(np.abs(Hin) ** 2, axis=1, keepdims=True))      # per-response RMS
    return CN.to_images(Hin / s), CN.to_images(H / s)


def fit(model, xtr, ytr, xva, yva, epochs, bs, lr, dev, tag, log):
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    lossf = nn.MSELoss()
    xtr_t, ytr_t = torch.as_tensor(xtr), torch.as_tensor(ytr)
    xva_t, yva_t = torch.as_tensor(xva, device=dev), torch.as_tensor(yva, device=dev)
    best, best_state, g = np.inf, None, torch.Generator().manual_seed(0)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr_t), generator=g)
        tot = 0.0
        for k in range(0, len(perm), bs):
            idx = perm[k:k + bs]
            xb, yb = xtr_t[idx].to(dev), ytr_t[idx].to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            va = sum(lossf(model(xva_t[k:k + 1024]), yva_t[k:k + 1024]).item() * len(xva_t[k:k + 1024])
                     for k in range(0, len(xva_t), 1024)) / len(xva_t)
        if va < best:
            best, best_state = va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log.write(f"{tag},{ep},{tot/len(perm):.6e},{va:.6e}\n"); log.flush()
        if ep == 1 or ep % 10 == 0:
            print(f"[{tag}] ep {ep}/{epochs} train {tot/len(perm):.3e} val {va:.3e} best {best:.3e} "
                  f"{time.time()-t0:.0f}s", flush=True)
    model.load_state_dict(best_state)
    return best


@torch.no_grad()
def predict(model, x, dev, bs=1024):
    model.eval()
    return np.concatenate([model(torch.as_tensor(x[k:k + bs], device=dev)).cpu().numpy()
                           for k in range(0, len(x), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs-sr", type=int, default=300)
    ap.add_argument("--epochs-dn", type=int, default=200)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)
    xtr, ytr = load(a.train); xva, yva = load(a.val)
    print(f"train {xtr.shape}  val {xva.shape}", flush=True)
    log = open(os.path.join(a.out, "history.csv"), "w"); log.write("stage,epoch,train_mse,val_mse\n")

    sr = CN.SRCNN1D().to(dev)
    best_sr = fit(sr, xtr, ytr, xva, yva, a.epochs_sr, a.bs, a.lr, dev, "srcnn", log)
    xtr2, xva2 = predict(sr, xtr, dev), predict(sr, xva, dev)
    dn = CN.DnCNN1D().to(dev)
    best_dn = fit(dn, xtr2, ytr, xva2, yva, a.epochs_dn, a.bs, a.lr, dev, "dncnn", log)
    log.close()

    torch.save({"srcnn": sr.state_dict(), "dncnn": dn.state_dict()}, os.path.join(a.out, "channelnet.pt"))
    n_par = sum(p.numel() for m in (sr, dn) for p in m.parameters())
    meta = dict(best_val_mse_srcnn=best_sr, best_val_mse_dncnn=best_dn, params=n_par,
                ls_taps=CN.LS_TAPS, epochs_sr=a.epochs_sr, epochs_dn=a.epochs_dn, bs=a.bs, lr=a.lr,
                train=a.train, val=a.val, cmdline=" ".join(sys.argv))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
