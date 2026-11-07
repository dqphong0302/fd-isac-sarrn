#!/usr/bin/env python3
"""
make_dataset.py — generate an independent SARRN dataset split (HDF5).

Each frame is simulated once; features are shared, and the channel estimates
(LS, Ridge = SARRN anchor, RLS, LMMSE) are stored for every pilot count in --np.

Examples
  python make_dataset.py --split train --n 16000 --seed 1001 --out data/train.h5
  python make_dataset.py --split val   --n 4000  --seed 2002 --out data/val.h5
  python make_dataset.py --split test  --gamma-grid 0:2:30 --per-gamma 500 --seed 3003 --out data/test.h5

HDF5 layout (rows = frames)
  /feat (n,202)  /y (n,8)  /gamma_db /com_db /noise_db /n_target /n_si (n,)
  /est/<method>/np<k> (n,8)   method in {ls, ridge, rls, lmmse}
  attrs: seed, split, profile, np_list, git_commit, config (json)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from sarrn import sim as S  # noqa: E402


def parse_grid(s):
    a, b, c = (float(v) for v in s.split(":"))
    return np.arange(a, c + 1e-9, b)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=0, help="number of frames (train/val)")
    ap.add_argument("--gamma-grid", default=None, help="a:step:b, stratified gamma_c (test)")
    ap.add_argument("--per-gamma", type=int, default=0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--np", default="1,2,4")
    ap.add_argument("--profile", default="v36", choices=["v36", "paper"])
    ap.add_argument("--continuous-powers", action="store_true")
    ap.add_argument("--tap-db", default=None, help="comma list, mismatch test (e.g. 0,-6,-12,-18,-24,-30)")
    ap.add_argument("--tap-jitter-db", type=float, default=0.0)
    ap.add_argument("--doppler-hz", type=float, default=0.0)
    ap.add_argument("--fixed-phase", default=None, help="comma list (figure protocol)")
    ap.add_argument("--rls-batch", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    np_list = sorted(int(v) for v in args.np.split(","))
    cfg = S.SimConfig(profile=args.profile, integer_powers=not args.continuous_powers,
                      tap_jitter_db=args.tap_jitter_db, doppler_hz=args.doppler_hz)
    if args.tap_db:
        cfg.tap_db = np.array([float(v) for v in args.tap_db.split(",")])
    if args.fixed_phase:
        cfg.fixed_phase = np.array([float(v) for v in args.fixed_phase.split(",")])

    if args.gamma_grid:
        gammas = np.repeat(parse_grid(args.gamma_grid), args.per_gamma)
    else:
        gammas = [None] * args.n
    n = len(gammas)
    if n == 0:
        sys.exit("nothing to generate (use --n or --gamma-grid/--per-gamma)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".partial"
    f = h5py.File(tmp, "w")
    d = {
        "feat": f.create_dataset("feat", (n, S.FEAT_DIM), "f8"),
        "y": f.create_dataset("y", (n, 8), "f8"),
    }
    for k in ["gamma_db", "com_db", "noise_db", "nf"]:
        d[k] = f.create_dataset(k, (n,), "f8")
    for k in ["n_target", "n_si"]:
        d[k] = f.create_dataset(k, (n,), "i4")
    for m in ["ls", "ridge", "rls", "lmmse"]:
        for k in np_list:
            d[f"{m}/np{k}"] = f.create_dataset(f"est/{m}/np{k}", (n, 8), "f8")

    cfg_json = asdict(cfg)
    cfg_json["tap_db"] = list(map(float, cfg.tap_db))
    cfg_json["fixed_phase"] = None if cfg.fixed_phase is None else list(map(float, cfg.fixed_phase))
    f.attrs.update(seed=args.seed, split=args.split, profile=args.profile,
                   np_list=np_list, git_commit=git_commit(), config=json.dumps(cfg_json),
                   cmdline=" ".join(sys.argv))

    t0 = time.time()
    buf_idx, buf_x, buf_y = [], [], []
    snaps = [k * S.NC for k in np_list]

    def flush_rls():
        if not buf_idx:
            return
        out = S.rls_fir_batch(torch.stack(buf_x), torch.stack(buf_y), S.HCOM_LEN, snaps)
        for k, s in zip(np_list, snaps):
            h = out[s].cpu().numpy()
            d[f"rls/np{k}"][buf_idx[0]: buf_idx[-1] + 1] = np.concatenate([h.real, h.imag], axis=1)
        buf_idx.clear(); buf_x.clear(); buf_y.clear()

    for i, g in enumerate(gammas):
        r = S.simulate_frame(cfg, rng, gen, device, np_list=np_list, gamma_db=g)
        d["feat"][i] = r["feat"]; d["y"][i] = r["y"]; d["nf"][i] = r["nf"]
        for k in ["gamma_db", "com_db", "noise_db", "n_target", "n_si"]:
            d[k][i] = r[k]
        for m in ["ls", "ridge", "lmmse"]:
            for k in np_list:
                d[f"{m}/np{k}"][i] = r["est"][m][k]
        buf_idx.append(i); buf_x.append(r["xp"]); buf_y.append(r["yp"])
        if len(buf_idx) == args.rls_batch:
            flush_rls()
        if (i + 1) % 500 == 0 or i == 0:
            el = time.time() - t0
            print(f"[{args.split}] {i+1}/{n}  {el:.0f}s  eta {el/(i+1)*(n-i-1):.0f}s", flush=True)
    flush_rls()
    f.close()
    os.replace(tmp, args.out)
    print(f"done: {args.out}  ({n} frames, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
