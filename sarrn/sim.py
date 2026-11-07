"""
sarrn.sim — Python/PyTorch port of the FD-ISAC simulation used for SARRN (v36).

Faithful port of (MATLAB):
  generate_dataset_v36_optimized.m, useful_function/{fdic,generate_target_echo,generate_noise}.m

Conventions kept from MATLAB:
  DFT_matrix * X      == fft(X, axis=0, norm="ortho")
  DFT_matrix' * X     == ifft(X, axis=0, norm="ortho")
  reshape(X, [], 1)   == column-major flatten  (X.T.reshape(-1))
  comm.PSKModulator(2) has PhaseOffset = pi/4   -> symbols +-exp(j*pi/4)
  prctile             == numpy percentile method="hazen"
  std                 == unbiased (ddof = 1)

All heavy arrays are complex128 on the GPU (the direct SI path can be 100 dB
above the noise, so complex64 is not accurate enough for FDIC).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

# ----------------------------------------------------------------------------
# System parameters (identical to the MATLAB scripts)
# ----------------------------------------------------------------------------
C0 = 3e8
DELTA_F = 78.125e3
NC = 1024
NS = 512
FC = 5e9
TE = 1.0 / DELTA_F
TS = TE
DELTA_R = 1.1574
R_MAX = 2500.0
V_MAX = 75.0

HCOM_DB = np.array([0.0, -9.7, -19.2, -22.8])
HCOM_LEN = 4
RIDGE_LAMBDA = 0.1
RLS_FORGET = 0.999          # useful_function/RLS_function.m
RLS_P0 = 100.0              # P(0) = RLS_P0 * I

B_BANDS = 64
PERCENTILES = [10, 25, 50, 75, 90]
FEAT_DIM = 3 * B_BANDS + 4 + len(PERCENTILES) + 1     # 202 (without the 8-dim anchor)
X_DIM = FEAT_DIM + 2 * HCOM_LEN                        # 210 = SARRN input
PSK_PHASE = math.pi / 4


@dataclass
class SimConfig:
    """Random-scenario configuration. profile='v36' reproduces the submitted dataset."""
    profile: str = "v36"            # "v36" | "paper" (adds a 60-100 dB static direct SI path)
    # communication power / noise (dB)
    com_power_range: tuple = (20, 30)
    noise_power_range: tuple = (0, 20)
    integer_powers: bool = True     # v36 used randi() -> integer dB values
    # targets
    n_target_range: tuple = (1, 5)
    # SI
    n_static_range: tuple = (0, 5)
    n_slow_range: tuple = (0, 15)
    direct_si_db: tuple = (60, 100)  # only for profile="paper"
    # communication channel (defaults = v36: fixed PDP, random phase, static)
    tap_db: np.ndarray = field(default_factory=lambda: HCOM_DB.copy())
    tap_jitter_db: float = 0.0       # >0: random tap magnitude around tap_db (mismatch test)
    doppler_hz: float = 0.0          # >0: per-path Doppler nu_p ~ U[-doppler_hz, doppler_hz]
    fixed_phase: np.ndarray | None = None  # e.g. [0,-.8,1.6,-2.6] (figure protocol)


# ----------------------------------------------------------------------------
# Low-level blocks
# ----------------------------------------------------------------------------
def fft_c(x):   # DFT_matrix * x
    return torch.fft.fft(x, dim=0, norm="ortho")


def ifft_c(x):  # DFT_matrix' * x
    return torch.fft.ifft(x, dim=0, norm="ortho")


def col_flatten(X):          # MATLAB reshape(X, [], 1)
    return X.T.reshape(-1)


def col_unflatten(v, nr, nc):  # MATLAB reshape(v, nr, nc)
    return v.reshape(nc, nr).T


def generate_echo(powers_db, velocities, ranges, F, device):
    """generate_target_echo.m — returns the TIME-domain echo (Nc x Ns)."""
    k = torch.arange(NC, device=device, dtype=torch.float64)
    m = torch.arange(NS, device=device, dtype=torch.float64)
    Y = torch.zeros(NC, NS, dtype=torch.complex128, device=device)
    for p_db, v, r in zip(powers_db, velocities, ranges):
        alpha = math.sqrt(10 ** (p_db / 10))
        fd = 2 * FC * v / C0
        tau = 2 * r / C0
        vtl = torch.exp(-1j * 2 * math.pi * DELTA_F * tau * k)            # Nc
        vfl_conj = torch.exp(-1j * 2 * math.pi * fd * TS * m)              # conj(vfl)
        Dl = torch.exp(1j * 2 * math.pi * fd * TE / NC * k)                # Nc
        Y += alpha * Dl[:, None] * ifft_c(F * (vtl[:, None] * vfl_conj[None, :]))
    return Y


def generate_noise(power_db, gen, device):
    """generate_noise.m — complex AWGN normalised to exactly the requested power."""
    re = torch.randn(NC, NS, generator=gen, device=device, dtype=torch.float64)
    im = torch.randn(NC, NS, generator=gen, device=device, dtype=torch.float64)
    w = (re + 1j * im) / math.sqrt(2)
    w = w / torch.sqrt(torch.mean(torch.abs(w) ** 2))
    return w * math.sqrt(10 ** (power_db / 10))


def fdic(R, F, H_comp=None):
    """fdic.m (vectorised). R, F: Nc x Ns."""
    T = torch.zeros_like(R)
    T[:, :-1] = R[:, :-1] - (F[:, :-1] / F[:, 1:]) * R[:, 1:]
    if H_comp is not None:
        T[:, :-1] += H_comp[:, :-1] * F[:, :-1]
    W = F[:, :1] / F                       # W[:,0] = 1
    U = torch.cumsum(W * T, dim=1)
    E = torch.empty_like(R)
    E0 = U[:, : NS - 1].sum(dim=1) / (NS - 1)
    E[:, 0] = E0
    E[:, 1:] = (F[:, 1:] / F[:, :1]) * (E0[:, None] - U[:, :-1])
    return E


def fir_channel(x, h, nu_hz=None):
    """y[n] = sum_p h_p * exp(j 2 pi nu_p n Tsamp) * x[n-p]   (MATLAB filter(h,1,x) when nu=0)."""
    y = torch.zeros_like(x)
    n = torch.arange(x.numel(), device=x.device, dtype=torch.float64)
    tsamp = TE / NC
    for p in range(h.numel()):
        xp = torch.zeros_like(x)
        xp[p:] = x[: x.numel() - p]
        if nu_hz is not None and nu_hz[p] != 0:
            xp = xp * torch.exp(1j * 2 * math.pi * float(nu_hz[p]) * tsamp * n)
        y += h[p] * xp
    return y


def conv_matrix(x, L):
    """Rows kk: [x[kk+L-1], ..., x[kk]]  (ls_fir_from_pilot_robust)."""
    N = x.numel()
    idx = torch.arange(N - L + 1, device=x.device)[:, None] + torch.arange(L - 1, -1, -1, device=x.device)[None, :]
    return x[idx]


def ridge_fir(x, y, L, lam):
    Xc = conv_matrix(x, L)
    yv = y[L - 1:]
    A = Xc.conj().T @ Xc + lam * torch.eye(L, dtype=Xc.dtype, device=Xc.device)
    return torch.linalg.solve(A, Xc.conj().T @ yv), Xc, yv


def lmmse_fir(Xc, yv, Rh_diag, L):
    """LMMSE with known PDP prior Rh = diag(Rh_diag); noise+residual-interference
    variance estimated from the LS residual (practical, no genie noise power)."""
    G = Xc.conj().T @ Xc
    h_ls = torch.linalg.solve(G, Xc.conj().T @ yv)
    resid = yv - Xc @ h_ls
    sigma2 = (torch.sum(torch.abs(resid) ** 2) / max(yv.numel() - L, 1)).real
    A = G + sigma2 * torch.diag(1.0 / Rh_diag.to(G.dtype))
    return torch.linalg.solve(A, Xc.conj().T @ yv)


def rls_fir_batch(xs, ys, L, snapshots, lam=RLS_FORGET, p0=RLS_P0):
    """Standard exponentially-weighted RLS for y[n] = h^T [x[n],...,x[n-L+1]],
    run over the pilot samples, batched over B frames.
    xs, ys: (B, N). Returns {snap: (B, L)} estimates after `snap` samples."""
    B, N = xs.shape
    dev = xs.device
    h = torch.zeros(B, L, dtype=xs.dtype, device=dev)
    P = (p0 * torch.eye(L, dtype=xs.dtype, device=dev)).expand(B, L, L).clone()
    xpad = torch.cat([torch.zeros(B, L - 1, dtype=xs.dtype, device=dev), xs], dim=1)
    out = {}
    snaps = set(snapshots)
    inv_lam = 1.0 / lam
    for n in range(N):
        u = xpad[:, n: n + L].flip(1)                 # (B,L) = [x[n],...,x[n-L+1]]
        uc = u.conj()
        Pu = torch.einsum("bij,bj->bi", P, uc)          # P u*
        denom = lam + torch.einsum("bi,bi->b", u, Pu)   # lam + u^T P u*
        k = Pu / denom[:, None]
        e = ys[:, n] - torch.einsum("bi,bi->b", u, h)
        h = h + k * e[:, None]
        uP = torch.einsum("bi,bij->bj", u, P)           # u^T P
        P = inv_lam * (P - k[:, :, None] * uP[:, None, :])
        if (n + 1) in snaps:
            out[n + 1] = h.clone()
    return out


def spectral_features(E):
    """202-dim feature vector of generate_dataset_v36_optimized.m (without anchor)."""
    nf = torch.sqrt(torch.mean(torch.abs(E) ** 2)) + np.finfo(float).eps
    En = E / nf
    P_sc = torch.mean(torch.abs(En) ** 2, dim=1)
    logP = torch.log10(P_sc + 1e-12)
    bands = logP.reshape(B_BANDS, NC // B_BANDS)
    band_mean = bands.mean(dim=1)
    band_std = bands.std(dim=1, unbiased=True)
    band_max = bands.max(dim=1).values
    lp = logP.cpu().numpy()
    g_p = np.percentile(lp, PERCENTILES, method="hazen")
    g = np.concatenate([[lp.mean(), lp.std(ddof=1), lp.max(), lp.min()], g_p])
    feat = np.concatenate([band_mean.cpu().numpy(), band_std.cpu().numpy(), band_max.cpu().numpy(),
                           g, [math.log10(float(nf))]])
    return feat.astype(np.float64), float(nf)


def c2ri(h):
    h = h.detach().cpu().numpy() if torch.is_tensor(h) else np.asarray(h)
    return np.concatenate([h.real, h.imag]).astype(np.float64)


# ----------------------------------------------------------------------------
# One frame
# ----------------------------------------------------------------------------
def simulate_frame(cfg: SimConfig, rng: np.random.Generator, gen: torch.Generator, device,
                   np_list=(1, 2, 4), gamma_db=None):
    """Simulate one FD-ISAC frame and return features + all channel estimates.

    gamma_db: if given, force gamma_c = Pcom - Pnoise (dB) while keeping Pcom and
    Pnoise inside the training ranges (stratified test set)."""
    # --- powers --------------------------------------------------------------
    if gamma_db is None:
        if cfg.integer_powers:
            noise_db = float(rng.integers(cfg.noise_power_range[0], cfg.noise_power_range[1] + 1))
            com_db = float(rng.integers(cfg.com_power_range[0], cfg.com_power_range[1] + 1))
        else:
            noise_db = float(rng.uniform(*cfg.noise_power_range))
            com_db = float(rng.uniform(*cfg.com_power_range))
    else:
        lo = max(cfg.com_power_range[0], gamma_db + cfg.noise_power_range[0])
        hi = min(cfg.com_power_range[1], gamma_db + cfg.noise_power_range[1])
        if lo > hi:
            raise ValueError(f"gamma={gamma_db} dB not reachable with the configured power ranges")
        com_db = float(rng.uniform(lo, hi))
        noise_db = com_db - gamma_db

    # --- radar waveform --------------------------------------------------------
    F = (2.0 * torch.randint(0, 2, (NC, NS), generator=gen, device=device) - 1.0).to(torch.complex128)

    # --- targets -----------------------------------------------------------------
    n_t = int(rng.integers(cfg.n_target_range[0], cfg.n_target_range[1] + 1))
    t_pow = -25 + 20 * rng.random(n_t)
    t_rng = rng.integers(5, math.floor(R_MAX / DELTA_R) + 1, n_t) * DELTA_R
    t_vel = (2 * rng.random(n_t) - 1) * V_MAX
    Y_tgt = generate_echo(t_pow, t_vel, t_rng, F, device)

    # --- self-interference ---------------------------------------------------------
    n_st = int(rng.integers(cfg.n_static_range[0], cfg.n_static_range[1] + 1))
    n_sl = int(rng.integers(cfg.n_slow_range[0], cfg.n_slow_range[1] + 1))
    n_si = n_st + n_sl
    si_pow = list(-35 + 30 * rng.random(n_si))
    si_rng = list(rng.integers(1, math.floor(100 / DELTA_R) + 1, n_si) * DELTA_R)
    si_vel = [0.0] * n_st + list((2 * rng.random(n_sl) - 1) * 2)
    if cfg.profile == "paper":
        si_pow.append(float(rng.uniform(*cfg.direct_si_db)))
        si_rng.append(float(rng.integers(1, 3)) * DELTA_R)
        si_vel.append(0.0)
    Y_si = generate_echo(si_pow, si_vel, si_rng, F, device) if si_pow else torch.zeros_like(Y_tgt)

    # --- noise ----------------------------------------------------------------------
    R_noise = generate_noise(noise_db, gen, device)

    # --- communication -------------------------------------------------------------
    bits = torch.randint(0, 2, (NC, NS), generator=gen, device=device)
    Fcom = (1.0 - 2.0 * bits.to(torch.float64)) * complex(math.cos(PSK_PHASE), math.sin(PSK_PHASE))
    x_td = ifft_c(Fcom)
    x_vec = col_flatten(x_td)

    L = len(cfg.tap_db)
    mags = 10 ** (np.asarray(cfg.tap_db, dtype=float) / 20)
    if cfg.tap_jitter_db > 0:
        mags = 10 ** ((np.asarray(cfg.tap_db) + cfg.tap_jitter_db * rng.standard_normal(L)) / 20)
        mags[0] = 10 ** (cfg.tap_db[0] / 20)
    phase = np.asarray(cfg.fixed_phase) if cfg.fixed_phase is not None else 2 * np.pi * rng.random(L)
    h_true = mags * np.exp(1j * phase) * math.sqrt(10 ** (com_db / 10))
    nu = (2 * rng.random(L) - 1) * cfg.doppler_hz if cfg.doppler_hz > 0 else None
    h_t = torch.as_tensor(h_true, dtype=torch.complex128, device=device)
    y_vec = fir_channel(x_vec, h_t, nu)
    Y_com = fft_c(col_unflatten(y_vec, NC, NS))

    # --- receiver -------------------------------------------------------------------
    A = fft_c(Y_tgt + Y_si) + Y_com + R_noise
    E = fdic(A, F)
    e_td = ifft_c(E)
    feat, nf = spectral_features(E)

    # pilot sequences for the largest Np (shorter Np are prefixes)
    np_max = max(np_list)
    xp_all = col_flatten(x_td[:, :np_max])
    yp_all = col_flatten(e_td[:, :np_max])

    Lh = HCOM_LEN
    est = {"ls": {}, "ridge": {}, "lmmse": {}}
    # LMMSE prior: nominal PDP (first Lh taps) x true Pcom ("known second-order statistics")
    Rh_np = np.full(Lh, 1e-6)
    Rh_np[: min(L, Lh)] = (10 ** (np.asarray(cfg.tap_db[: min(L, Lh)]) / 10)) * 10 ** (com_db / 10)
    Rh = torch.as_tensor(Rh_np, device=device)
    for npil in np_list:
        xp, yp = xp_all[: npil * NC], yp_all[: npil * NC]
        h_r, Xc, yv = ridge_fir(xp, yp, Lh, RIDGE_LAMBDA)
        h_ls, _, _ = ridge_fir(xp, yp, Lh, 0.0)
        est["ridge"][npil] = c2ri(h_r)
        est["ls"][npil] = c2ri(h_ls)
        est["lmmse"][npil] = c2ri(lmmse_fir(Xc, yv, Rh, Lh))

    h_true_L = np.zeros(Lh, dtype=complex)
    h_true_L[: min(L, Lh)] = h_true[: min(L, Lh)]
    return {
        "feat": feat, "nf": nf,
        "y": c2ri(h_true_L),
        "est": est,
        "xp": xp_all, "yp": yp_all,                 # kept on GPU for batched RLS
        "gamma_db": com_db - noise_db, "com_db": com_db, "noise_db": noise_db,
        "n_target": n_t, "n_si": len(si_pow),
    }
