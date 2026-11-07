"""
sarrn.receiver — IFD-DIC (GSFDIC) receiver: corrected 2-D periodogram, CA-CFAR,
channel estimators (LS/ridge, LMMSE, RLS, DD-LS, SARRN, SARRN-DA), data modulation and the
iterative detection / cancellation chain (port of final/NMSE_{LS,RLS,DL}.m).

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
import math

import numpy as np
import onnxruntime as ort
import torch
from scipy import ndimage
from scipy.signal import lfilter

from . import sim as S

# ---------------- scenario (NMSE_DL.m / Table III) ----------------
SI_POW = [100] + [-10] * 15
SI_VEL = [0] + [1] * 15
SI_RNG = [1, 10, 10, 10, 10, 10, 11, 11, 12, 12, 13, 11, 11, 12, 12, 13]
T_POW, T_VEL, T_RNG = [-10, -10, -10], [-20, -10, 15], [5, 12, 20]
H_DB, H_PH = S.HCOM_DB, np.array([0, -.8, 1.6, -2.6])
G_PER = D_PER = 1 / 80
# calc_periodogram.m maps range to the negative-frequency half (peak falls outside the [:Nmax] crop) and
# flips the velocity sign. RADAR_LEGACY=True reproduces it bit-for-bit; False applies the fix
# (conjugate R/F before the 2-D transform), verified to < 0.5 oversampled bin on single targets.
RADAR_LEGACY = False


def mround(x):  # MATLAB round (half away from zero)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def periodogram(R, F, G=G_PER, D=D_PER):
    """calc_periodogram.m. Uses: Cper'->ifft(.,Mper)->' == fft along symbols / Mper."""
    Nper, Mper = 32 * S.NC, 32 * S.NS
    Nmax, Mmax = int(mround(G * Nper)), int(mround(D * Mper))
    w = torch.as_tensor(np.outer(np.hamming(S.NC), np.hamming(S.NS)), device=R.device)
    C = (R * w) / F
    if not RADAR_LEGACY:
        C = torch.conj(C)
    C = torch.fft.fft(C, n=Mper, dim=1) / Mper                 # Nc x Mper
    C = torch.fft.fftshift(C, dim=1)
    c0 = Mper // 2 - Mmax - 1                                  # MATLAB (numCols/2)-Mmax, 1-based
    C = C[:, c0: c0 + 2 * Mmax]
    C = torch.fft.fft(C, n=Nper, dim=0)[:Nmax]                 # first Nmax range bins
    C = torch.flip(C, dims=[0])
    return (torch.abs(C) ** 2 / (Nmax * (2 * Mmax + 1))).cpu().numpy(), Nper, Mper


def cfar(Per, k, Nper, Mper, offset=1.0, Tr=3, Tc=3, Gr=3, Gc=3):
    """cfar_ofdm_radar.m (CA-CFAR, 8-connected components -> centroid, top-k by amplitude)."""
    Per = Per / Per.max()
    row, col = Per.shape
    S2 = np.pad(Per, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def box(r0, r1, c0, c1):  # inclusive, arrays
        return S2[r1 + 1][:, c1 + 1] - S2[r0][:, c1 + 1] - S2[r1 + 1][:, c0] + S2[r0][:, c0]
    a, b = Tr + Gr, Tc + Gc
    ii = np.arange(a, row - a)               # 0-based interior rows
    jj = np.arange(b, col - b)
    tot = box(ii - a, ii + a, jj - b, jj + b)
    grd = box(ii - Gr, ii + Gr, jj - Gc, jj + Gc)
    ntr = (2 * a + 1) * (2 * b + 1) - (2 * Gr + 1) * (2 * Gc + 1)
    CUT = np.zeros_like(Per, dtype=bool)
    CUT[np.ix_(ii, jj)] = Per[np.ix_(ii, jj)] > (tot - grd) / ntr * offset
    lab, n = ndimage.label(CUT, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(0), np.zeros(0)
    cents = ndimage.center_of_mass(CUT, lab, range(1, n + 1))
    ys = np.array([int(mround(c[0])) for c in cents])        # 0-based row
    xs = np.array([int(mround(c[1])) for c in cents])
    # MATLAB find() returns column-major order; then stable sort by amplitude (desc)
    order = np.lexsort((ys, xs))
    ys, xs = ys[order], xs[order]
    amp = Per[ys, xs]
    srt = np.argsort(-amp, kind="stable")[:k]
    y1, x1 = ys[srt] + 1, xs[srt] + 1                        # 1-based like MATLAB
    m_hat = x1 - (col / 2 + 1)
    n_hat = row - y1 + 1
    dist = (n_hat - 1) * S.C0 / (2 * S.DELTA_F * Nper)
    vel = (m_hat - 1) * S.C0 / (2 * S.FC * S.TS * Mper)
    return dist, vel


# ---------------- estimators ----------------
class RLS:
    def __init__(self, carry=False):
        self.carry, self.P = carry, None

    def reset_frame(self):
        self.h = np.zeros(4, complex)
        if not self.carry or self.P is None:
            self.P = 1e4 * np.eye(4, dtype=complex)          # epsilon = 1e-4 -> P = 1e4 I

    def __call__(self, xp, yp, **_):
        """RLS_function.m exactly (a = conj(h), lambda = 0.999); returns h."""
        lam = 0.999
        a, P = np.conj(self.h), self.P
        for m in range(3, len(xp)):
            x = xp[m - 3: m + 1][::-1]                       # [x[m], ..., x[m-3]]
            alpha = yp[m] - np.vdot(a, x)
            Px = P @ x / lam
            g = Px / (1 + np.vdot(x, Px))
            P = (P - np.outer(g, x.conj() @ P)) / lam
            a = a + g * np.conj(alpha)
        self.P, self.h = P, np.conj(a)
        return self.h


def ridge_np(xp, yp, lam):
    Xc = np.lib.stride_tricks.sliding_window_view(xp, 4)[:, ::-1]
    yv = yp[3:]
    return np.linalg.solve(Xc.conj().T @ Xc + lam * np.eye(4), Xc.conj().T @ yv), Xc, yv


def lmmse_np(xp, yp, Rh):
    _, Xc, yv = ridge_np(xp, yp, 0.0)
    G = Xc.conj().T @ Xc
    hls = np.linalg.solve(G, Xc.conj().T @ yv)
    s2 = np.sum(np.abs(yv - Xc @ hls) ** 2) / (len(yv) - 4)
    return np.linalg.solve(G + s2 * np.diag(1 / Rh), Xc.conj().T @ yv)


class SARRN:
    def __init__(self, onnx_path):
        self.s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name

    def __call__(self, xp, yp, E=None, **_):
        h0, _, _ = ridge_np(xp, yp, S.RIDGE_LAMBDA)
        feat, _ = S.spectral_features(E)
        x = np.concatenate([feat, h0.real, h0.imag])[None].astype(np.float32)
        y = self.s.run(None, {self.inp: x})[0][0].astype(np.float64)
        return y[:4] + 1j * y[4:]


def ddls_np(x_full, y_full):
    """Decision-directed LS over the whole frame: regressors from the re-modulated
    decisions of the previous GSFDIC iteration (pilots forced to their true values)."""
    return ridge_np(x_full, y_full, 0.0)[0]


class DDLS:
    """fb1: Ridge on pilots (no decisions yet). fb>1: DD-LS on the full frame."""
    def __call__(self, xp, yp, x_prev=None, e_full=None, **_):
        if x_prev is None:
            return ridge_np(xp, yp, S.RIDGE_LAMBDA)[0]
        return ddls_np(x_prev, e_full)


class SARRN_DA:
    """fb1: Ridge on pilots (same chain as training). fb>1: NN on
    [202 spectral features, pilot Ridge (8), DD-LS anchor (8)], residual on DD-LS."""
    def __init__(self, onnx_path):
        self.s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name

    def __call__(self, xp, yp, E=None, x_prev=None, e_full=None, **_):
        h0 = ridge_np(xp, yp, S.RIDGE_LAMBDA)[0]
        if x_prev is None:
            return h0
        hd = ddls_np(x_prev, e_full)
        feat, _ = S.spectral_features(E)
        x = np.concatenate([feat, h0.real, h0.imag, hd.real, hd.imag])[None].astype(np.float32)
        y = self.s.run(None, {self.inp: x})[0][0].astype(np.float64)
        return y[:4] + 1j * y[4:]


# ---------------- modulation (data symbols; pilots stay rotated BPSK) ----------------
MOD = "bpsk"
BITS_PER_SYM = {"bpsk": 1, "qpsk": 2, "16qam": 4}


def modulate(b, rot):
    """b: (..., k) int tensor of bits -> unit-average-energy symbols, Gray mapping."""
    b = b.double()
    if MOD == "bpsk":
        return (1.0 - 2.0 * b[..., 0]) * rot
    if MOD == "qpsk":
        return ((1.0 - 2.0 * b[..., 0]) + 1j * (1.0 - 2.0 * b[..., 1])) / math.sqrt(2)
    lev = lambda b0, b1: (1.0 - 2.0 * b0) * (1.0 + 2.0 * b1)      # Gray: -3,-1,+1,+3 <- 11,10,00,01
    return (lev(b[..., 0], b[..., 1]) + 1j * lev(b[..., 2], b[..., 3])) / math.sqrt(10)


def demodulate(z, rot):
    """hard decisions -> bits (..., k) bool"""
    if MOD == "bpsk":
        return (torch.real(z * np.conj(rot)) < 0)[..., None]
    if MOD == "qpsk":
        return torch.stack([z.real < 0, z.imag < 0], -1)
    t = 2.0 / math.sqrt(10)
    return torch.stack([z.real < 0, z.real.abs() > t, z.imag < 0, z.imag.abs() > t], -1)


# ---------------- one GSFDIC chain ----------------
def gsfdic_chain(A, F, x_td, Fcom, bits, Np, est, conj_eq=False, n_fb=2, keep_per=None):
    dev = A.device
    if bits.dim() == 2:          # callers that pass one BPSK bit per symbol as (NC, NS)
        bits = bits[..., None]
    Yt = torch.zeros_like(A); Yc = torch.zeros_like(A)
    out = []
    Fcom_vec = S.col_flatten(Fcom).cpu().numpy()
    xp = S.col_flatten(x_td[:, :Np]).cpu().numpy()
    rot = complex(math.cos(S.PSK_PHASE), math.sin(S.PSK_PHASE))
    x_prev = None
    for fb in range(1, n_fb + 1):
        Yt_prev, Yc_prev = Yt, Yc
        At = A - Yc_prev - Yt_prev
        Hc = None
        if fb > 1:
            Hc = (-1 / (S.NS - 1) * ((At[:, 0] / F[:, 0]) - (At[:, -1] / F[:, -1])))[:, None].expand(-1, S.NS)
        E_si = S.fdic(At, F, Hc)
        Per, Nper, Mper = periodogram(E_si + Yt_prev, F)
        dist, vel = cfar(Per, len(T_POW), Nper, Mper)
        if keep_per is not None:
            keep_per.append(Per)
        Yt = S.fft_c(S.generate_echo(T_POW[: len(dist)], vel, dist, F, dev))
        Et = E_si + Yt_prev - Yt + Yc_prev
        etd = S.ifft_c(Et)
        yp = S.col_flatten(etd[:, :Np]).cpu().numpy()
        e_full = S.col_flatten(etd).cpu().numpy()
        h = est(xp, yp, E=Et, fb=fb, x_prev=x_prev, e_full=e_full)
        # equalise, decide, rebuild
        den = np.conj(h) if conj_eq else h
        ec = lfilter([1.0], den, e_full)
        Ec = S.fft_c(S.col_unflatten(torch.as_tensor(ec, device=dev), S.NC, S.NS))
        bhat = demodulate(Ec, rot)
        Fest = modulate(bhat, rot)
        Fv = S.col_flatten(Fest).clone()
        Fv[: Np * S.NC] = torch.as_tensor(Fcom_vec[: Np * S.NC], device=dev)
        x_est = S.col_flatten(S.ifft_c(S.col_unflatten(Fv, S.NC, S.NS))).cpu().numpy()
        x_prev = x_est
        Yc = S.fft_c(S.col_unflatten(torch.as_tensor(lfilter(h, [1.0], x_est), device=dev), S.NC, S.NS))
        ber = (bhat[:, Np:] != bits[:, Np:].bool()).double().mean().item()
        out.append((h, ber, len(dist), np.asarray(dist), np.asarray(vel)))
    return out


# name used in the manuscript
ifddic_chain = gsfdic_chain
