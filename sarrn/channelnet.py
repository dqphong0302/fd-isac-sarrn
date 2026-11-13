"""
sarrn.channelnet — re-implementation of ChannelNet for the comparison requested in Major comment 4.

Reference: M. Soltani, V. Pourahmadi, A. Mirzaei, H. Sheikhzadeh, "Deep learning-based channel
estimation", IEEE Commun. Lett. 23(4):652-655, 2019 (ref. [19] of the manuscript).
Authors' code: https://github.com/Mehran-Soltani/ChannelNet (Keras; no licence file, so the
architecture is re-written here from the paper and that code, not copied).

ChannelNet treats the noisy channel response at the pilots as an image and applies
  SRCNN  (64 filters 9, ReLU -> 32 filters 1, ReLU -> 1 filter 5)
  DnCNN  (64 filters 3 + ReLU, 18 x [64 filters 3 + BN + ReLU], 1 filter 3; output = input - noise)
with real and imaginary parts processed as separate single-channel images, MSE loss, Adam(1e-3),
batch 128, SRCNN trained first and DnCNN trained on the SRCNN output.

Adaptation to this system (documented in the response letter):
  * The pilot is one full OFDM symbol (all 1024 subcarriers) and the channel is quasi-static, so the
    "image" is the 1024-point frequency response; 2-D kernels k x k become 1-D kernels k along
    the subcarrier axis. No super-resolution of missing pilot positions is needed; SRCNN is kept
    as the first stage, as in the original pipeline.
  * The system has no cyclic prefix (continuous FIR channel over the frame), so the noisy input
    response is taken from an unstructured LS fit with LS_TAPS = 64 taps (no knowledge of the
    true channel length), H_in = FFT_1024([h_LS64, 0...]). This plays the role of the noisy
    pilot-position estimate of the original paper.
  * One model for gamma_c in [0, 30] dB instead of one model per SNR; each response is divided by
    its RMS value before the network and multiplied back afterwards.
  * The receiver uses a 4-tap FIR equaliser like every other estimator, so the output response is
    converted back with the inverse FFT and the first 4 taps are kept.
"""
import numpy as np
import torch
import torch.nn as nn

LS_TAPS = 64
NFFT = 1024
OUT_TAPS = 4


def ls_taps(xp, yp, L=LS_TAPS):
    """Unstructured LS FIR fit of length L from the time-domain pilot (same regression as ridge_np)."""
    Xc = np.lib.stride_tricks.sliding_window_view(xp, L)[:, ::-1]
    yv = yp[L - 1:]
    return np.linalg.solve(Xc.conj().T @ Xc, Xc.conj().T @ yv)


def input_response(xp, yp):
    h = np.zeros(NFFT, complex)
    h[:LS_TAPS] = ls_taps(xp, yp)
    return np.fft.fft(h)


def true_response(h_true):
    h = np.zeros(NFFT, complex)
    h[: len(h_true)] = h_true
    return np.fft.fft(h)


class SRCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 9, padding=4), nn.ReLU(),
            nn.Conv1d(64, 32, 1), nn.ReLU(),
            nn.Conv1d(32, 1, 5, padding=2))

    def forward(self, x):
        return self.net(x)


class DnCNN1D(nn.Module):
    def __init__(self, depth=20, ch=64):
        super().__init__()
        layers = [nn.Conv1d(1, ch, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch, eps=1e-3), nn.ReLU()]
        layers += [nn.Conv1d(ch, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)          # residual: input minus estimated noise


def to_images(H):
    """complex (B, N) -> real (2B, 1, N): real parts then imaginary parts, as in the original code."""
    return np.concatenate([H.real, H.imag], 0)[:, None, :].astype(np.float32)


def from_images(x):
    """real (2B, 1, N) -> complex (B, N)"""
    x = x[:, 0, :]
    b = x.shape[0] // 2
    return x[:b] + 1j * x[b:]


class ChannelNetEst:
    """Estimator interface of sarrn.receiver: est(xp, yp, **kw) -> 4 complex taps."""

    def __init__(self, ckpt, device=None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        st = torch.load(ckpt, map_location=self.dev)
        self.sr, self.dn = SRCNN1D().to(self.dev), DnCNN1D().to(self.dev)
        self.sr.load_state_dict(st["srcnn"]); self.dn.load_state_dict(st["dncnn"])
        self.sr.eval(); self.dn.eval()

    @torch.no_grad()
    def response(self, xp, yp):
        H = input_response(xp, yp)
        s = np.sqrt(np.mean(np.abs(H) ** 2))
        x = torch.as_tensor(to_images(H[None] / s), device=self.dev)
        out = self.dn(self.sr(x)).cpu().numpy().astype(np.float64)
        return from_images(out)[0] * s

    def __call__(self, xp, yp, **_):
        return np.fft.ifft(self.response(xp, yp))[:OUT_TAPS]
