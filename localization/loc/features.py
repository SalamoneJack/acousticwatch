"""Multi-channel direction-of-arrival features for the 8-mic DDL array (computed on GPU, batched).

Two feature families, kept separate so their contribution can be measured:
  * TIMING   - GCC-PHAT cross-correlation for all 28 microphone pairs, band-limited to 100-8000 Hz (where drone
               rotor harmonics live; this excludes wind rumble), with 4x interpolation for sub-sample delays.
               Shape (28, 2*MAX_LAG*UPS+1). The lag of the peak is the time-difference of arrival (TDOA).
  * SPECTRAL - per-channel log-mel spectrogram (8, 64, T) plus inter-channel level differences
               (each channel minus the 8-channel mean, per mel band) - the "what does it sound like / how loud
               on which side" information, with no inter-channel phase.
"""
import itertools

import torch
import torchaudio

SR = 48_000
N_CH = 8
PAIRS = list(itertools.combinations(range(N_CH), 2))     # 28 pairs
MAX_LAG = 64           # +-1.33 ms at 48 kHz (+-46 cm path); measured DDL pair TDOAs: 98th pct <= ~70 samples, most <= 40
UPS = 4                # interpolation factor for sub-sample TDOA
F_LO, F_HI = 100.0, 8000.0
N_MELS = 64


def gcc_phat(x, max_lag=MAX_LAG, ups=UPS):
    """x: (B, C, T) float -> (B, 28, 2*max_lag*ups+1) GCC-PHAT, lag axis in samples/ups (centre = 0 delay).

    Positive lag k for pair (i, j) means channel j lags channel i by k/ups samples.
    """
    B, C, T = x.shape
    n = 2 * T
    X = torch.fft.rfft(x.float() * torch.hann_window(T, device=x.device), n=n)      # (B, C, F)
    f = torch.fft.rfftfreq(n, 1 / SR).to(x.device)
    band = ((f >= F_LO) & (f <= F_HI)).float()
    i, j = zip(*PAIRS)
    R = X[:, list(j)] * X[:, list(i)].conj()
    R = R / (R.abs() + 1e-8) * band
    cc = torch.fft.irfft(R, n=n * ups)                                              # (B, 28, n*ups)
    L = max_lag * ups
    return torch.cat([cc[..., -L:], cc[..., :L + 1]], dim=-1) * ups                # (B, 28, 2L+1)


class Spectral(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=240, n_mels=N_MELS,
                                                        f_min=50, f_max=12000, power=2.0)

    def forward(self, x):                      # x: (B, C, T)
        with torch.autocast("cuda", enabled=False):
            m = torch.log(self.mel(x.float()) + 1e-8)          # (B, C, M, frames)
        ild = m - m.mean(dim=1, keepdim=True)                  # inter-channel level differences
        return torch.cat([m, ild], dim=1)                      # (B, 2C, M, frames)


def lag_axis_seconds(max_lag=MAX_LAG, ups=UPS):
    return torch.arange(-max_lag * ups, max_lag * ups + 1) / (SR * ups)
