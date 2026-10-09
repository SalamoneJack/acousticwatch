"""Bearing (+ range) estimator: two-branch CNN over timing and spectral DOA features.

feature_set: "both" | "timing" | "spectral"   (separate models are trained for the feature ablation)
Outputs: (cos θ, sin θ) of the bearing (normalised to the unit circle) and log-range.
"""
import torch
import torch.nn as nn

from .features import N_CH, N_MELS, PAIRS


def conv_bn(cin, cout, k, pool=None, dim=2):
    C, BN, P = (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d) if dim == 2 else (nn.Conv1d, nn.BatchNorm1d, nn.MaxPool1d)
    layers = [C(cin, cout, k, padding=k // 2, bias=False), BN(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(P(pool))
    return nn.Sequential(*layers)


class TimingBranch(nn.Module):
    """GCC-PHAT (B, 28 pairs, lags) -> 1-D CNN over the lag axis, pairs as input channels.

    Pairs-as-channels lets the first layer learn which combinations of pair delays are consistent with each
    direction (the job a classical steered-response beamformer does with a known geometry)."""

    def __init__(self, out=256):
        super().__init__()
        self.net = nn.Sequential(conv_bn(len(PAIRS), 128, 7, 2, dim=1), conv_bn(128, 128, 5, 2, dim=1),
                                 conv_bn(128, 256, 3, 2, dim=1), conv_bn(256, 256, 3, None, dim=1))
        self.head = nn.Linear(256 * 2, out)

    def forward(self, g):
        h = self.net(g)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))


class SpectralBranch(nn.Module):
    """(B, 16 = 8 log-mel + 8 level-difference maps, 64 mel, T) -> 2-D CNN + GRU over time (CRNN)."""

    def __init__(self, out=256):
        super().__init__()
        self.cnn = nn.Sequential(conv_bn(2 * N_CH, 64, 3, (2, 1)), conv_bn(64, 128, 3, (2, 1)),
                                 conv_bn(128, 128, 3, (2, 2)), conv_bn(128, 128, 3, (2, 1)))
        self.gru = nn.GRU(128 * (N_MELS // 16), 128, batch_first=True, bidirectional=True)
        self.head = nn.Linear(256, out)

    def forward(self, s):
        h = self.cnn(s)                                   # (B, 128, 4, T')
        h = h.flatten(1, 2).transpose(1, 2)               # (B, T', 512)
        h, _ = self.gru(h)
        return self.head(h.mean(1))


class DOANet(nn.Module):
    def __init__(self, feature_set="both", dropout=0.3):
        super().__init__()
        self.fs = feature_set
        self.timing = TimingBranch() if feature_set in ("both", "timing") else None
        self.spectral = SpectralBranch() if feature_set in ("both", "spectral") else None
        d = 256 * ((self.timing is not None) + (self.spectral is not None))
        self.mlp = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 256), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(256, 3))
        # per-feature normalisation (set from the training split only)
        self.register_buffer("s_mu", torch.zeros(1, 2 * N_CH, N_MELS, 1))
        self.register_buffer("s_sd", torch.ones(1, 2 * N_CH, N_MELS, 1))

    def forward(self, g, s, drop_timing=False, drop_spectral=False):
        z = []
        if self.timing is not None:
            t = self.timing(g)
            z.append(torch.zeros_like(t) if drop_timing else t)
        if self.spectral is not None:
            sp = self.spectral((s - self.s_mu) / self.s_sd)
            z.append(torch.zeros_like(sp) if drop_spectral else sp)
        o = self.mlp(torch.cat(z, 1))
        v = o[:, :2]
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return v, o[:, 2]                                 # unit (cos, sin), log-range
