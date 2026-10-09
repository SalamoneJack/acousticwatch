"""Compact spectrogram CNN for drone-vs-not detection."""
import torch
import torch.nn as nn

from .features import N_MELS


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class DroneCNN(nn.Module):
    """4 conv blocks (32-64-128-256 ch) -> mean over frequency, mean+max over time -> logit.

    Input is a log-mel patch (B, 1, 64 mel bins, 51 frames = 0.5 s). Per-mel-bin mean/std computed on the
    *training split only* are stored as buffers so inference needs no external stats file.
    Mean-over-time pooling rewards patterns that persist through the whole window (a steady rotor/engine
    harmonic stack); max-over-time keeps short strong cues.
    """

    def __init__(self, dropout=0.3):
        super().__init__()
        self.register_buffer("mu", torch.zeros(1, 1, N_MELS, 1))
        self.register_buffer("sd", torch.ones(1, 1, N_MELS, 1))
        self.b1, self.b2, self.b3, self.b4 = block(1, 32), block(32, 64), block(64, 128), block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(512, 1)

    def embed(self, x):
        x = (x - self.mu) / self.sd
        x = self.pool(self.b1(x))
        x = self.pool(self.b2(x))
        x = self.pool(self.b3(x))
        x = self.b4(x)                                # (B, 256, 8, 6)
        x = x.mean(dim=2)                             # collapse frequency -> (B, 256, T)
        return torch.cat([x.mean(-1), x.amax(-1)], 1)  # (B, 512)

    def forward(self, x):
        return self.fc(self.drop(self.embed(x))).squeeze(1)
