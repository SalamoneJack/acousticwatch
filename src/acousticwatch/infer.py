"""Shared helpers for evaluation scripts: load the trained model and score waveforms."""
from pathlib import Path

import numpy as np
import torch

from .features import LogMel
from .model import DroneCNN

ROOT = Path(__file__).resolve().parents[2]


DEFAULT_TAG = "detect_v1"


def load_model(tag=DEFAULT_TAG, dev="cuda"):
    ck = torch.load(ROOT / "checkpoints" / tag / "best.pt", map_location=dev)
    m = DroneCNN().to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, LogMel().to(dev), ck


@torch.no_grad()
def score_wave(model, mel, x, dev="cuda", bs=512, guard=None):
    """x: float tensor/array (N, WIN) on any device -> np.array of drone probabilities."""
    x = torch.as_tensor(x, dtype=torch.float32)
    out = []
    for s in range(0, len(x), bs):
        if guard is not None:
            guard.check("eval")
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(model(mel(x[s:s + bs].to(dev))).float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def get_windows(wav, idx):
    idx = np.sort(np.asarray(idx))
    return idx, torch.from_numpy(np.asarray(wav[idx], np.int16).astype(np.float32) / 32768.0)
