"""Access to the prepared window store (data/processed/)."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .features import WIN

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


def load_store(include_dropped=False):
    """Returns (manifest DataFrame indexed by row in windows.bin, int16 memmap of shape (N, WIN)).

    Rows flagged keep=False (exact duplicates of audio in another split) are excluded unless asked for.
    """
    man = pd.read_parquet(PROC / "manifest.parquet")
    wav = np.memmap(PROC / "windows.bin", dtype=np.int16, mode="r", shape=(len(man), WIN))
    return (man if include_dropped else man[man.keep]), wav


class WindowDataset(torch.utils.data.Dataset):
    """Reads windows from windows.bin. Only the path is pickled to DataLoader workers (Windows spawns workers and
    would otherwise try to copy the whole multi-GB memmap); each worker opens its own memmap lazily."""

    def __init__(self, wav, idx, labels):
        self.path, self.shape = wav.filename, wav.shape
        self.idx, self.labels = np.asarray(idx), np.asarray(labels, np.float32)
        self._wav = None

    def __getstate__(self):
        return {**self.__dict__, "_wav": None}

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self._wav is None:
            self._wav = np.memmap(self.path, dtype=np.int16, mode="r", shape=self.shape)
        j = self.idx[i]
        return torch.from_numpy(np.asarray(self._wav[j], np.int16).astype(np.float32) / 32768.0), \
            self.labels[i], j


def batched(wav, idx, bs=512):
    """Yield (indices, float32 tensor) batches of windows in order (for evaluation)."""
    idx = np.asarray(idx)
    for s in range(0, len(idx), bs):
        b = np.sort(idx[s:s + bs])
        yield b, torch.from_numpy(np.asarray(wav[b], np.int16).astype(np.float32) / 32768.0)
