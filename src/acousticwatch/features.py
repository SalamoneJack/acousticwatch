"""Audio front-end: waveform windows -> log-mel spectrograms (computed on GPU)."""
import numpy as np
import torch
import torchaudio

SR = 16_000           # all audio resampled to 16 kHz mono
WIN = SR // 2         # 0.5-second analysis windows (two decisions per second; = native DADS drone clip length)
N_FFT = 1024
HOP = 160             # 10 ms -> 51 frames per window
N_MELS = 64
F_MIN, F_MAX = 20.0, 8000.0


def load_audio(path_or_bytes, sr=SR):
    """Load any audio file (path or raw bytes) as float32 mono at `sr`."""
    import io
    import soundfile as sf
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    try:
        x, fs = sf.read(src, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception:
        import librosa
        if hasattr(src, "seek"):
            src.seek(0)
        x, fs = librosa.load(src, sr=None, mono=True)
    if fs != sr:
        import librosa
        x = librosa.resample(x, orig_sr=fs, target_sr=sr, res_type="soxr_hq")
    return x.astype(np.float32)


def to_windows(x, max_windows=None, rng=None, tol=0.05):
    """Split a waveform into non-overlapping 0.5 s windows. Never tiles or zero-pads.

    Tiling/padding would create artefacts that differ between datasets (DADS drone clips are exactly 0.5 s, most
    negatives are 4-5 s), which a CNN can learn as a shortcut. A trailing piece is kept only if it is at most
    `tol` (5 %) short, and is completed by reflecting its last few milliseconds; anything shorter is dropped.
    If `max_windows` is given, a random subset is kept so long recordings cannot dominate the data.
    """
    n = len(x)
    wins = [x[i:i + WIN] for i in range(0, n - WIN + 1, WIN)]
    rem = n - len(wins) * WIN
    if rem >= int((1 - tol) * WIN):
        tail = x[len(wins) * WIN:]
        wins.append(np.pad(tail, (0, WIN - len(tail)), mode="reflect"))
    if not wins:
        return np.zeros((0, WIN), np.float32)
    w = np.stack(wins).astype(np.float32)
    if max_windows is not None and len(w) > max_windows:
        rng = rng or np.random.default_rng(0)
        w = w[np.sort(rng.choice(len(w), max_windows, replace=False))]
    return w


def to_int16(w):
    """Store windows compactly. Each window is peak-normalised first, so the stored amplitude carries no
    dataset-specific recording level (level is randomised again during training)."""
    peak = np.abs(w).max(axis=1, keepdims=True)
    w = w / np.maximum(peak, 1e-4) * 0.9
    return (w * 32767).astype(np.int16)


class LogMel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=F_MIN, f_max=F_MAX, power=2.0)

    @torch.no_grad()
    def forward(self, wav):                       # wav: (B, WIN) float
        with torch.autocast("cuda", enabled=False):
            # Remove each window's DC offset first: DADS drone clips carry a ~5x larger DC offset than every other
            # source, which leaks into the lowest mel bands and would be a pure dataset shortcut.
            wav = wav.float() - wav.float().mean(dim=1, keepdim=True)
            m = self.mel(wav)
            return torch.log(m + 1e-6).unsqueeze(1)   # (B, 1, N_MELS, 51)


def mel_centre_freqs():
    """Approximate centre frequency (Hz) of each mel bin, for plots."""
    import librosa
    return librosa.mel_frequencies(N_MELS + 2, fmin=F_MIN, fmax=F_MAX, htk=True)[1:-1]


# ISO 9613-1 pure-tone atmospheric absorption, 20 C / 70 % RH (dB per km) at octave-band centres
ABS_F = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], float)
ABS_DB_KM = np.array([0.1, 0.3, 1.1, 2.8, 5.0, 9.0, 22.9, 76.6])


def absorption_db_per_m(n=WIN, sr=SR):
    """Absorption (dB per metre) for each rfft bin of an n-sample window."""
    f = np.fft.rfftfreq(n, 1 / sr)
    a = np.interp(np.log(np.maximum(f, 20)), np.log(ABS_F), ABS_DB_KM) / 1000
    a[f > 8000] = ABS_DB_KM[-1] / 1000
    return torch.tensor(a, dtype=torch.float32)
