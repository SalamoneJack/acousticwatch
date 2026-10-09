"""Task (a): how bearing accuracy degrades as the drone signal gets weaker relative to noise.

DDL contains no drone-free recordings, so noise is added synthetically to the held-out TEST clips at a controlled
signal-to-noise ratio (SNR, relative to the clip's own 8-channel power). Lower SNR = a farther / quieter drone or
a noisier site. Three noise fields, from easiest to hardest for direction finding:
  1. spatially white   - independent Gaussian noise on each mic (electronic/self noise)
  2. diffuse ambience  - real outdoor ambience (AeroSonicDB environment, test-split windows from the detection
                         store), with an independent random segment on every channel (incoherent, like wind/ambient)
  3. directional engine- one traffic-engine recording (UrbanSound8K engine_idling, test split) arriving coherently
                         from a random direction, rendered with the calibrated array geometry. A competing point
                         source is the classic failure mode of acoustic direction finding.
Also: the error of the clean model vs the TRUE recorded range (real distance effect, no simulation).

Models compared: CNN timing+spectral, CNN timing-only, CNN spectral-only, SRP-PHAT classical baseline.
Outputs: localization/results/noise_curve.csv, noise_summary.json, error_vs_snr.png, error_vs_true_range.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "localization"), str(ROOT / "src"), str(Path(__file__).resolve().parent)]
from acousticwatch import gpu_guard                                   # noqa: E402
from acousticwatch.data import load_store                             # noqa: E402
from acousticwatch.plotstyle import INK2, SERIES, plt                 # noqa: E402
from eval_loc import load, run, run_srp                               # noqa: E402
from loc.data import PROC, load_audio                                 # noqa: E402
from loc.features import SR, Spectral                                 # noqa: E402
from loc.geometry import SRPPHAT, load_geometry, speed_of_sound, unit_vectors  # noqa: E402
from train_loc import ang_err                                         # noqa: E402

RES = ROOT / "localization" / "results"
DEV = "cuda"
SNRS = [30, 20, 15, 10, 5, 0, -5, -10, -15, -20]
T = 4800


def detection_noise(category, n, rng):
    """Single-channel 16 kHz windows from the detection store (test split) -> 48 kHz float32 (n, >=T)."""
    man, wav = load_store()
    idx = man.index[(man.split == "test") & (man.category == category)].values
    idx = np.sort(rng.choice(idx, min(n, len(idx)), replace=False))
    x = np.asarray(wav[idx], np.float32) / 32768
    return resample_poly(x, 3, 1, axis=1).astype(np.float32)             # (n, 24000) = 0.5 s at 48 kHz


class NoiseField:
    def __init__(self, kind, P, rng):
        self.kind, self.rng = kind, rng
        if kind == "diffuse ambience":
            self.pool = torch.from_numpy(detection_noise("aerosonicdb:environment", 3000, rng))
        elif kind == "directional engine":
            self.pool = torch.from_numpy(detection_noise("us8k:engine_idling", 3000, rng))
            self.P = P

    def sample(self, B):
        if self.kind == "spatially white":
            return torch.randn(B, 8, T)
        L = self.pool.shape[1]
        if self.kind == "diffuse ambience":   # independent clip + offset per channel -> incoherent field
            j = torch.from_numpy(self.rng.integers(0, len(self.pool), (B, 8)))
            o = self.rng.integers(0, L - T, (B, 8))
            return torch.stack([torch.stack([self.pool[j[b, c], o[b, c]:o[b, c] + T] for c in range(8)])
                                for b in range(B)])
        # directional: same clip on all channels with geometric delays for a random direction
        j = self.rng.integers(0, len(self.pool), B)
        o = self.rng.integers(0, L - 2 * T, B)
        src = torch.stack([self.pool[j[b], o[b]:o[b] + 2 * T] for b in range(B)])        # (B, 2T)
        u = unit_vectors(torch.from_numpy(self.rng.uniform(0, 360, B)), torch.from_numpy(self.rng.uniform(0, 30, B)))
        delays = -(u @ self.P.T) / 343.0                                                  # (B, 8) arrival offset (s)
        f = torch.fft.rfftfreq(2 * T, 1 / SR)
        S = torch.fft.rfft(src)[:, None, :] * torch.exp(-2j * np.pi * f[None, None] * delays[..., None])
        return torch.fft.irfft(S, n=2 * T)[..., T // 2:T // 2 + T]


def add_noise(x, field, snr_db):
    n = field.sample(len(x)).to(x.device)
    ps = x.pow(2).mean(dim=(1, 2), keepdim=True)
    pn = n.pow(2).mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    return x + n * torch.sqrt(ps / (pn * 10 ** (snr_db / 10)))


def main(n_clips=3000):
    gpu_guard.ensure_power_limit()
    guard = gpu_guard.ThermalGuard()
    rng = np.random.default_rng(0)
    meta = pd.read_parquet(PROC / "meta.parquet")
    te = meta[meta.split == "test"]
    te = te.iloc[np.sort(rng.choice(len(te), min(n_clips, len(te)), replace=False))].reset_index(drop=True)
    audio = load_audio(te)
    truth = te.bearing.values
    spec = Spectral().to(DEV)
    P, _ = load_geometry()
    srp = SRPPHAT(P, c=float(speed_of_sound(te.temp_k.values).mean()))
    models = {fs: load(fs)[0] for fs in ["both", "timing", "spectral"]
              if (ROOT / "checkpoints" / "localization" / fs / "best.pt").exists()}

    rows = []
    for kind in ["spatially white", "diffuse ambience", "directional engine"]:
        field = NoiseField(kind, P, np.random.default_rng(1))
        for snr in SNRS:
            tf = (lambda x, f=field, s=snr: add_noise(x, f, s))
            for fs, m in models.items():
                torch.manual_seed(0); field.rng = np.random.default_rng(1)
                b, r, _ = run(m, spec, audio, transform=tf, guard=guard)
                e = ang_err(b, truth)
                rows.append({"noise": kind, "snr_db": snr, "method": f"cnn_{fs}", "bearing_mae_deg": e.mean(),
                             "bearing_median_deg": np.median(e), "within_20deg": (e <= 20).mean(),
                             "range_mae_m": np.abs(r - te.range.values).mean()})
            torch.manual_seed(0); field.rng = np.random.default_rng(1)
            e = ang_err(run_srp(audio, srp, transform=tf, guard=guard), truth)
            rows.append({"noise": kind, "snr_db": snr, "method": "srp_phat", "bearing_mae_deg": e.mean(),
                         "bearing_median_deg": np.median(e), "within_20deg": (e <= 20).mean(), "range_mae_m": np.nan})
            print(kind, snr, "done", flush=True)
    df = pd.DataFrame(rows)
    # clean reference on exactly the same clips (no added noise)
    clean = {}
    for fs, m in models.items():
        b, _, _ = run(m, spec, audio, guard=guard)
        clean[f"cnn_{fs}"] = ang_err(b, truth)
    clean["srp_phat"] = ang_err(run_srp(audio, srp, guard=guard), truth)
    for meth, e in clean.items():
        rows.append({"noise": "none (clean)", "snr_db": np.inf, "method": meth, "bearing_mae_deg": e.mean(),
                     "bearing_median_deg": np.median(e), "within_20deg": (e <= 20).mean(), "range_mae_m": np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "noise_curve.csv", index=False, float_format="%.3f")
    te[["name"]].to_csv(RES / "noise_subset_clips.csv", index=False)

    summarize_and_plot(df, len(te))


def summarize_and_plot(df, n_clips):
    clean = df[df.noise == "none (clean)"].set_index("method")
    df = df[df.noise != "none (clean)"]
    summ = {"n_test_clips": int(n_clips), "chance_level_median_deg": 90.0,
            "clean_same_clips": {m: {"median_deg": float(r.bearing_median_deg), "mae_deg": float(r.bearing_mae_deg)}
                                 for m, r in clean.iterrows()}, "thresholds": {}}
    for (kind, meth), g in df.groupby(["noise", "method"]):
        g = g.sort_values("snr_db", ascending=False)
        out = {"snr30_median": float(g.bearing_median_deg.iloc[0])}
        for lim in [20, 45]:
            ok = g[g.bearing_median_deg <= lim]
            out[f"lowest_snr_with_median_le_{lim}deg"] = float(ok.snr_db.min()) if len(ok) else None
        summ["thresholds"][f"{kind} | {meth}"] = out
    (RES / "noise_summary.json").write_text(json.dumps(summ, indent=2))

    show = [("cnn_both", "CNN timing + spectral"), ("cnn_timing", "CNN timing only"), ("srp_phat", "SRP-PHAT (classical)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for ax, kind in zip(axes, ["spatially white", "diffuse ambience", "directional engine"]):
        for (meth, lab), c in zip(show, SERIES):
            g = df[(df.noise == kind) & (df.method == meth)].sort_values("snr_db")
            if len(g):
                ax.plot(g.snr_db, g.bearing_median_deg, color=c, marker="o", ms=3, label=lab)
                if meth in clean.index:
                    ax.scatter([36], [clean.loc[meth, "bearing_median_deg"]], color=c, marker="D", s=22, zorder=4)
        ax.axhline(90, color=INK2, lw=0.8, ls=":")
        ax.text(-19, 92, "chance (90°)", fontsize=7, color=INK2)
        ax.set_xticks([36, 30, 20, 10, 0, -10, -20]); ax.set_xticklabels(["clean", "30", "20", "10", "0", "-10", "-20"])
        ax.set_xlim(38, -22); ax.set_title(f"noise: {kind}")
        ax.set_xlabel("SNR (dB)   ← weaker / farther drone")
    axes[0].set_ylabel("Median bearing error (deg), test clips"); axes[0].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(RES / "error_vs_snr.png", dpi=150); plt.close(fig)

    # real distance effect on clean test data (all test clips, from eval predictions)
    p = pd.read_parquet(RES / "predictions_test.parquet")
    p["rb"] = pd.cut(p.range, [10, 35, 65, 95, 130, 300])
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for (meth, lab), c in zip([("both", "CNN timing + spectral"), ("timing", "CNN timing only"),
                               ("srp_phat", "SRP-PHAT (classical)")], SERIES):
        col = f"bearing_{meth}"
        if col not in p:
            continue
        g = p.groupby("rb", observed=True).apply(lambda d: (float(d.range.median()),
                                                            float(np.median(ang_err(d[col].values, d.bearing.values)))))
        ax.plot([v[0] for v in g.values], [v[1] for v in g.values], color=c, marker="o", label=lab)
    ax.set_xlabel("True drone range (m), test trials (bin medians)"); ax.set_ylabel("Median bearing error (deg)")
    ax.set_title("Bearing error vs real distance (no added noise)"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(RES / "error_vs_true_range.png", dpi=150); plt.close(fig)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--plot-only":
        summarize_and_plot(pd.read_csv(RES / "noise_curve.csv"), 3000)
    else:
        main()
