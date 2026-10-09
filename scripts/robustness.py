"""Task 2: how detection recall falls as the drone signal weakens (a proxy for detection range).

Simulation
  received(d) = drone * g(d)  +  background         (background held at a fixed level = the sensor's noise floor)
  * Drone windows: held-out TEST drone windows (DADS) and, separately, the EXTERNAL Svanström drone windows.
  * Backgrounds (three noise floors, each normalised to RMS 0.02, so the drone sits at a typical training level):
        pink noise            - idealised broadband floor
        outdoor ambience      - AeroSonicDB environment recordings from the TEST split (real field ambience)
        traffic engine noise  - UrbanSound8K engine_idling from the TEST split (the hardest masker)
  * Reference point: at 0 dB attenuation the drone is +20 dB above the floor (broadband RMS). We do not know the
    true recording distance/SNR of the source clips, so this anchor is an ASSUMPTION; results are reported against
    attenuation (dB) first, distance second.
  * Two attenuation models:
        A) broadband attenuation (gain only): SNR = 20 dB - attenuation
        B) distance model: spherical spreading 20*log10(d/d0) + ISO 9613-1 atmospheric absorption
           (20 C, 70 % RH) applied per frequency, with an assumed reference distance d0 = 10 m.
           High frequencies fade faster, so model B tests whether the detector relies on upper harmonics.
  * Thresholds: the val-derived "standard" (val FPR <= 1 %) and "strict" (val FPR <= 0.1 %) operating points.
  * Background-only windows are scored too: that is the false-alarm floor at each noise condition.

Outputs: results/range_curve.csv, results/range_summary.json, results/range_curve_snr.png,
         results/range_curve_distance.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from acousticwatch import gpu_guard                                   # noqa: E402
from acousticwatch.data import load_store                             # noqa: E402
from acousticwatch.features import SR, WIN, absorption_db_per_m       # noqa: E402
from acousticwatch.infer import get_windows, load_model, score_wave  # noqa: E402
from acousticwatch.plotstyle import INK2, SERIES, plt                 # noqa: E402

RESULTS = ROOT / "results"
TAG = "detect_v1"
SNR0_DB, D0_M = 20.0, 10.0
FLOOR_RMS = 0.02   # noise-floor level; puts the drone at RMS 0.2 at +20 dB = typical training-window level
ATTEN_DB = np.arange(0, 52.5, 2.5)


def pink(n, rng):
    X = np.fft.rfft(rng.standard_normal((n, WIN)))
    f = np.fft.rfftfreq(WIN, 1 / SR); f[0] = f[1]
    return np.fft.irfft(X / np.sqrt(f), n=WIN).astype(np.float32)


def unit_rms(x):
    return x / x.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)


def absorption_gain(distance_m):
    return 10 ** (-absorption_db_per_m() * max(distance_m - D0_M, 0) / 20)


def crossing(x, y, level):
    """First x where y drops below `level` (linear interpolation); None if it never does."""
    for i in range(1, len(y)):
        if y[i - 1] >= level > y[i]:
            return float(x[i - 1] + (level - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))
    return None


def main(n_drone=3000, n_bg=3000):
    gpu_guard.ensure_power_limit()
    guard = gpu_guard.ThermalGuard()
    rng = np.random.default_rng(0)
    man, wav = load_store()
    model, mel, _ = load_model(TAG)
    thr = json.loads((RESULTS / "metrics.json").read_text())["thresholds_from_val"]

    def sample(d, n):
        return d.index.values if len(d) <= n else rng.choice(d.index.values, n, replace=False)

    te = man[man.split == "test"]
    drones = {"DADS test": sample(te[te.label == 1], n_drone),
              "External (Svanström)": sample(man[(man.split == "external") & (man.label == 1)], n_drone)}
    amb = te[te.category == "aerosonicdb:environment"]
    if len(amb) < 50:   # fallback if AeroSonicDB ambience is unavailable
        amb = te[(te.label == 0) & ~te.hard_negative]
    bgs = {"pink noise": torch.from_numpy(pink(n_bg, rng)),
           "outdoor ambience": get_windows(wav, sample(amb, n_bg))[1],
           "traffic engine noise": get_windows(wav, sample(te[te.category == "us8k:engine_idling"], n_bg))[1]}
    bgs = {k: unit_rms(v) * FLOOR_RMS for k, v in bgs.items()}

    rows, floor = [], {}
    for bname, bg in bgs.items():
        p = score_wave(model, mel, bg, guard=guard)
        floor[bname] = {op: float((p >= thr[op]).mean()) for op in ["standard", "strict"]}
    for dname, didx in drones.items():
        _, S = get_windows(wav, didx)
        S = unit_rms(S) * FLOOR_RMS * 10 ** (SNR0_DB / 20)
        for bname, bg in bgs.items():
            B = bg[torch.from_numpy(rng.integers(0, len(bg), len(S)))]
            for model_name in ["broadband", "distance"]:
                for a in ATTEN_DB:
                    d = D0_M * 10 ** (a / 20)
                    s = S * 10 ** (-a / 20)
                    if model_name == "distance":
                        s = torch.fft.irfft(torch.fft.rfft(s) * absorption_gain(d), n=WIN)
                    snr = 10 * np.log10(float(s.pow(2).mean() / B.pow(2).mean()))
                    p = score_wave(model, mel, s + B, guard=guard)
                    rows.append({"drone_set": dname, "background": bname, "model": model_name, "attenuation_db": a,
                                 "distance_m_assumed": d, "effective_snr_db": snr,
                                 "recall_standard": float((p >= thr["standard"]).mean()),
                                 "recall_strict": float((p >= thr["strict"]).mean()),
                                 "mean_prob": float(p.mean())})
            print(dname, bname, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "range_curve.csv", index=False, float_format="%.4f")

    summ = {"assumptions": {"snr_at_0db_attenuation": SNR0_DB, "reference_distance_m": D0_M,
                            "absorption": "ISO 9613-1, 20C, 70% RH"},
            "false_alarm_floor_background_only": floor, "crossings": {}}
    for (dn, bn, mn), g in df.groupby(["drone_set", "background", "model"]):
        g = g.sort_values("attenuation_db")
        key = f"{dn} | {bn} | {mn}"
        c = {}
        for op in ["standard", "strict"]:
            for lvl in [0.9, 0.5]:
                a = crossing(g.attenuation_db.values, g[f"recall_{op}"].values, lvl)
                c[f"{op}_recall{int(lvl*100)}_attenuation_db"] = a
                if mn == "distance" and a is not None:
                    c[f"{op}_recall{int(lvl*100)}_distance_m_assumed"] = D0_M * 10 ** (a / 20)
                    c[f"{op}_recall{int(lvl*100)}_snr_db"] = SNR0_DB - a
            c["recall_at_0db_standard"] = float(g.recall_standard.iloc[0])
        summ["crossings"][key] = c
    (RESULTS / "range_summary.json").write_text(json.dumps(summ, indent=2))

    # ---- figures: one panel per drone set, one line per noise floor (3 series, fixed colour order)
    for model_name, xcol, xlabel, fname in [
            ("broadband", "effective_snr_db", "Drone-to-noise ratio, SNR (dB)   ← weaker / farther", "range_curve_snr.png"),
            ("distance", "distance_m_assumed", "Proxy distance (m), assuming +20 dB SNR at 10 m", "range_curve_distance.png")]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        for ax, dn in zip(axes, drones):
            for (bn, c) in zip(bgs, SERIES):
                g = df[(df.drone_set == dn) & (df.background == bn) & (df.model == model_name)].sort_values(xcol)
                ax.plot(g[xcol], g.recall_standard * 100, color=c, marker="o", ms=3, label=bn)
                ax.axhline(floor[bn]["standard"] * 100, color=c, lw=0.8, ls=":")
            ax.set_title(dn); ax.set_xlabel(xlabel); ax.set_ylim(-2, 102)
            if model_name == "broadband":
                ax.invert_xaxis()
            else:
                ax.set_xscale("log")
        axes[0].set_ylabel("Recall at standard threshold (%)")
        axes[0].legend(loc="lower left")
        fig.text(0.01, 0.005, "Dotted lines: false-alarm rate on the noise floor alone (no drone).", color=INK2, fontsize=7)
        fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(RESULTS / fname, dpi=150); plt.close(fig)
    print(json.dumps(summ, indent=1)[:3000])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="detect_v1", help="checkpoint folder under checkpoints/")
    ap.add_argument("--out", default=None, help="results folder (default results/)")
    a = ap.parse_args()
    TAG = a.tag
    if a.out:
        RESULTS = Path(a.out)
        RESULTS.mkdir(parents=True, exist_ok=True)
    main()
