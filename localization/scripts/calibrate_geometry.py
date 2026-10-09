"""Calibrate the DDL array geometry from TRAINING clips only, then sanity-check it with a classical
SRP-PHAT direction finder on the validation split (no learning involved).

Outputs: localization/results/array_geometry.json, localization/results/array_geometry.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "localization"), str(ROOT / "src")]
from acousticwatch.plotstyle import INK2, SERIES, plt                        # noqa: E402
from loc.data import PROC, load_audio                                        # noqa: E402
from loc.features import PAIRS, gcc_phat                                     # noqa: E402
from loc.geometry import (SRPPHAT, angular_error_deg, elevation_deg, fit_geometry, peak_tdoa,  # noqa: E402
                          save_geometry, speed_of_sound, unit_vectors)

RES = ROOT / "localization" / "results"


def gcc_batches(audio, bs=512):
    out = []
    for s in range(0, len(audio), bs):
        out.append(gcc_phat(torch.from_numpy(audio[s:s + bs].astype(np.float32)).cuda()).cpu())
    return torch.cat(out)


def main(n_fit=12000):
    import pandas as pd
    meta = pd.read_parquet(PROC / "meta.parquet")
    rng = np.random.default_rng(0)
    tr = meta[meta.split == "train"]
    tr = tr.iloc[np.sort(rng.choice(len(tr), min(n_fit, len(tr)), replace=False))]
    G = gcc_batches(load_audio(tr))
    tdoa, height = peak_tdoa(G)
    u = unit_vectors(tr.bearing.values, elevation_deg(tr.alt.values, tr.range.values))
    c = speed_of_sound(tr.temp_k.values)
    # weight = peak strength relative to its pair's median (weak/ambiguous peaks count less); drop weakest 50 %
    w = height / height.median(0, keepdim=True).values
    w = torch.where(w > 1.0, w.clamp(max=3.0), torch.zeros_like(w))
    best = None
    for seed in range(6):                      # several random starts; keep the best fit
        P, loss, resid = fit_geometry(tdoa, w, u, c, seed=seed)
        if best is None or loss < best[1]:
            best = (P, loss, resid, seed)
    P, loss, resid, seed = best
    used = w > 0
    info = {"fitted_on": f"{len(tr)} training clips (strongest-peak 50% of pair measurements)",
            "huber_loss": loss, "best_seed": seed,
            "median_abs_residual_samples_48k": float(resid[used].median()),
            "pair_spacing_cm": {f"{i}-{j}": float((P[i] - P[j]).norm() * 100) for i, j in PAIRS},
            "aperture_cm": float(torch.cdist(P, P).max() * 100),
            "note": "Positions in metres, x = east, y = north, z = up, centred on the array; fitted (not published)."}

    # SRP-PHAT sanity check on validation
    va = meta[meta.split == "val"]
    va = va.iloc[np.sort(rng.choice(len(va), min(5000, len(va)), replace=False))]
    Gv = gcc_batches(load_audio(va))
    srp = SRPPHAT(P, c=float(speed_of_sound(va.temp_k.values).mean()))
    est = torch.cat([srp(Gv[s:s + 256])[0] for s in range(0, len(Gv), 256)])
    err = angular_error_deg(est, va.bearing.values)
    info["srp_phat_val_bearing_mae_deg"] = float(err.mean())
    info["srp_phat_val_bearing_median_deg"] = float(err.median())
    save_geometry(P, info)
    print(json.dumps(info, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax = axes[0]
    ax.scatter(P[:, 0] * 100, P[:, 1] * 100, s=60, color=SERIES[0], zorder=3)
    for k, p in enumerate(P):
        ax.annotate(f"mic {k}\nz={p[2]*100:.0f} cm", (p[0] * 100, p[1] * 100), textcoords="offset points",
                    xytext=(6, 6), fontsize=7, color=INK2)
    ax.margins(0.25); ax.set_aspect("equal"); ax.set_xlabel("east (cm)"); ax.set_ylabel("north (cm)")
    ax.set_title("Calibrated microphone positions (top view)")
    ax = axes[1]
    ax.hist(resid[used].numpy().ravel(), bins=np.linspace(0, 20, 41), color=SERIES[0])
    ax.set_xlabel("|measured - modelled| TDOA (samples at 48 kHz)"); ax.set_ylabel("pair measurements")
    ax.set_title("Geometry fit residuals (training clips)")
    fig.tight_layout(); fig.savefig(RES / "array_geometry.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True)
    main()
