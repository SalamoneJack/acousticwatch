"""Evaluate localization on the held-out TEST trials.

Reports for each trained model (both / timing-only / spectral-only) and the classical SRP-PHAT baseline:
  * bearing MAE, median, RMSE, % within 10 / 20 / 45 degrees (per 100 ms clip)
  * bearing MAE after 0.5 s smoothing (circular mean of 5 consecutive clips of the same trial; causal window)
  * range MAE / median (m), and relative error
  * breakdown by true range bin and by drone model
Feature-importance analysis (task b):
  * ablation: compare the three separately trained models
  * permutation importance on the combined model: shuffle the timing (GCC-PHAT) input or the spectral input
    across the batch, so that feature family no longer matches the clip, and measure how much the error grows

Outputs: localization/results/metrics.json, per_range_bin.csv, predictions_test.parquet, plots
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "localization"), str(ROOT / "src"), str(Path(__file__).resolve().parent)]
from acousticwatch import gpu_guard                                   # noqa: E402
from acousticwatch.plotstyle import INK2, SERIES, plt                 # noqa: E402
from loc.data import PROC, load_audio                                 # noqa: E402
from loc.features import Spectral, gcc_phat                           # noqa: E402
from loc.geometry import SRPPHAT, load_geometry, speed_of_sound       # noqa: E402
from loc.model import DOANet                                          # noqa: E402
from train_loc import ang_err, batches                                # noqa: E402

RES = ROOT / "localization" / "results"
DEV = "cuda"
BINS = [0, 35, 65, 95, 130, 300]


def load(fs):
    m = DOANet(fs).to(DEV)
    ck = torch.load(ROOT / "checkpoints" / "localization" / fs / "best.pt", map_location=DEV)
    m.load_state_dict(ck["model"]); m.eval()
    return m, int(ck["epoch"])


@torch.no_grad()
def run(model, spec, audio, idx=None, transform=None, permute=None, seed=0, guard=None):
    """permute: None | 'timing' | 'spectral' -> shuffle that input across the batch (permutation importance)."""
    gen = torch.Generator().manual_seed(seed)
    idx = np.arange(len(audio)) if idx is None else idx
    V, R = [], []
    for _, x in batches(audio, idx, 256, False):
        if guard is not None:
            guard.check("eval_loc")
        x = x.to(DEV)
        if transform is not None:
            x = transform(x)
        g, s = gcc_phat(x), spec(x)
        if permute == "timing":
            g = g[torch.randperm(len(g), generator=gen).to(DEV)]
        elif permute == "spectral":
            s = s[torch.randperm(len(s), generator=gen).to(DEV)]
        with torch.autocast("cuda", dtype=torch.float16):
            v, r = model(g, s)
        V.append(v.float().cpu()); R.append(r.float().cpu())
    V, R = torch.cat(V), torch.cat(R)
    return (torch.rad2deg(torch.atan2(V[:, 1], V[:, 0])) % 360).numpy(), torch.exp(R).numpy(), V.numpy()


@torch.no_grad()
def run_srp(audio, srp, transform=None, guard=None):
    out = []
    for _, x in batches(audio, np.arange(len(audio)), 128, False):
        if guard is not None:
            guard.check("eval_srp")
        x = x.to(DEV)
        if transform is not None:
            x = transform(x)
        out.append(srp(gcc_phat(x))[0])
    return torch.cat(out).numpy()


def smooth(meta, vec, k=5):
    """Causal circular mean of the last k clips within each trial (clips sorted by sequence id)."""
    out = np.zeros(len(meta))
    for _, g in meta.groupby("folder"):
        pos = g.index.values
        v = pd.DataFrame(vec[pos]).rolling(k, min_periods=1).mean().values
        out[pos] = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360
    return out


def bearing_stats(e):
    return {"mae_deg": float(e.mean()), "median_deg": float(np.median(e)), "rmse_deg": float(np.sqrt((e ** 2).mean())),
            "within_10deg": float((e <= 10).mean()), "within_20deg": float((e <= 20).mean()),
            "within_45deg": float((e <= 45).mean())}


def range_stats(p, t):
    a = np.abs(p - t)
    return {"mae_m": float(a.mean()), "median_m": float(np.median(a)), "mean_relative_error": float((a / t).mean())}


def main():
    gpu_guard.ensure_power_limit()
    guard = gpu_guard.ThermalGuard()
    meta = pd.read_parquet(PROC / "meta.parquet")
    te = meta[meta.split == "test"].reset_index(drop=True)
    audio = load_audio(te)
    spec = Spectral().to(DEV)
    res = {"test_clips": int(len(te)), "test_trials": sorted(te.folder.unique().tolist()), "models": {}}
    preds = te[["name", "folder", "drone", "bearing", "range", "alt"]].copy()

    for fs in ["both", "timing", "spectral"]:
        if not (ROOT / "checkpoints" / "localization" / fs / "best.pt").exists():
            continue
        model, ep = load(fs)
        b, r, v = run(model, spec, audio, guard=guard)
        e = ang_err(b, te.bearing.values)
        bs = smooth(te, v)
        res["models"][fs] = {"checkpoint_epoch": ep, "bearing": bearing_stats(e),
                             "bearing_smoothed_0.5s": bearing_stats(ang_err(bs, te.bearing.values)),
                             "range": range_stats(r, te.range.values)}
        preds[f"bearing_{fs}"], preds[f"range_{fs}"], preds[f"bearing_smoothed_{fs}"] = b, r, bs
        if fs == "both":
            base = e.mean()
            perm = {}
            for which in ["timing", "spectral"]:
                pb, pr, _ = run(model, spec, audio, permute=which, guard=guard)
                pe = ang_err(pb, te.bearing.values)
                perm[which] = {"bearing_mae_deg": float(pe.mean()), "increase_deg": float(pe.mean() - base),
                               "range_mae_m": float(np.abs(pr - te.range.values).mean())}
            res["permutation_importance_combined_model"] = perm

    # classical baseline
    P, ginfo = load_geometry()
    srp = SRPPHAT(P, c=float(speed_of_sound(te.temp_k.values).mean()))
    sb = run_srp(audio, srp, guard=guard)
    preds["bearing_srp_phat"] = sb
    res["models"]["srp_phat_classical"] = {"bearing": bearing_stats(ang_err(sb, te.bearing.values)),
                                           "note": "no learning; calibrated geometry from training clips only"}
    preds.to_parquet(RES / "predictions_test.parquet")

    # breakdowns
    preds["range_bin"] = pd.cut(preds.range, BINS)
    rows = []
    for (rb, dr), g in preds.groupby(["range_bin", "drone"], observed=True):
        row = {"range_bin_m": str(rb), "drone": dr, "clips": len(g)}
        for fs in ["both", "timing", "spectral", "srp_phat"]:
            col = f"bearing_{fs}"
            if col in g:
                row[f"bearing_mae_{fs}"] = float(ang_err(g[col].values, g.bearing.values).mean())
        if "range_both" in g:
            row["range_mae_both"] = float(np.abs(g.range_both - g.range).mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(RES / "per_range_bin.csv", index=False, float_format="%.2f")
    # per-trial breakdown (where does the heavy tail of large errors come from?)
    trows = []
    for f, g in preds.groupby("folder"):
        row = {"trial": f, "drone": g.drone.iloc[0], "clips": len(g), "median_range_m": float(g.range.median()),
               "distinct_bearings": int(g.bearing.nunique())}
        for fs in ["both", "timing", "spectral", "srp_phat"]:
            col = f"bearing_{fs}"
            if col in g:
                e = ang_err(g[col].values, g.bearing.values)
                row[f"mae_{fs}"], row[f"median_{fs}"], row[f"over90deg_{fs}"] = e.mean(), np.median(e), (e > 90).mean()
        trows.append(row)
    pd.DataFrame(trows).to_csv(RES / "per_trial.csv", index=False, float_format="%.2f")
    res["per_trial"] = trows
    (RES / "metrics.json").write_text(json.dumps(res, indent=2))

    # ---- plots
    if "bearing_both" in preds:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax = axes[0]
        ax.scatter(preds.bearing, preds.bearing_both, s=2, alpha=0.25, color=SERIES[0], rasterized=True)
        ax.plot([0, 360], [0, 360], color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("True bearing (deg)"); ax.set_ylabel("Predicted bearing (deg)")
        ax.set_title("Bearing: predicted vs true (test trials)")
        ax.set_xlim(0, 360); ax.set_ylim(0, 360)
        ax = axes[1]
        for (fs, lab), c in zip([("both", "CNN, timing + spectral"), ("timing", "CNN, timing only"),
                                 ("srp_phat", "SRP-PHAT (classical)")], SERIES):
            col = f"bearing_{fs}"
            if col in preds:
                e = np.sort(ang_err(preds[col].values, preds.bearing.values))
                ax.plot(e, np.arange(1, len(e) + 1) / len(e) * 100, color=c, label=lab)
        ax.set_xlim(0, 90); ax.set_xlabel("Absolute bearing error (deg)"); ax.set_ylabel("% of test clips")
        ax.set_title("Cumulative bearing error"); ax.legend(loc="lower right")
        fig.tight_layout(); fig.savefig(RES / "bearing_test.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(preds.range + np.random.default_rng(0).normal(0, 1.5, len(preds)), preds.range_both, s=2, alpha=0.2,
                   color=SERIES[0], rasterized=True)
        ax.plot([0, 260], [0, 260], color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("True range (m)"); ax.set_ylabel("Predicted range (m)"); ax.set_title("Range: predicted vs true")
        fig.tight_layout(); fig.savefig(RES / "range_test.png", dpi=150); plt.close(fig)

    fam = [(k, res["models"][k]["bearing"]["mae_deg"]) for k in ["both", "timing", "spectral", "srp_phat_classical"]
           if k in res["models"]]
    if "permutation_importance_combined_model" in res:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        ax = axes[0]
        labels = {"both": "timing + spectral", "timing": "timing only (GCC-PHAT)", "spectral": "spectral only",
                  "srp_phat_classical": "SRP-PHAT (no learning)"}
        ax.barh([labels[k] for k, _ in fam][::-1], [v for _, v in fam][::-1], color=SERIES[0], height=0.6)
        for yv, v in enumerate([v for _, v in fam][::-1]):
            ax.text(v + 0.5, yv, f"{v:.1f}°", va="center", fontsize=8, color=INK2)
        ax.set_xlabel("Test bearing MAE (deg)"); ax.set_title("Ablation: separately trained models")
        ax.grid(axis="y", visible=False)
        ax = axes[1]
        perm = res["permutation_importance_combined_model"]
        ks = ["timing", "spectral"]
        ax.barh(["shuffle timing input", "shuffle spectral input"][::-1], [perm[k]["increase_deg"] for k in ks][::-1],
                color=SERIES[1], height=0.5)
        for yv, k in enumerate(ks[::-1]):
            ax.text(perm[k]["increase_deg"] + 0.5, yv, f"+{perm[k]['increase_deg']:.1f}°", va="center", fontsize=8,
                    color=INK2)
        ax.set_xlabel("Increase in bearing MAE (deg)"); ax.set_title("Permutation importance (combined model)")
        ax.grid(axis="y", visible=False)
        fig.tight_layout(); fig.savefig(RES / "feature_importance.png", dpi=150); plt.close(fig)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
