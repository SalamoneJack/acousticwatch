"""Simple detection baselines, scored exactly like the CNN (scripts/evaluate.py).

Does the CNN earn its complexity? Three baselines on the same leakage-safe splits:
  1. logreg_logmel   - logistic regression on per-mel-band mean + std of the SAME log-mel front-end (128 features),
                       trained on the training split (balanced class weights; hard negatives weighted 3x like the CNN)
  2. harmonicity     - no learning: cepstral peak prominence (f0 search 40-800 Hz), i.e. "how clear is the
                       harmonic stack" - the textbook drone cue
  3. highband_ratio  - no learning: fraction of energy in 2.7-7.7 kHz, the band the CNN turned out to rely on
For every method the operating thresholds are chosen on VALIDATION only ("standard": val FPR <= 1 %,
"strict": val FPR <= 0.1 %), then applied to the held-out test split and the external Svanström set.

Outputs: results/baselines/metrics.json, results/baselines/comparison.csv, results/baselines/roc_comparison.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
from acousticwatch import gpu_guard                                    # noqa: E402
from acousticwatch.data import batched, load_store                     # noqa: E402
from acousticwatch.features import SR, WIN, LogMel                     # noqa: E402
from acousticwatch.plotstyle import INK2, SERIES, plt                  # noqa: E402
from evaluate import metrics, pick_thresholds                          # noqa: E402
from explain import cepstral_harmonicity                               # noqa: E402

OUT = ROOT / "results" / "baselines"
DEV = "cuda"


@torch.no_grad()
def features(wav, idx, mel, guard):
    """Per-window: log-mel mean+std (128), cepstral peak prominence, 2.7-7.7 kHz energy ratio."""
    f = torch.fft.rfftfreq(WIN, 1 / SR)
    hb = ((f >= 2700) & (f <= 7700)).to(DEV)
    rows_idx, F, C, H = [], [], [], []
    for b, x in batched(wav, idx, 1024):
        guard.check("baselines")
        xg = x.to(DEV)
        S = mel(xg)[:, 0]                                            # (B, 64, T), DC removed inside LogMel
        F.append(torch.cat([S.mean(-1), S.std(-1)], 1).cpu())
        xd = xg - xg.mean(1, keepdim=True)
        P = torch.fft.rfft(xd * torch.hann_window(WIN, device=DEV)).abs() ** 2
        H.append(((P * hb).sum(1) / P.sum(1).clamp_min(1e-12)).cpu())
        C.append(torch.from_numpy(cepstral_harmonicity(xd.cpu())[0]))
        rows_idx.append(b)
    return np.concatenate(rows_idx), torch.cat(F).numpy(), torch.cat(C).numpy(), torch.cat(H).numpy()


def auc_ci(d, col, n=500, seed=0):
    """95% CI of ROC AUC, resampling whole recording groups (not windows)."""
    rng = np.random.default_rng(seed)
    groups = d.group.unique()
    by = {g: x for g, x in d.groupby("group")}
    vals = []
    for _ in range(n):
        x = pd.concat([by[g] for g in rng.choice(groups, len(groups))])
        if x.label.nunique() == 2:
            vals.append(roc_auc_score(x.label, x[col]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main(n_train=60000):
    gpu_guard.ensure_power_limit()
    guard = gpu_guard.ThermalGuard()
    OUT.mkdir(parents=True, exist_ok=True)
    man, wav = load_store()
    mel = LogMel().to(DEV)
    rng = np.random.default_rng(0)
    tr = man[man.split == "train"]
    tr = tr.loc[np.sort(rng.choice(tr.index.values, min(n_train, len(tr)), replace=False))]
    ev = man[man.split.isin(["val", "test", "external"])]

    i_tr, F_tr, _, _ = features(wav, tr.index.values, mel, guard)
    i_ev, F_ev, C_ev, H_ev = features(wav, ev.index.values, mel, guard)
    ytr = man.loc[i_tr, "label"].values
    w = np.where(man.loc[i_tr, "hard_negative"].values, 3.0, 1.0)
    w = w * np.where(ytr == 1, 0.5 / (ytr == 1).mean(), 0.5 / (ytr == 0).mean())
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(F_tr, ytr, logisticregression__sample_weight=w)

    scores = pd.DataFrame({"row": i_ev, "logreg_logmel": clf.predict_proba(F_ev)[:, 1],
                           "harmonicity": C_ev, "highband_ratio": H_ev}).set_index("row")
    ev = ev.join(scores)
    cnn = pd.read_parquet(ROOT / "results" / "detect_v1" / "predictions.parquet")[["prob"]]
    ev = ev.join(cnn.rename(columns={"prob": "cnn_v1"}))

    methods = ["cnn_v1", "logreg_logmel", "harmonicity", "highband_ratio"]
    ev[["split", "category", "label", "group", "hard_negative", *methods]].to_parquet(OUT / "scores.parquet")
    res, rows = {"n_train_windows_logreg": int(len(i_tr))}, []
    va, te, ex = (ev[ev.split == s] for s in ["val", "test", "external"])
    for m in methods:
        thr = pick_thresholds(va.label.values, va[m].values)
        r = {"thresholds_from_val": thr}
        for name, d in [("test", te), ("external_svanstrom", ex)]:
            auc = float(roc_auc_score(d.label, d[m]))
            lo, hi = auc_ci(d, m)
            r[name] = {"roc_auc": auc, "roc_auc_ci95_group_bootstrap": [lo, hi], **{op: metrics(d.label.values, d[m].values, thr[op]) for op in ["standard", "strict"]}}
            hard = d[(d.label == 0) & d.hard_negative]
            r[name]["hard_negative_fpr_standard"] = float((hard[m] >= thr["standard"]).mean())
            rows.append({"method": m, "split": name, "roc_auc": auc, "auc_ci_low": lo, "auc_ci_high": hi,
                         "recall_standard": r[name]["standard"]["recall"],
                         "fpr_standard": r[name]["standard"]["false_positive_rate"],
                         "recall_strict": r[name]["strict"]["recall"],
                         "fpr_strict": r[name]["strict"]["false_positive_rate"],
                         "hard_negative_fpr_standard": r[name]["hard_negative_fpr_standard"]})
        res[m] = r
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "comparison.csv", index=False, float_format="%.4f")
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2))

    labels = {"cnn_v1": "CNN (v1)", "logreg_logmel": "Logistic regression, log-mel stats",
              "harmonicity": "Harmonicity rule (no learning)", "highband_ratio": "2.7-7.7 kHz energy rule (no learning)"}
    colors = {"cnn_v1": SERIES[0], "logreg_logmel": SERIES[1], "harmonicity": SERIES[2], "highband_ratio": INK2}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, (name, d) in zip(axes, [("Held-out test", te), ("External (Svanström)", ex)]):
        for m in methods:
            f, t, _ = roc_curve(d.label, d[m])
            ax.plot(f, t, color=colors[m], lw=2 if m != "highband_ratio" else 1.5,
                    ls="-" if m != "highband_ratio" else "--",
                    label=f"{labels[m]}  AUC {roc_auc_score(d.label, d[m]):.3f}")
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
        ax.set_title(name); ax.set_xlabel("False-positive rate"); ax.legend(loc="lower right", fontsize=7)
    axes[0].set_ylabel("Recall")
    fig.tight_layout(); fig.savefig(OUT / "roc_comparison.png", dpi=150); plt.close(fig)
    pd.set_option("display.width", 200)
    print(comp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
