"""Evaluate the trained detector (task 1): accuracy, recall, false-positive rate, per-category false positives.

* Decision thresholds are chosen on the VALIDATION split only (two operating points):
    - "standard": the lowest threshold (= highest recall) whose val false-positive rate is <= 1 %
    - "strict":   the lowest threshold whose val false-positive rate is <= 0.1 %
  plus the default 0.5 threshold for reference.
* Reported on the held-out TEST split (same source collections, disjoint recordings) and on the EXTERNAL
  Svanström set (unseen microphones / sites / drones).
* 95 % confidence intervals by bootstrapping recording groups (not windows), so correlated windows from
  one recording are not counted as independent evidence.

Outputs: results/metrics.json, results/per_category.csv, results/false_positives_engine.csv,
         results/roc.png, results/fpr_by_category.png, results/predictions.parquet
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from acousticwatch import gpu_guard                                   # noqa: E402
from acousticwatch.data import load_store                             # noqa: E402
from acousticwatch.infer import get_windows, load_model, score_wave  # noqa: E402
from acousticwatch.plotstyle import INK2, SERIES, plt                 # noqa: E402

RESULTS = ROOT / "results"
TAG = "detect_v1"
ENGINE_LIKE = ["esc50:engine", "us8k:engine_idling", "aerosonicdb:aircraft", "esc50:airplane",
               "esc50:helicopter", "svanstrom:helicopter", "esc50:chainsaw", "esc50:train",
               "esc50:hand_saw", "us8k:drilling", "us8k:jackhammer", "us8k:air_conditioner",
               "esc50:vacuum_cleaner"]


def metrics(y, p, thr):
    pred = p >= thr
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    fp, tn = int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum())
    rec = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1); prec = tp / max(tp + fp, 1)
    return {"threshold": float(thr), "n_pos": tp + fn, "n_neg": fp + tn, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "accuracy": (tp + tn) / max(len(y), 1), "balanced_accuracy": (rec + 1 - fpr) / 2,
            "recall": rec, "false_positive_rate": fpr, "precision": prec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9)}


def group_bootstrap(df, thr, n=300, seed=0):
    """95% CI for recall / FPR / balanced accuracy, resampling whole recording groups."""
    rng = np.random.default_rng(seed)
    groups = df.group.unique()
    by = {g: d for g, d in df.groupby("group")}
    out = {"recall": [], "false_positive_rate": [], "balanced_accuracy": []}
    for _ in range(n):
        d = pd.concat([by[g] for g in rng.choice(groups, len(groups))])
        m = metrics(d.label.values, d.prob.values, thr)
        for k in out:
            out[k].append(m[k])
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in out.items()}


def pick_thresholds(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    lowest_with = lambda limit: float(thr[np.where(fpr <= limit)[0][-1]])  # noqa: E731
    return {"0.5": 0.5, "standard": lowest_with(0.01), "strict": lowest_with(0.001)}


def main():
    gpu_guard.ensure_power_limit()
    guard = gpu_guard.ThermalGuard()
    man, wav = load_store()
    model, mel, ck = load_model(TAG)
    ev = man[man.split.isin(["val", "test", "external"])].copy()
    idx, x = get_windows(wav, ev.index.values)
    ev.loc[idx, "prob"] = score_wave(model, mel, x, guard=guard)
    ev.to_parquet(RESULTS / "predictions.parquet")

    va, te, ex = (ev[ev.split == s] for s in ["val", "test", "external"])
    thr = pick_thresholds(va.label.values, va.prob.values)
    res = {"checkpoint_epoch": int(ck["epoch"]), "thresholds_from_val": thr, "splits": {}}
    for name, d in [("val", va), ("test", te), ("external_svanstrom", ex)]:
        r = {"roc_auc": float(roc_auc_score(d.label, d.prob)), "pr_auc": float(average_precision_score(d.label, d.prob)),
             "n_windows": int(len(d)), "n_groups": int(d.group.nunique())}
        for op, t in thr.items():
            r[op] = metrics(d.label.values, d.prob.values, t)
            if name != "val" and op != "0.5":
                r[op]["ci95_group_bootstrap"] = group_bootstrap(d, t)
        # clip-level: average window probabilities per source file (one decision per recording/clip)
        c = d.groupby(["source", "file"]).agg(label=("label", "first"), prob=("prob", "mean"))
        r["clip_level_standard"] = metrics(c.label.values, c.prob.values, thr["standard"])
        res["splits"][name] = r

    # per-category table (test + external), at both operating points
    rows = []
    for name, d in [("test", te), ("external", ex)]:
        for cat, g in d.groupby("category"):
            row = {"split": name, "category": cat, "label": int(g.label.iloc[0]), "n_windows": len(g),
                   "n_groups": g.group.nunique(), "hard_negative": bool(g.hard_negative.iloc[0]),
                   "mean_prob": g.prob.mean()}
            for op in ["standard", "strict"]:
                pos = (g.prob >= thr[op]).mean()
                row[("recall_" if row["label"] else "fpr_") + op] = pos
            rows.append(row)
    pc = pd.DataFrame(rows).sort_values(["split", "label", "fpr_standard"], ascending=[True, False, False])
    pc.to_csv(RESULTS / "per_category.csv", index=False, float_format="%.4f")

    neg = pc[pc.label == 0]
    eng = neg[neg.category.isin(ENGINE_LIKE)]
    res["engine_like_negatives"] = {
        r.category + f" ({r.split})": {"fpr_standard": r.fpr_standard, "fpr_strict": r.fpr_strict,
                                       "n_windows": int(r.n_windows)} for r in eng.itertuples()}
    te_neg = te[te.label == 0]
    hard, easy = te_neg[te_neg.hard_negative], te_neg[~te_neg.hard_negative]
    res["test_fpr_hard_vs_other_negatives_standard"] = {
        "hard_negatives": float((hard.prob >= thr["standard"]).mean()),
        "other_negatives": float((easy.prob >= thr["standard"]).mean())}
    # worst engine-like false positives for listening
    fp = ev[(ev.label == 0) & ev.category.isin(ENGINE_LIKE) & (ev.split != "val")].nlargest(40, "prob")
    fp[["split", "category", "file", "win_in_file", "prob"]].to_csv(RESULTS / "false_positives_engine.csv", index=False)

    (RESULTS / "metrics.json").write_text(json.dumps(res, indent=2))

    # ---- figures
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for (name, d), c in zip([("DADS-based test", te), ("External (Svanström)", ex)], SERIES):
        f, t, _ = roc_curve(d.label, d.prob)
        ax.plot(f, t, color=c, label=f"{name}  AUC {roc_auc_score(d.label, d.prob):.3f}")
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("False-positive rate"); ax.set_ylabel("Recall (true-positive rate)")
    ax.set_title("ROC: drone vs not-drone (0.5 s windows)"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(RESULTS / "roc.png", dpi=150); plt.close(fig)

    tn = neg[neg.split == "test"].sort_values("fpr_standard")
    tn = pd.concat([tn[~tn.hard_negative].tail(8), tn[tn.hard_negative]]).sort_values("fpr_standard")
    exn = neg[neg.split == "external"]
    tn = pd.concat([tn, exn]).sort_values("fpr_standard")
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(tn) + 1.2))
    cols = [SERIES[1] if h else SERIES[0] for h in tn.hard_negative]
    ax.barh(tn.category + np.where(tn.split == "external", " (ext)", ""), tn.fpr_standard * 100, color=cols, height=0.7)
    for yv, v in enumerate(tn.fpr_standard * 100):
        ax.text(v + 0.3, yv, f"{v:.1f}%", va="center", fontsize=7, color=INK2)
    ax.set_xlabel("False-positive rate at the standard threshold (%)")
    ax.set_title("False alarms by negative class")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES[1], label="hard negative (engine / aircraft / machinery)"),
                       Patch(color=SERIES[0], label="other negatives (worst 8 shown)")], loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(RESULTS / "fpr_by_category.png", dpi=150); plt.close(fig)
    print(json.dumps({k: res["splits"][k]["standard"] for k in res["splits"]}, indent=1))
    print(json.dumps(res["engine_like_negatives"], indent=1))


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
