"""Task 3: which spectral features (harmonic structure) drive correct detections?

Four analyses on held-out TEST windows (thresholds from validation):
  1. Frequency-band occlusion: replace one band of the log-mel input (16 bands x 4 mel bins) with the training
     mean and measure the drop in drone probability. Done for correct detections (true positives) and for
     false alarms on hard negatives, to compare what drives each.
  2. Harmonicity: cepstral peak prominence (CPP, f0 search 40-800 Hz) of every window. A strong CPP means a
     clear harmonic stack (rotor blade-pass frequency + multiples). Compared across hits / misses / false alarms
     / correct rejections, and correlated (Spearman) with the model's drone probability.
  3. Causal harmonic notch test on correctly detected, clearly harmonic drone windows:
        a) remove harmonics      - notch +-w Hz around k*f0 (k = 1..up to 8 kHz)
        b) remove between them   - identical notch widths centred on (k+0.5)*f0 (equal energy-removal control)
        c) keep harmonics only   - zero everything except the harmonic notches
     If (a) hurts much more than (b), and (c) keeps detections, the model relies on harmonic structure.
  4. Example spectrograms (hits, misses, engine false alarms) with f0 harmonics overlaid.

Outputs: results/explain_summary.json, results/band_importance.png, results/mean_spectra.png,
         results/harmonicity.png, results/harmonic_notch_test.png, results/example_spectrograms.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from acousticwatch import gpu_guard                                   # noqa: E402
from acousticwatch.data import load_store                             # noqa: E402
from acousticwatch.features import N_MELS, SR, WIN, mel_centre_freqs  # noqa: E402
from acousticwatch.infer import get_windows, load_model, score_wave  # noqa: E402
from acousticwatch.plotstyle import INK2, SERIES, plt            # noqa: E402

RESULTS = ROOT / "results"
TAG = "detect_v1"
DEV = "cuda"


def cepstral_harmonicity(x, fmin=40, fmax=800):
    """x: (N, WIN) tensor -> (cpp, f0). CPP = cepstral peak minus regression baseline in the f0 quefrency range."""
    X = torch.fft.rfft(x * torch.hann_window(WIN), dim=1)
    logmag = torch.log(X.abs() + 1e-6)
    cep = torch.fft.irfft(logmag, n=WIN, dim=1)
    qlo, qhi = int(SR / fmax), int(SR / fmin)
    seg = cep[:, qlo:qhi]
    q = torch.arange(qlo, qhi, dtype=torch.float32)
    # linear baseline per row (least squares), CPP = peak height above it
    A = torch.stack([q, torch.ones_like(q)], 1)
    coef = torch.linalg.lstsq(A, seg.T).solution          # (2, N)
    base = (A @ coef).T
    resid = seg - base
    k = resid.argmax(1)
    return resid.max(1).values.numpy(), (SR / (qlo + k).float()).numpy()


def comb_mask(f0, mode, rel_w=0.04, min_w=4.0):
    """Frequency-domain mask (N, WIN//2+1). mode: 'remove_harm', 'remove_between', 'keep_harm'."""
    f = torch.fft.rfftfreq(WIN, 1 / SR)[None]
    f0 = torch.as_tensor(f0, dtype=torch.float32)[:, None]
    w = torch.clamp(rel_w * f0, min=min_w)
    off = 0.5 if mode == "remove_between" else 0.0
    r = (f / f0 - off)
    near = (torch.abs(r - torch.round(r)) * f0 <= w) & (torch.round(r) >= (1 if off == 0 else 0)) & (f <= 8000)
    if mode == "keep_harm":
        return near.float()
    return (~near).float()


@torch.no_grad()
def band_occlusion(model, mel, x, n_bands=16, bs=256):
    per = N_MELS // n_bands
    drops = np.zeros(n_bands)
    base_all, n = 0.0, 0
    for s in range(0, len(x), bs):
        S = mel(x[s:s + bs].to(DEV))
        with torch.autocast("cuda", dtype=torch.float16):
            base = torch.sigmoid(model(S).float())
        for b in range(n_bands):
            S2 = S.clone()
            S2[:, :, b * per:(b + 1) * per, :] = model.mu[:, :, b * per:(b + 1) * per, :]
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.sigmoid(model(S2).float())
            drops[b] += (base - p).sum().item()
        base_all += base.sum().item(); n += len(S)
    return drops / n, base_all / n


def main(n=2000):
    gpu_guard.ensure_power_limit()
    guard = gpu_guard.ThermalGuard()
    rng = np.random.default_rng(0)
    man, wav = load_store()
    model, mel, _ = load_model(TAG)
    pred = pd.read_parquet(RESULTS / "predictions.parquet")
    thr = json.loads((RESULTS / "metrics.json").read_text())["thresholds_from_val"]["standard"]
    te = pred[pred.split.isin(["test", "external"])].copy()
    te["outcome"] = np.select(
        [(te.label == 1) & (te.prob >= thr), (te.label == 1) & (te.prob < thr),
         (te.label == 0) & (te.prob >= thr) & te.hard_negative, (te.label == 0) & (te.prob < thr) & te.hard_negative,
         (te.label == 0) & (te.prob >= thr), (te.label == 0) & (te.prob < thr)],
        ["hit", "miss", "false alarm (hard neg)", "correct reject (hard neg)", "false alarm (other)",
         "correct reject (other)"])

    def pick(mask, k=n):
        ids = te.index.values[mask]
        return ids if len(ids) <= k else rng.choice(ids, k, replace=False)

    # ---- 2. harmonicity for every outcome group (test + external)
    cpp_rows = []
    for oc in te.outcome.unique():
        ids, x = get_windows(wav, pick((te.outcome == oc).values))
        cpp, f0 = cepstral_harmonicity(x)
        cpp_rows.append(pd.DataFrame({"row": ids, "outcome": oc, "cpp": cpp, "f0": f0,
                                      "prob": te.loc[ids, "prob"].values, "label": te.loc[ids, "label"].values,
                                      "category": te.loc[ids, "category"].values}))
    H = pd.concat(cpp_rows)
    H.to_parquet(RESULTS / "harmonicity.parquet")
    summ = {"threshold_standard": thr, "harmonicity_by_outcome": H.groupby("outcome").cpp.describe()[
        ["count", "25%", "50%", "75%"]].round(4).to_dict(orient="index")}
    for lab, name in [(1, "drones"), (0, "negatives")]:
        h = H[H.label == lab]
        r = spearmanr(h.cpp, h.prob)
        summ[f"spearman_cpp_vs_prob_{name}"] = {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(h))}
    hits = H[H.outcome == "hit"]
    summ["drone_f0_hz_quantiles_hits"] = {q: float(np.quantile(hits.f0, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}

    # ---- 1. band occlusion for hits (DADS test) and hard-negative false alarms
    fc = mel_centre_freqs()
    edges = [(fc[b * 4], fc[b * 4 + 3]) for b in range(16)]
    occ = {}
    for name, mask in [("hits (correct drone detections)", (te.outcome == "hit") & (te.split == "test")),
                       ("false alarms on hard negatives", te.outcome == "false alarm (hard neg)")]:
        ids, x = get_windows(wav, pick(mask.values, 1500))
        if len(ids) < 10:
            continue
        guard.check("occlusion")
        d, base = band_occlusion(model, mel, x)
        occ[name] = {"n": int(len(ids)), "mean_prob": base, "prob_drop_per_band": d.tolist()}
    summ["band_occlusion"] = {"band_edges_hz": [[round(a), round(b)] for a, b in edges], **occ}

    # ---- 3. causal harmonic notch test (clearly harmonic, correctly detected test drones)
    hh = hits[(hits.cpp > hits.cpp.quantile(0.5))]
    ids = hh.row.values[:1500]
    _, x = get_windows(wav, ids)
    f0 = hh.set_index("row").loc[np.sort(ids), "f0"].values
    X = torch.fft.rfft(x)
    notch = {"n_windows": int(len(ids)), "f0_median_hz": float(np.median(f0))}
    p0 = score_wave(model, mel, x, guard=guard)
    notch["original"] = {"recall": float((p0 >= thr).mean()), "mean_prob": float(p0.mean())}
    for mode in ["remove_harm", "remove_between", "keep_harm"]:
        m = comb_mask(f0, mode)
        y = torch.fft.irfft(X * m, n=WIN)
        frac_removed = float(1 - m.mean())
        e_removed = float(1 - (y.pow(2).sum() / x.pow(2).sum()))
        p = score_wave(model, mel, y, guard=guard)
        notch[mode] = {"recall": float((p >= thr).mean()), "mean_prob": float(p.mean()),
                       "fraction_of_spectrum_zeroed": frac_removed, "fraction_of_energy_removed": e_removed}
    summ["harmonic_notch_test"] = notch
    (RESULTS / "explain_summary.json").write_text(json.dumps(summ, indent=2))

    # ---- figures
    centres = np.array([np.sqrt(a * b) for a, b in edges])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for (name, o), c in zip(occ.items(), SERIES):
        ax.plot(centres, o["prob_drop_per_band"], color=c, marker="o", ms=4, label=f"{name} (n={o['n']})")
    ax.set_xscale("log"); ax.set_xlabel("Frequency band centre (Hz)")
    ax.set_ylabel("Mean drop in drone probability\nwhen this band is hidden")
    ax.set_title("Which frequency bands the detector relies on"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(RESULTS / "band_importance.png", dpi=150); plt.close(fig)

    # mean log-mel spectra: hits vs engine-like correct rejects vs engine-like false alarms
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for (oc, lab), c in zip([("hit", "drone hits"), ("correct reject (hard neg)", "hard negatives, correctly rejected"),
                             ("false alarm (hard neg)", "hard negatives, false alarms")], SERIES):
        ids, x = get_windows(wav, pick((te.outcome == oc).values, 800))
        if len(ids) == 0:
            continue
        with torch.no_grad():
            S = mel(x.to(DEV)).mean(dim=(0, 1, 3)).cpu().numpy()
        ax.plot(mel_centre_freqs(), S - S.mean(), color=c, label=f"{lab} (n={len(ids)})")
    ax.set_xscale("log"); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Mean log-mel energy (centred)")
    ax.set_title("Average spectra: drones vs engine-like sounds"); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(RESULTS / "mean_spectra.png", dpi=150); plt.close(fig)

    order = ["hit", "miss", "false alarm (hard neg)", "correct reject (hard neg)", "false alarm (other)",
             "correct reject (other)"]
    order = [o for o in order if o in set(H.outcome)]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    data = [H[H.outcome == o].cpp.values for o in order]
    bp = ax.boxplot(data, vert=False, labels=[f"{o} (n={len(d)})" for o, d in zip(order, data)], showfliers=False,
                    patch_artist=True, medianprops={"color": "#0b0b0b"})
    for patch, o in zip(bp["boxes"], order):
        patch.set_facecolor(SERIES[0] if H[H.outcome == o].label.iloc[0] == 1 else SERIES[1]); patch.set_alpha(0.8)
    ax.set_xlabel("Harmonicity (cepstral peak prominence; higher = clearer harmonic stack)")
    ax.set_title("Harmonic structure by detection outcome"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(RESULTS / "harmonicity.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    labels = ["original", "remove between harmonics\n(control)", "remove harmonics", "keep harmonics only"]
    keys = ["original", "remove_between", "remove_harm", "keep_harm"]
    vals = [notch[k]["recall"] * 100 for k in keys]
    ax.barh(labels[::-1], vals[::-1], color=[SERIES[0], SERIES[1], SERIES[0], SERIES[0]][::-1], height=0.6)
    for yv, (v, k) in enumerate(zip(vals[::-1], keys[::-1])):
        extra = "" if k == "original" else f"  ({notch[k]['fraction_of_energy_removed']*100:.0f}% energy removed)"
        ax.text(v + 1, yv, f"{v:.1f}%{extra}", va="center", fontsize=7, color=INK2)
    ax.set_xlim(0, 160); ax.set_xticks(range(0, 101, 20)); ax.set_xlabel("Recall on clearly-harmonic drone hits (%)")
    ax.set_title("Causal test: notching the harmonic comb"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(RESULTS / "harmonic_notch_test.png", dpi=150); plt.close(fig)

    # example spectrograms
    import librosa
    ex = []
    for oc, k in [("hit", 3), ("miss", 3), ("false alarm (hard neg)", 3)]:
        h = H[H.outcome == oc]
        ex += [(oc, r) for r in h.sample(min(k, len(h)), random_state=1).itertuples()]
    if ex:
        fig, axes = plt.subplots(3, 3, figsize=(10, 7.5))
        for ax, (oc, r) in zip(axes.flat, ex):
            x = np.asarray(wav[r.row], np.float32) / 32768
            S = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=2048, hop_length=128)), ref=np.max)
            ax.imshow(S[:513], origin="lower", aspect="auto", cmap="magma",
                      extent=[0, WIN / SR, 0, 4000], vmin=-80, vmax=0)
            for k in range(1, int(4000 / r.f0) + 1):
                ax.axhline(k * r.f0, color="white", lw=0.4, ls=":", alpha=0.6)
            ax.set_title(f"{oc}\n{r.category}  p={r.prob:.2f}  f0≈{r.f0:.0f} Hz", fontsize=8)
            ax.set_ylabel("Hz"); ax.set_xlabel("s"); ax.grid(False)
        for ax in axes.flat[len(ex):]:
            ax.axis("off")
        fig.suptitle("Example spectrograms (dotted: harmonics of estimated f0)", x=0.01, ha="left", fontsize=11)
        fig.tight_layout(); fig.savefig(RESULTS / "example_spectrograms.png", dpi=130); plt.close(fig)
    print(json.dumps(summ, indent=1)[:4000])


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
