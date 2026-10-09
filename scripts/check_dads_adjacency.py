"""Test the assumption behind the DADS split: clips with consecutive IDs come from the same recording.

DADS renames every clip to drone-N.wav / no-drone-N.wav and ships no recording IDs. If consecutive N are
consecutive chunks of one recording, a random clip-level split would leak (near-identical audio in train and
test). This script measures it: for every pair (N, N+1) it compares a spectral fingerprint (mean log-mel profile)
and sample-rate/duration against random pairs, then estimates "run lengths" of same-recording chunks.

Output: results/dads_adjacency.json (+ a histogram png). Uses whatever DADS shards are on disk.

With --all it instead fingerprints EVERY DADS clip (parallel over shards) and writes
data/processed/dads_fingerprints.parquet, which prepare_data.py uses to cut split blocks only at points where
consecutive clips are NOT near-duplicates (so no recording run can straddle two splits).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from acousticwatch.features import load_audio  # noqa: E402


def fingerprint(x):
    import librosa
    m = librosa.feature.melspectrogram(y=x[:16000 * 2], sr=16000, n_fft=1024, hop_length=320, n_mels=64)
    v = np.log(m + 1e-8).mean(1)
    return (v - v.mean()) / (v.std() + 1e-8)


def main(max_shards=6):
    files = sorted((ROOT / "data" / "raw" / "dads" / "data").glob("*.parquet"))[:max_shards]
    recs = {}
    for fp in files:
        for r in pq.read_table(fp, columns=["audio", "label"]).to_pylist():
            p = r["audio"]["path"]
            pre, n = p.rsplit("-", 1)
            x = load_audio(r["audio"]["bytes"])
            if len(x) < 1600:
                continue
            recs[(pre, int(n[:-4]))] = (fingerprint(x), len(x))
    keys = sorted(recs)
    adj, rnd = [], []
    rng = np.random.default_rng(0)
    for pre, n in keys:
        if (pre, n + 1) in recs:
            a, b = recs[(pre, n)], recs[(pre, n + 1)]
            adj.append(float(np.dot(a[0], b[0]) / 64))
            o = keys[rng.integers(len(keys))]
            rnd.append(float(np.dot(a[0], recs[o][0]) / 64))
    adj, rnd = np.array(adj), np.array(rnd)
    thr = float(np.quantile(rnd, 0.95))
    # run lengths: consecutive IDs whose fingerprints stay above the random-pair 95th percentile
    runs, cur = [], 1
    for pre, n in keys:
        if (pre, n + 1) in recs:
            s = np.dot(recs[(pre, n)][0], recs[(pre, n + 1)][0]) / 64
            if s > thr:
                cur += 1
                continue
        runs.append(cur); cur = 1
    runs = np.array(runs)
    out = {
        "clips_analysed": len(recs), "adjacent_pairs": int(len(adj)),
        "adjacent_similarity_median": float(np.median(adj)), "random_similarity_median": float(np.median(rnd)),
        "random_95th_percentile": thr,
        "fraction_adjacent_above_random95": float((adj > thr).mean()),
        "run_length_quantiles": {q: float(np.quantile(runs, q)) for q in [0.5, 0.9, 0.99, 0.999]},
        "max_run_length": int(runs.max()),
        "note": "IDs are only partly present when not all shards are analysed, which cuts runs short.",
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "dads_adjacency.json").write_text(json.dumps(out, indent=2))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(rnd, bins=60, alpha=0.6, density=True, label="random pairs")
    ax.hist(adj, bins=60, alpha=0.6, density=True, label="consecutive IDs (N, N+1)")
    ax.axvline(thr, color="k", ls="--", lw=1, label="random 95th pct")
    ax.set_xlabel("spectral fingerprint similarity"); ax.set_ylabel("density"); ax.legend(frameon=False)
    ax.set_title("DADS: consecutive clip IDs are near-duplicates")
    fig.tight_layout(); fig.savefig(ROOT / "results" / "dads_adjacency.png", dpi=130)
    print(json.dumps(out, indent=2))


def _fp_shard(fp):
    out = []
    for r in pq.read_table(fp, columns=["audio"]).to_pylist():
        p = r["audio"]["path"]
        x = load_audio(r["audio"]["bytes"])
        f = fingerprint(x) if len(x) >= 1600 else np.zeros(64)
        out.append((p, f.astype(np.float16)))
    return out


def fingerprint_all(procs=8):
    import pandas as pd
    from multiprocessing import Pool
    files = sorted((ROOT / "data" / "raw" / "dads" / "data").glob("*.parquet"))
    with Pool(procs) as pool:
        rows = [r for part in pool.imap_unordered(_fp_shard, files) for r in part]
    df = pd.DataFrame({"path": [r[0] for r in rows]})
    df["prefix"] = df.path.str.extract(r"^(.*?)-\d+\.wav$")[0]
    df["num"] = df.path.str.extract(r"-(\d+)\.wav$")[0].astype(int)
    F = np.stack([r[1] for r in rows]).astype(np.float32)
    df = df.join(pd.DataFrame(F, columns=[f"f{i}" for i in range(64)]))
    out = ROOT / "data" / "processed" / "dads_fingerprints.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"wrote {len(df)} fingerprints -> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        fingerprint_all()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
