"""DDL data access: parse labels from file names, assign leakage-safe splits, load 8-channel audio.

Leakage rule: the unit of splitting is the RECORDING FOLDER (one autopilot trial, e.g. "310321-T008" = 31 Mar 2021,
trial 8). All 100 ms clips of a trial are consecutive slices of one continuous recording of one flight, so they
are near-duplicates of each other; a random clip-level split would put neighbouring slices of the same flight into
train and test. Here every trial goes wholly into exactly one split.

To keep the splits comparable, trials are stratified by drone model and by median range: within each drone
model, trials are sorted by median range and dealt round-robin (positions 0..4 -> train, train, val, train, test).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "ddl"
PROC = ROOT / "data" / "processed" / "ddl"
RX = re.compile(r"(\d{14})(MINI|PRO4|XXXX)(\d{3})(\d{3})([-\d]{3})(\d{4})([RS])(\d{6})-(T\d{3})-(\d{6})\.wav")
MIN_RANGE_M, MIN_ALT_M = 10, 20     # exclude take-off/landing clips: GPS bearing is unreliable that close
ROUND_ROBIN = ["train", "train", "val", "train", "test"]


def parse(name):
    m = RX.match(name)
    if not m:
        return None
    ts, drone, b, r, a, t, typ, date, trial, seq = m.groups()
    return {"name": name, "drone": drone, "bearing": int(b), "range": int(r), "alt": int(a.replace("-", "")) *
            (-1 if "-" in a else 1), "temp_k": int(t) / 10, "folder": f"{date}-{trial}", "date": date,
            "seq": int(seq), "utc": ts}


def assign_splits(meta):
    folders = meta.groupby("folder").agg(drone=("drone", "first"), med_range=("range", "median")).reset_index()
    out = {}
    for drone, g in folders.groupby("drone"):
        for k, f in enumerate(g.sort_values(["med_range", "folder"]).folder):
            out[f] = ROUND_ROBIN[k % len(ROUND_ROBIN)]
    return meta.folder.map(out)


def build_meta():
    """Scan downloaded shards -> meta DataFrame (one row per clip, with shard + index)."""
    rows, skipped = [], 0
    for sh in sorted((RAW / "shards").glob("shard_*.npz")):
        names = np.load(sh)["names"]
        for i, n in enumerate(names):
            r = parse(str(n))
            if r is None:
                skipped += 1
                continue
            rows.append({**r, "shard": sh.name, "idx": i})
    meta = pd.DataFrame(rows)
    meta["excluded"] = (meta.range < MIN_RANGE_M) | (meta.alt < MIN_ALT_M)
    meta = meta[~meta.excluded].drop(columns="excluded").reset_index(drop=True)
    meta["split"] = assign_splits(meta)
    meta = meta.sort_values(["folder", "seq"]).reset_index(drop=True)
    PROC.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(PROC / "meta.parquet")
    return meta, skipped


def load_audio(meta):
    """Load the clips listed in `meta` (in order) into one float16 array (N, 8, 4800) in RAM (~4.5 GB for all)."""
    out = np.zeros((len(meta), 8, 4800), np.float16)
    pos = {k: v for v, k in enumerate(meta.index)}
    for sh, g in meta.groupby("shard"):
        a = np.load(RAW / "shards" / sh)["audio"]
        n = min(a.shape[2], 4800)
        out[[pos[k] for k in g.index], :, :n] = a[g.idx.values, :, :n]
    return out


def split_report(meta):
    rep = {}
    for s, g in meta.groupby("split"):
        rep[s] = {"clips": int(len(g)), "folders": sorted(g.folder.unique().tolist()),
                  "per_drone": g.drone.value_counts().to_dict(),
                  "range_m_quantiles": {q: float(g.range.quantile(q)) for q in [0.05, 0.5, 0.95]}}
    shared = set(meta[meta.split == "train"].folder) & (set(meta[meta.split != "train"].folder))
    rep["folders_shared_between_train_and_eval"] = sorted(shared)
    return rep


if __name__ == "__main__":
    m, sk = build_meta()
    print(json.dumps({"clips": len(m), "unparsed": sk, **split_report(m)}, indent=1))
