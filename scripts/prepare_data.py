"""Build the leakage-safe window store from data/raw/ -> data/processed/.

Outputs
  data/processed/windows.bin      int16 [N, 8000]  (0.5 s @ 16 kHz, peak-normalised)
  data/processed/manifest.parquet one row per window: source, category, label, group, split, hard_negative, file
  results/data_summary.json       counts per split / source / category

Leakage rules (see SUMMARY.md, "How leakage was avoided"):
  1. Splits are assigned per *recording group*, never per window: every 0.5 s window cut from a file inherits
     that file's split, and files that come from the same original recording share a group.
  2. ESC-50: official folds (grouped by Freesound source file): 1-3 train, 4 val, 5 test.
     UrbanSound8K: official folds (grouped by Freesound ID): 1-8 train, 9 val, 10 test.
  3. DADS has no recording IDs (files are renamed drone-N / no-drone-N). Consecutive N are consecutive chunks of
     the same source recording (verified in results/dads_adjacency.json; runs reach several hundred IDs), so DADS
     is split in contiguous ID blocks (>= 1000 IDs) that are only ever cut where consecutive clips are NOT
     near-duplicates (fingerprint similarity below the random-pair 95th percentile), plus a small guard band.
     Audit: results/dads_split_audit.json (near-duplicate pairs across splits must be 0).
  4. AeroSonicDB: split by recording *session* (date + location), so one overflight's audio never straddles splits.
  5. Svanström dataset: never used for training or model selection; it is an external test domain only.
  6. Exact-duplicate audio (hash of the int16 window) is detected across everything; any window whose hash also
     occurs in a different split is dropped from val/test.
  7. A few Freesound source recordings appear in two official ESC-50 / US8K folds (under different labels).
     Each recording group is kept only in its earliest split (train < val < test); later copies are dropped.
  8. Normalisation statistics are computed later from the training split only (scripts/train.py).
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from acousticwatch.features import load_audio, to_int16, to_windows  # noqa: E402

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

HARD_NEG = {
    "esc50": {"airplane", "engine", "helicopter", "chainsaw", "hand_saw", "train", "vacuum_cleaner"},
    "us8k": {"engine_idling", "drilling", "jackhammer", "air_conditioner"},
    "aerosonicdb": {"aircraft"},
    "svanstrom": {"helicopter"},
}


class Store:
    def __init__(self):
        PROC.mkdir(parents=True, exist_ok=True)
        self.f = open(PROC / "windows.bin", "wb")
        self.rows, self.n = [], 0

    def add(self, wins, **meta):
        if len(wins) == 0:
            return
        w16 = to_int16(wins)
        self.f.write(w16.tobytes())
        for k, w in enumerate(w16):
            self.rows.append({**meta, "win_in_file": k, "hash": hashlib.md5(w.tobytes()).hexdigest()[:16]})
        self.n += len(w16)

    def close(self):
        self.f.close()
        return pd.DataFrame(self.rows)


# ----------------------------------------------------------------------------------------------- ESC-50
def add_esc50(st, rng):
    base = RAW / "esc50" / "ESC-50-master"
    meta = pd.read_csv(base / "meta" / "esc50.csv")
    fold2split = {1: "train", 2: "train", 3: "train", 4: "val", 5: "test"}
    for r in tqdm(meta.itertuples(), total=len(meta), desc="esc50"):
        x = load_audio(str(base / "audio" / r.filename))
        # ESC-50 clips are zero-padded to 5 s; drop near-silent windows so padding isn't learned as a cue
        w = to_windows(x)
        w = w[np.sqrt((w ** 2).mean(1)) > 1e-4]
        st.add(w, source="esc50", category=f"esc50:{r.category}", label=0, group=f"esc50:{r.src_file}",
               split=fold2split[r.fold], hard_negative=r.category in HARD_NEG["esc50"], file=r.filename)


# ------------------------------------------------------------------------------------------ UrbanSound8K
def add_us8k(st, rng):
    files = sorted((RAW / "urbansound8k" / "data").glob("*.parquet"))
    split = lambda fold: "train" if fold <= 8 else ("val" if fold == 9 else "test")  # noqa: E731
    for fp in tqdm(files, desc="us8k"):
        t = pq.read_table(fp, columns=["audio", "fsID", "fold", "class", "slice_file_name"]).to_pylist()
        for r in t:
            x = load_audio(r["audio"]["bytes"])
            w = to_windows(x)
            w = w[np.sqrt((w ** 2).mean(1)) > 1e-4]
            st.add(w, source="us8k", category=f"us8k:{r['class']}", label=0, group=f"us8k:{r['fsID']}",
                   split=split(r["fold"]), hard_negative=r["class"] in HARD_NEG["us8k"], file=r["slice_file_name"])


# ------------------------------------------------------------------------------------------------- DADS
def dads_breaks(nums, F, thr):
    """Boolean array: True at position i if clips nums[i] and nums[i+1] are NOT near-duplicates
    (fingerprint similarity <= thr, or a gap in the ID sequence)."""
    sim = (F[:-1] * F[1:]).sum(1) / F.shape[1]
    return (sim <= thr) | (np.diff(nums) != 1)


def dads_split(nums, breaks, block, guard, frac, rng):
    """Break-aware contiguous-block split.

    Blocks of >= `block` consecutive IDs are closed only at a similarity break, so a run of near-duplicate chunks
    (one recording) can never be cut in two. Blocks are then assigned to splits as whole units; as an extra
    margin, `guard` IDs are dropped at block edges whose neighbouring block is in a different split.
    Returns dict id -> (split, block_index)."""
    blocks, cur = [], []
    for i, n in enumerate(nums):
        cur.append(n)
        if len(cur) >= block and (i == len(nums) - 1 or breaks[i]):
            blocks.append(cur); cur = []
    if cur:
        if blocks:
            blocks[-1] += cur      # trailing remainder joins the last block (no break needed at the end)
        else:
            blocks.append(cur)
    order = rng.permutation(len(blocks))
    cut1, cut2 = int(frac[0] * len(blocks)), int((frac[0] + frac[1]) * len(blocks))
    split = np.empty(len(blocks), object)
    for rank, bi in enumerate(order):
        split[bi] = "train" if rank < cut1 else ("val" if rank < cut2 else "test")
    out = {}
    for bi, b in enumerate(blocks):
        lo = guard if bi > 0 and split[bi - 1] != split[bi] else 0
        hi = guard if bi < len(blocks) - 1 and split[bi + 1] != split[bi] else 0
        for n in b[lo:len(b) - hi]:
            out[n] = (split[bi], bi)
    return out


def add_dads(st, rng, block, guard, max_clips_per_label, max_windows_per_clip):
    files = sorted((RAW / "dads" / "data").glob("*.parquet"))
    names = []   # first pass: file names + labels only (no audio bytes)
    for fp in files:
        t = pq.ParquetFile(fp).read(columns=["audio.path", "label"])
        paths = t.column("audio").combine_chunks().field("path").to_pylist() if "audio" in t.column_names \
            else t.column("audio.path").to_pylist()
        names += list(zip(paths, t.column("label").to_pylist()))
    df = pd.DataFrame(names, columns=["path", "label"])
    df["prefix"] = df.path.str.extract(r"^(.*?)-\d+\.wav$")[0]
    df["num"] = df.path.str.extract(r"-(\d+)\.wav$")[0].astype(int)
    # Fingerprints of every clip (scripts/check_dads_adjacency.py --all) locate the points where consecutive IDs
    # stop being near-duplicates; split blocks are only ever cut there.
    fps = pd.read_parquet(PROC / "dads_fingerprints.parquet")
    fcols = [f"f{i}" for i in range(64)]
    assignments, report = {}, {}
    for pre, g in df.groupby("prefix"):
        fg = fps[fps.prefix == pre].sort_values("num")
        F = np.nan_to_num(fg[fcols].values.astype(np.float32))
        nums = fg.num.values
        r = np.random.default_rng(7)
        a, b = r.integers(0, len(F), 20000), r.integers(0, len(F), 20000)
        thr = float(np.quantile((F[a] * F[b]).sum(1) / 64, 0.95))     # "near-duplicate" = above random-pair 95th pct
        brk = dads_breaks(nums, F, thr)
        sp = dads_split(nums, brk, block, guard, (0.7, 0.15, 0.15), rng)
        # audit: consecutive near-duplicate pairs whose two clips ended up in different splits (must be 0)
        cross = sum(1 for i in range(len(nums) - 1) if not brk[i] and nums[i] in sp and nums[i + 1] in sp
                    and sp[nums[i]][0] != sp[nums[i + 1]][0])
        report[pre] = {"clips": int(len(nums)), "near_duplicate_threshold": thr, "breaks": int(brk.sum()),
                       "blocks": int(len({v[1] for v in sp.values()})), "kept_after_guard": int(len(sp)),
                       "near_duplicate_pairs_across_splits": int(cross)}
        sel = g[g.num.isin(sp)]
        if len(sel) > max_clips_per_label:
            sel = sel.sample(max_clips_per_label, random_state=0)
        for p, n in zip(sel.path, sel.num):
            assignments[p] = (sp[n][0], f"dads:{pre}:{sp[n][1]}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "dads_split_audit.json").write_text(json.dumps(report, indent=2))
    print(f"dads: {len(df)} clips, keeping {len(assignments)} after guard bands / subsampling; audit {report}")
    for fp in tqdm(files, desc="dads"):
        pf = pq.ParquetFile(fp)
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=["audio", "label"]).to_pylist()
            for r in t:
                p = r["audio"]["path"]
                if p not in assignments:
                    continue
                split, group = assignments[p]
                x = load_audio(r["audio"]["bytes"])
                w = to_windows(x, max_windows=max_windows_per_clip, rng=rng)
                w = w[np.sqrt((w ** 2).mean(1)) > 1e-4]
                st.add(w, source="dads", category="dads:drone" if r["label"] == 1 else "dads:no-drone",
                       label=int(r["label"]), group=group, split=split, hard_negative=False, file=p)


# ------------------------------------------------------------------------------------------ AeroSonicDB
def add_aerosonicdb(st, rng, max_windows_per_file):
    base = RAW / "aerosonicdb"
    meta = pd.read_csv(base / "sample_meta.csv").dropna(subset=["class"])
    meta = meta.rename(columns={"class": "class_"})
    wavs = {p.name: p for p in base.rglob("*.wav")}
    # session = recording day (one session per date in AeroSonicDB); sessions are assigned to splits as whole units
    sess = meta["date"].astype(str).values
    usess = np.array(sorted(set(sess)))
    order = np.random.default_rng(1).permutation(len(usess))
    n = len(usess)
    s2split = {usess[i]: ("train" if r < 0.7 * n else ("val" if r < 0.85 * n else "test")) for r, i in enumerate(order)}
    for r, s in tqdm(zip(meta.itertuples(), sess), total=len(meta), desc="aerosonicdb"):
        p = wavs.get(r.filename)
        if p is None:
            continue
        x = load_audio(str(p))
        w = to_windows(x, max_windows=max_windows_per_file, rng=rng)
        w = w[np.sqrt((w ** 2).mean(1)) > 1e-4]
        cat = "aircraft" if int(r.class_) == 1 else "environment"
        st.add(w, source="aerosonicdb", category=f"aerosonicdb:{cat}", label=0, group=f"aero:{s}",
               split=s2split[s], hard_negative=cat == "aircraft", file=r.filename)


# ------------------------------------------------------------------------------------------- Svanström
def add_svanstrom(st, rng):
    wavs = sorted((RAW / "svanstrom").rglob("*.wav"))
    for p in tqdm(wavs, desc="svanstrom"):
        cls = re.match(r"^([A-Za-z]+)", p.stem).group(1).lower()
        if cls not in {"drone", "helicopter", "background"}:
            continue
        w = to_windows(load_audio(str(p)))
        w = w[np.sqrt((w ** 2).mean(1)) > 1e-4]
        st.add(w, source="svanstrom", category=f"svanstrom:{cls}", label=int(cls == "drone"),
               group=f"svan:{p.stem}", split="external", hard_negative=cls in HARD_NEG["svanstrom"], file=p.name)


SPLIT_ORDER = {"train": 0, "val": 1, "test": 2, "external": 3}


def enforce_group_exclusivity(man):
    """Rule 7: a recording group may only live in one split; keep its earliest-split copy (in place on man.keep)."""
    rank = man.split.map(SPLIT_ORDER)
    first = rank.groupby(man.group).transform("min")
    drop = rank != first
    report = {"groups_spanning_splits": int(man[drop].group.nunique()), "windows_dropped": int(drop.sum()),
              "groups": sorted(man[drop].group.unique().tolist())}
    man.loc[drop, "keep"] = False
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dads-block", type=int, default=1000)
    ap.add_argument("--dads-guard", type=int, default=25)
    ap.add_argument("--dads-max-clips-per-label", type=int, default=40_000)
    ap.add_argument("--dads-max-windows-per-clip", type=int, default=3)
    ap.add_argument("--aero-max-windows-per-file", type=int, default=20)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    st = Store()
    add_esc50(st, rng)
    add_us8k(st, rng)
    add_aerosonicdb(st, rng, args.aero_max_windows_per_file)
    add_svanstrom(st, rng)
    add_dads(st, rng, args.dads_block, args.dads_guard, args.dads_max_clips_per_label, args.dads_max_windows_per_clip)
    man = st.close()

    # Rule 6: exact duplicates across splits -> drop the val/test/external copies
    hsplits = man.groupby("hash").split.agg(lambda s: set(s))
    cross = hsplits[hsplits.map(len) > 1].index
    drop = man.hash.isin(cross) & (man.split != "train")
    dup_report = {"duplicate_hashes_spanning_splits": int(len(cross)),
                  "windows_dropped_from_eval": int(drop.sum()),
                  "by_category": man[drop].category.value_counts().to_dict()}
    man["keep"] = ~drop
    group_report = enforce_group_exclusivity(man)
    man.to_parquet(PROC / "manifest_full.parquet")

    # manifest.parquet indexes the rows of windows.bin; dropped windows stay in the file but are flagged
    man.index.name = "row"
    man.to_parquet(PROC / "manifest.parquet")
    summary = {
        "args": vars(args), "total_windows": int(len(man)), "kept_windows": int(man.keep.sum()),
        "duplicates": dup_report, "group_exclusivity": group_report,
        "per_split": man[man.keep].groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(),
        "per_category_split": man[man.keep].groupby(["category", "split"]).size().unstack(fill_value=0).to_dict(),
        "groups_per_split": man[man.keep].groupby("split").group.nunique().to_dict(),
        "group_overlap_between_splits_after_cleaning": int(man[man.keep].groupby("group").split.nunique().gt(1).sum()),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "data_summary.json").write_text(json.dumps(summary, indent=2, default=int))
    print(json.dumps({k: summary[k] for k in ["total_windows", "kept_windows", "duplicates", "per_split",
                                              "group_overlap_between_splits_after_cleaning"]}, indent=2, default=int))


if __name__ == "__main__":
    main()
