"""Stream the DDL real-recording archive (Zenodo 6459183, 12.6 GB zip) into compact shards.

Instead of downloading the whole zip and extracting it (~25 GB of disk), this script reads the zip's central
directory over HTTP, then fetches contiguous byte ranges in parallel, decodes each 100 ms 8-channel 96 kHz clip,
resamples it to 48 kHz and stores it as float16 (all 8 channels and their relative levels/timing preserved).

Output: data/raw/ddl/listing.json, data/raw/ddl/shards/shard_XXXXX.npz  (names, audio[n, 8, 4800] float16)
Resumable: finished shards are skipped.   Usage: python localization/scripts/download_ddl.py [--workers 8]
"""
import argparse
import io
import json
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_zip import HttpRangeFile  # noqa: E402

URL = "https://zenodo.org/api/records/6459183/files/MLSP_2022_Real_Data.zip/content"
OUT = ROOT / "data" / "raw" / "ddl"
SR_OUT = 48_000
CHUNK_BYTES = 48 << 20
UA = {"User-Agent": "acousticwatch/1.0"}


def get_range(start, end):
    for attempt in range(10):
        try:
            req = urllib.request.Request(URL, headers={**UA, "Range": f"bytes={start}-{end - 1}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if len(data) == end - start:
                return data
        except Exception as e:  # Zenodo intermittently returns 504
            err = e
        time.sleep(min(10 * (attempt + 1), 60))
    raise RuntimeError(f"range {start}-{end} failed: {err}")


def listing():
    p = OUT / "listing.json"
    if p.exists():
        return json.loads(p.read_text())
    z = zipfile.ZipFile(HttpRangeFile(URL, block=8 << 20))
    items = [{"name": i.filename, "off": i.header_offset, "csize": i.compress_size, "ctype": i.compress_type}
             for i in z.infolist() if i.filename.lower().endswith(".wav")]
    items.sort(key=lambda d: d["off"])
    OUT.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items))
    return items


def decode(buf, base, it):
    o = it["off"] - base
    n, m = struct.unpack("<HH", buf[o + 26:o + 30])
    data = buf[o + 30 + n + m:o + 30 + n + m + it["csize"]]
    if it["ctype"] == zipfile.ZIP_DEFLATED:
        data = zlib.decompressobj(-15).decompress(data)
    x, fs = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)    # (T, C)
    x = resample_poly(x, SR_OUT, fs, axis=0) if fs != SR_OUT else x
    return x.T.astype(np.float16)                                      # (C, T)


def do_shard(k, items):
    path = OUT / "shards" / f"shard_{k:05d}.npz"
    if path.exists():
        return k, len(items), 0
    base = items[0]["off"]
    end = items[-1]["off"] + 30 + 512 + items[-1]["csize"] + 4096
    buf = get_range(base, end)
    names, auds, bad = [], [], 0
    for it in items:
        try:
            a = decode(buf, base, it)
        except Exception:
            bad += 1
            continue
        names.append(Path(it["name"]).name); auds.append(a)
    L = max(a.shape[1] for a in auds)
    A = np.stack([np.pad(a, ((0, 0), (0, L - a.shape[1]))) for a in auds])
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, names=np.array(names), audio=A)
    tmp.rename(path)
    return k, len(names), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    items = listing()
    print(f"{len(items)} wav members", flush=True)
    shards, cur, start = [], [], None
    for it in items:
        if cur and it["off"] + it["csize"] - start > CHUNK_BYTES:
            shards.append(cur); cur = []
        if not cur:
            start = it["off"]
        cur.append(it)
    shards.append(cur)
    (OUT / "shards").mkdir(parents=True, exist_ok=True)
    done, t0 = 0, time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(do_shard, k, s) for k, s in enumerate(shards)]
        for f in as_completed(futs):
            k, n, bad = f.result()
            done += 1
            print(f"shard {k} ({n} clips, {bad} bad)  {done}/{len(shards)}  {time.time() - t0:.0f}s", flush=True)
    (OUT / ".done").touch()
    print("ddl download complete")


if __name__ == "__main__":
    main()
