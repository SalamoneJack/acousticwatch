"""Download all public datasets used by AcousticWatch into data/raw/.

Each dataset goes into its own directory. Archives are extracted and then
deleted to save disk space. Re-running skips anything already present.

Usage:  python scripts/download_data.py [--only dads esc50 ...]
"""
import argparse
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SOURCES = {
    # Drone positives + generic negatives (label 1 = drone, 0 = no drone). MIT licence.
    "dads": {"hf": "geronimobasso/drone-audio-detection-samples"},
    # Environmental sounds; hard negatives: airplane, engine, helicopter, chainsaw, train. CC BY-NC.
    "esc50": {"url": "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"},
    # Urban sounds; hard negatives: engine_idling, drilling, jackhammer, air_conditioner. CC BY-NC.
    # (HF mirror of the official Zenodo release; keeps the fold + fsID metadata needed for leakage-safe splits)
    "urbansound8k": {"hf": "danavery/urbansound8K"},
    # Svanstrom et al. multi-sensor drone detection dataset (audio: DRONE / HELICOPTER / BACKGROUND).
    # Used as an external, never-trained-on test domain.
    "svanstrom": {"url": "https://zenodo.org/api/records/5500576/files/DroneDetectionThesis/Drone-detection-dataset-v1.0.0.zip/content",
                  "fname": "svanstrom.zip"},
    # AeroSonicDB: manned aircraft overflights (hard negatives) + labelled environment audio. CC BY-NC 4.0.
    "aerosonicdb": {"files": {
        "audio.zip": "https://zenodo.org/api/records/8371595/files/audio.zip/content",
        "sample_meta.csv": "https://zenodo.org/api/records/8371595/files/sample_meta.csv/content",
        "aircraft_meta.csv": "https://zenodo.org/api/records/8371595/files/aircraft_meta.csv/content",
        "README.md": "https://zenodo.org/api/records/8371595/files/README.md/content",
    }},
}


def fetch(url, dest, connections=8):
    """Download with parallel HTTP range requests (Zenodo throttles single connections to ~0.4 MB/s)."""
    if dest.exists():
        print(f"  have {dest.name}")
        return
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    ua = {"User-Agent": "acousticwatch/1.0"}
    with urllib.request.urlopen(urllib.request.Request(url, headers={**ua, "Range": "bytes=0-0"})) as r:
        ranged = r.status == 206
        size = int(r.headers.get("Content-Range", "/0").split("/")[-1] or 0)
    if not ranged or size < (64 << 20):
        with urllib.request.urlopen(urllib.request.Request(url, headers=ua)) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 22)
    else:
        from concurrent.futures import ThreadPoolExecutor
        chunk = 32 << 20
        with open(tmp, "wb") as f:
            f.truncate(size)

        def part(start):
            end = min(start + chunk, size) - 1
            for attempt in range(8):
                try:
                    req = urllib.request.Request(url, headers={**ua, "Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        data = r.read()
                    assert len(data) == end - start + 1
                    with open(tmp, "r+b") as f:
                        f.seek(start); f.write(data)
                    return
                except Exception:
                    import time; time.sleep(5 * (attempt + 1))
            raise RuntimeError(f"range {start}-{end} failed")
        with ThreadPoolExecutor(connections) as ex:
            list(ex.map(part, range(0, size, chunk)))
    tmp.rename(dest)


def extract(archive, out):
    print(f"  extracting {archive.name}")
    if archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive) as t:
            t.extractall(out, filter="data") if sys.version_info >= (3, 12) else t.extractall(out)
    else:
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    archive.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(SOURCES))
    args = ap.parse_args()
    for name in args.only:
        spec, out = SOURCES[name], RAW / name
        done = out / ".done"
        if done.exists():
            print(f"[{name}] already done")
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"[{name}]")
        if "hf" in spec:
            from huggingface_hub import snapshot_download
            snapshot_download(spec["hf"], repo_type="dataset", local_dir=out, max_workers=4)
        elif "url" in spec:
            arc = out / spec.get("fname", spec["url"].rsplit("/", 1)[-1])
            fetch(spec["url"], arc)
            extract(arc, out)
        else:
            for fname, url in spec["files"].items():
                fetch(url, out / fname)
                if fname.endswith(".zip"):
                    extract(out / fname, out)
        done.touch()
    print("all downloads complete")


if __name__ == "__main__":
    main()
