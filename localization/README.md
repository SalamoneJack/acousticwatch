# localization/: stage 2, which direction is the drone?

Estimates the **bearing (direction of arrival)** of a drone, plus a rough **range**, from 100 ms of
8-channel microphone-array audio. It is stage 2 of AcousticWatch: stage 1 (`../scripts`, detection) decides
whether a drone is present, and this stage says where it is. It is a passive sensing and direction-finding
module only.

Results and a plain-language write-up are in [SUMMARY.md](SUMMARY.md). In brief: a 2–8° bearing error when the
drone is clearly audible, and a 15.5° median (CNN) or 10° (classical SRP-PHAT) over all unseen test flights. The
main failure mode is locking onto a competing noise source when the drone is faint. Figures and metrics are in [results/](results/).

## Data

[DDL: Drone Detection and Localization](https://zenodo.org/records/6459183) (Safavi et al., MLSP 2025;
CC-BY 4.0). It holds real field recordings of a DJI Mini 2 and a DJI Phantom 4 Pro flying autopilot orbits
around an 8-microphone array. The recordings were made at 96 kHz and cut into 100 ms clips, with bearing,
range and altitude labelled from GPS flight logs. Only the real-recording archive (12.6 GB) is used; the
synthetic archive is not.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/download_ddl.py` | Streams the 12.6 GB zip with parallel HTTP range requests and stores each clip as 8-channel 48 kHz float16 shards (about 11 GB, resumable). No full zip on disk. |
| 2 | `python -m loc.data` (run from `localization/`) | Parses labels from file names, drops take-off and landing clips (range < 10 m or altitude < 20 m), and assigns **trial-level** splits. |
| 3 | `scripts/calibrate_geometry.py` | Fits the 3-D microphone positions from *training* GCC-PHAT delays and GPS directions (not published by the dataset), then checks them with a classical SRP-PHAT beamformer. |
| 4 | `scripts/train_loc.py --features both\|timing\|spectral` | Trains the CNN/CRNN. Run all three for the feature ablation. |
| 4b | `scripts/select_checkpoint.py` | Picks each model's epoch by validation *median* bearing error. |
| 5 | `scripts/eval_loc.py` | Test metrics, range-bin breakdown, SRP-PHAT baseline, permutation importance. |
| 6 | `scripts/noise_loc.py` | Error versus SNR for three noise fields, and error versus true distance. |

Run every script from the repository root with the shared `.venv`:

```bash
.venv/Scripts/python localization/scripts/download_ddl.py
cd localization && ../.venv/Scripts/python -m loc.data && cd ..
.venv/Scripts/python localization/scripts/calibrate_geometry.py
for f in both timing spectral; do .venv/Scripts/python localization/scripts/train_loc.py --features $f --epochs 25; done
.venv/Scripts/python localization/scripts/select_checkpoint.py
.venv/Scripts/python localization/scripts/eval_loc.py
.venv/Scripts/python localization/scripts/noise_loc.py
```

## Inputs and model

- **Timing features:** GCC-PHAT for all 28 microphone pairs, band-limited to 100–8000 Hz, with 4× lag
  interpolation over ±1.33 ms. Shape 28 × 513.
- **Spectral features:** per-channel log-mel spectrograms plus inter-channel level differences. Shape 16 × 64 × 21.
- **Model:** a 1-D CNN timing branch and a 2-D CNN + bidirectional GRU spectral branch, followed by an MLP.
  It outputs a unit vector (cos, sin) for bearing and log-range. The loss is `1 − cos(error) + 0.5·|Δ log range|`.

## Shared code reused from stage 1

- `src/acousticwatch/gpu_guard.py`: power-limit check, thermal pause (above 80 °C until below 70 °C), event log.
- `src/acousticwatch/trainloop.py`: the safety-wrapped training loop both stages use (fp16, checkpoint every
  epoch, OOM batch halving).
- `src/acousticwatch/data.py` and `plotstyle.py`: detection-store access (its outdoor-ambience and engine
  windows are the noise sources here) and the shared figure style.

## Leakage

Each autopilot trial is one continuous flight recording, so its 100 ms clips are near-duplicates of their
neighbours. **Whole trials are assigned to train, val or test**; no trial spans two splits. Trials are
stratified by drone model and median range. Geometry calibration and feature normalisation use training
trials only. Details are in [SUMMARY.md](SUMMARY.md).
