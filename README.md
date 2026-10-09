# AcousticWatch

A reproducible, passive **acoustic drone-sensing** pipeline built from public datasets. It works in two stages:

| Stage | Question | Input | Output | Write-up |
|---|---|---|---|---|
| **1. Detection** | *Is a drone present?* | 0.5 s of single-microphone audio | drone probability, every 0.5 s | [SUMMARY.md](SUMMARY.md) |
| **2. Localization** | *Which direction is it?* | 100 ms of 8-channel microphone-array audio | bearing (and rough range) | [localization/SUMMARY.md](localization/SUMMARY.md) |

In a deployed sensor, stage 1 runs continuously on one channel and raises an alert. Stage 2 then runs on the
array's channels to attach a bearing to the alert. The two stages are trained and evaluated on different
datasets, because no public dataset has both large-scale detection negatives and multi-channel array audio.

## Headline results

**Detection** (final model v1; full write-up in [SUMMARY.md](SUMMARY.md)):
- **Held-out recordings:** 99.6% recall at 0.86% false alarms, or 97.5% recall at 0.07% at the strict setting.
  Traffic engines, engines and helicopters triggered 0% false alarms, real aircraft overflights 0.9%, and
  chainsaws 12.7% (the weak spot).
- **Independent external drone dataset: only about 15% recall.** The model does not yet generalise to unseen
  drones and microphones. This is the key limitation.
- **Against baselines:** in-domain it clearly beats logistic regression on log-mel statistics (test AUC 0.9998
  vs 0.977; recall 97.5% vs 79% at strict false-alarm settings). On the external set it is statistically tied
  with two learning-free rules (AUC 0.79 vs 0.82–0.84, overlapping intervals).
- **Range proxy:** 90% recall holds down to roughly 0 dB SNR (≈ 65–105 m under a +20 dB-at-10 m assumption).
- **What it listens for:** 2.7–7.7 kHz motor whine, not mainly the rotor-harmonic comb.

**Localization** (8-mic array, DDL dataset, unseen test flights; full write-up in [localization/SUMMARY.md](localization/SUMMARY.md)):
- **Clearly audible drone (20–60 m):** 2–8° bearing error, for both the CNN and a classical SRP-PHAT beamformer
  that uses an array geometry calibrated from training data.
- **All test flights:** median 15.5° (CNN) and 10° (SRP-PHAT). Mean range error 24 m.
- **Main failure:** a faint drone (the small Mini 2 at about 96 m, or a 149 m hover) loses out to a competing
  noise source, and the array points at that source. With a simulated directional engine interferer, the CNN
  is far more robust than the classical method (25° vs 57° at +20 dB SNR).
- **Against baselines:** the CNN roughly ties a calibrated classical SRP-PHAT beamformer. It is worse on clean
  signals (median 15.5° vs 10°), better on hard cases (mean 46° vs 53°) and much better with interference.
  Range error is 24 m, against 35 m for guessing the typical distance. Chance level is a 90° median error.
- **Feature importance:** the CNN relies mostly on spectral and level-difference cues (shuffling them adds
  11.7° of error, shuffling timing only 1.7°). The classical method gets its precision from timing. The two
  are complementary.

## Repository layout

```
scripts/                 stage 1 pipeline: download → leakage check → prepare → train → evaluate → robustness → explain
src/acousticwatch/       shared code used by BOTH stages
    gpu_guard.py         power-limit check, thermal pause (>80 °C until <70 °C), event log
    trainloop.py         safety-wrapped training loop (fp16, checkpoint every epoch, OOM batch halving)
    features.py          log-mel front-end, windowing, ISO 9613-1 air absorption
    data.py, infer.py, model.py, plotstyle.py
localization/            stage 2 module (own README, scripts, results, SUMMARY)
results/                 stage 1 metrics and figures (results/detect_v1 = final model)
DECISIONS.md             every decision made during the autonomous build, with reasons
logs/gpu_events.jsonl    power-limit checks, thermal pauses, OOM events
```

## Setup (Windows, RTX-class GPU, Python 3.10)

```bash
py -3.10 -m venv .venv
.venv/Scripts/python -m pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
.venv/Scripts/python -m pip install -r requirements.txt
nvidia-smi -pl 220            # needs an elevated prompt; resets on reboot (training re-checks and logs it)
```

## Reproduce stage 1 (detection)

```bash
.venv/Scripts/python scripts/download_data.py              # ~16 GB: DADS, ESC-50, UrbanSound8K, AeroSonicDB, Svanström
.venv/Scripts/python scripts/check_dads_adjacency.py       # evidence that DADS clip IDs are chunked recordings
.venv/Scripts/python scripts/check_dads_adjacency.py --all # fingerprints used for the break-aware split
.venv/Scripts/python scripts/prepare_data.py --dads-max-clips-per-label 50000 --dads-max-windows-per-clip 2
.venv/Scripts/python scripts/train.py --tag detect_v1
.venv/Scripts/python scripts/evaluate.py   --out results/detect_v1
.venv/Scripts/python scripts/robustness.py --out results/detect_v1
.venv/Scripts/python scripts/explain.py    --out results/detect_v1
.venv/Scripts/python scripts/baselines.py                 # logistic regression + two learning-free rules
```

Stage 2 commands are in [localization/README.md](localization/README.md). Run GPU jobs one at a time. On
Windows, a second job sharing the 8 GB card makes the driver page GPU memory into system RAM instead of raising
an out-of-memory error, and both jobs stall.

## GPU safety rules (both stages)

- Power limit 220 W, checked at the start of every run.
- Temperature is polled every epoch and between batches. Above 80 °C, training pauses until the GPU is below 70 °C.
- fp16 autocast and modest batches (128 for detection, 64 for localization).
- A checkpoint is written every epoch, and training can resume from it.
- On CUDA out-of-memory, the batch size is halved and the epoch restarts from the last checkpoint.

Peak temperature during the whole build was 55 °C, with no thermal pauses or OOM events (see `logs/gpu_events.jsonl`).

## Data licences

DADS (MIT); ESC-50, UrbanSound8K and AeroSonicDB (CC BY-NC); Svanström drone dataset (open, Zenodo 5500576);
DDL (CC BY 4.0). Datasets are downloaded by the scripts and are not redistributed in this repository.
