# AcousticWatch: detection stage summary

*Stage 1 of 2: is a drone present? The localization stage, which estimates bearing, is summarised in
[localization/SUMMARY.md](localization/SUMMARY.md). Every decision taken during the build is logged in
[DECISIONS.md](DECISIONS.md).*

## The short version

- A small spectrogram CNN (1.2 M parameters) decides **"drone / not drone" every 0.5 s**.
- **On held-out recordings from the collections it was trained on, it is excellent.** At the standard setting it
  catches 99.6% of drone windows with 0.86% false alarms; at the strict setting, 97.5% with 0.07% false alarms.
  Traffic engines, engine sounds and helicopters never triggered it, and real aircraft overflights only 0.9%
  of the time.
- **Against simple baselines:** in-domain it clearly beats logistic regression (97.5% vs 79% recall at strict
  false-alarm settings). On the independent dataset it is no better than two hand-written rules that use no
  training at all.
- **On a completely independent drone dataset, it largely fails.** Different drones, microphones and distances
  bring recall down to about 15% (AUC 0.79). **This is the most important result.** Public drone audio is
  dominated by close-range recordings of a few small quadcopters, and a detector trained on it does not yet
  generalise to real early-warning conditions.
- **Range proxy.** Under a stated assumption (+20 dB above the noise at 10 m), detection holds at 90% recall
  out to roughly **65–105 m** and falls to 50% at roughly **115–180 m**. That is the in-domain best case.
- **It does not mainly rely on the rotor harmonic comb.** It relies on broadband 2.7–7.7 kHz motor and
  propeller whine. That band fades fastest with distance and is shared by chainsaws and drills, which explains
  both the poor cross-dataset result and the worst false alarms.

## Datasets used

| Dataset | Role | What it contributes |
|---|---|---|
| **DADS** (HF `geronimobasso/drone-audio-detection-samples`, 180,320 clips, MIT) | positives + negatives | Largest public drone-audio set; aggregates several earlier sets (incl. Al-Emadi's multi-class Bebop/Membo set, DREGON, Salford). Mostly small multirotors at close range. |
| **ESC-50** | negatives | Hard negatives: airplane, engine, helicopter, chainsaw, hand saw, train, vacuum cleaner. |
| **UrbanSound8K** | negatives | Hard negatives: engine idling (traffic), drilling, jackhammer, air conditioner. |
| **AeroSonicDB (YPAD-0523)** | negatives | 619 real **manned-aircraft overflights** (propeller and jet) plus outdoor ambience. |
| **Svanström et al. drone dataset** | **external test only** | 90 × 10 s clips (drone / helicopter / background) from different drones, microphones and sites. Never used for training or for any choice. |

After cutting audio into 0.5 s windows: 197,188 windows (train 143,498 · validation 25,344 · test 26,546 · external 1,800).

**Gaps.**
- **No public fixed-wing or engine-driven UAV audio could be found.** Every available drone recording is a
  multirotor. The detector has never heard a petrol-engine or fixed-wing drone, and should be assumed unable to
  detect one until such data is added.
- **Neither ESC-50 nor UrbanSound8K has a motorcycle class.** The closest stand-ins are ESC-50 "engine" and
  UrbanSound8K "engine idling".

## Metrics (model v1, the final model)

Thresholds are chosen on the validation split only:
- **"standard":** the highest-recall threshold that keeps validation false alarms ≤ 1%.
- **"strict":** the same, with validation false alarms ≤ 0.1%.

Each decision covers one 0.5 s window. The 95% intervals come from resampling whole recordings, not windows.

| Test set | Operating point | Accuracy | Recall (drones caught) | False-positive rate |
|---|---|---|---|---|
| Held-out test (same collections, unseen recordings) | standard | 99.3% | **99.6%** (98.8–100) | **0.86%** (0.50–1.18) |
| | strict | 99.2% | **97.5%** (91.8–100) | **0.07%** (0.02–0.14) |
| **External (Svanström)** | standard | 70.7% | **14.7%** (7.7–23.4) | 1.3% (0.4–2.6) |
| | strict | 67.6% | 2.8% | 0.0% |

ROC AUC is 0.9998 on the held-out test and **0.79 on the external set**
([results/detect_v1/roc.png](results/detect_v1/roc.png)). Averaging each clip's windows into one decision
gives 99.8% recall and 1.3% false alarms on the test split, but only 17% recall on the external clips.

## Against simple baselines

Does the CNN earn its complexity? Three baselines were run on the same splits ([scripts/baselines.py](scripts/baselines.py),
[results/baselines/](results/baselines/)), with thresholds chosen on validation in the same way:
- **Logistic regression** on the same log-mel spectrogram, summarised as each band's mean and spread.
- **A harmonicity rule** (no learning): how clear the harmonic stack is.
- **A high-band energy rule** (no learning): the share of energy in 2.7–7.7 kHz, the band the CNN relies on.

AUC intervals are 95% intervals from resampling whole recordings.

| Method | Test AUC | Test recall @ standard / strict | External AUC | External recall @ standard |
|---|---|---|---|---|
| **CNN (v1)** | **0.9998** | **99.6% / 97.5%** | 0.79 (0.72–0.85) | 14.7% |
| Logistic regression, log-mel stats | 0.977 | 94.3% / 79.0% | 0.42 (0.33–0.50) | 0% |
| Harmonicity rule (no learning) | 0.44 | 0.2% / 0% | 0.82 (0.77–0.86) | 0% |
| 2.7–7.7 kHz energy rule (no learning) | 0.49 | 0% / 0% | **0.84** (0.77–0.90) | 0% |

How to read this:
- **In-domain, the CNN clearly earns its keep.** At strict false-alarm settings it catches 97.5% of drones;
  the best simple model catches 79%. Its hard-negative false alarms (1.0%) are comparable to logistic
  regression's (0.5%).
- **Out of domain, nothing learned generalises.** Logistic regression falls *below* chance (0.42). The CNN's
  external AUC is statistically indistinguishable from two rules that use no training at all; their
  intervals overlap.
- **No method is usable on the external set at a threshold chosen on in-domain validation data.** The
  learning-free rules rank external drones above external non-drones reasonably well, but their in-domain
  thresholds catch nothing there. A deployed sensor would need calibration data from its own site and microphone.
- **Caveat on the rules' external AUC.** The external negatives are helicopters and wind-dominated background,
  both heavy in low frequencies, so "more high-frequency energy" may partly reflect that set's make-up rather
  than real drone recognition. With only 90 clips, it is a small test.

## False positives on engine and aircraft sounds

At the standard threshold ([results/detect_v1/fpr_by_category.png](results/detect_v1/fpr_by_category.png)):

| Hard negative | Test windows | False alarms (standard) | False alarms (strict) |
|---|---|---|---|
| UrbanSound8K **engine idling (traffic)** | 731 | **0.0%** | 0.0% |
| ESC-50 engine / helicopter / train | 80 each | 0.0% | 0.0% |
| **AeroSonicDB real aircraft overflights** | 1,100 | **0.9%** | 0.0% |
| ESC-50 airplane | 80 | 2.5% | 0.0% |
| Jackhammer / drilling / air conditioner | 677 / 719 / 800 | 0.6% / 2.2% / 0.0% | 0.0% |
| **ESC-50 chainsaw** | 79 | **12.7%**, the worst class | 3.8% |
| Svanström helicopter (external) | 600 | 2.3% | 0.0% |

- Hard negatives as a group: 0.98% false alarms, against 0.82% for ordinary sounds such as washing machines and
  rain. Upweighting them 3× in training kept engine-like sounds from being *more* confusable than everyday ones.
- Motor-vehicle engines are not the problem. **High-revving small engines and power tools are**: chainsaws and
  drills, which share the drone's high-pitched motor whine.
- The loudest false alarms are listed for listening in
  [results/detect_v1/false_positives_engine.csv](results/detect_v1/false_positives_engine.csv).
- Caveat: the aircraft result rests on only 3 recording days, so its interval is wide.

## Signal attenuation as a proxy for range

The test is a simulation ([results/detect_v1/range_curve_distance.png](results/detect_v1/range_curve_distance.png),
[range_curve_snr.png](results/detect_v1/range_curve_snr.png)):
1. Take held-out drone windows and place them over a fixed noise floor. Three floors are used: pink noise,
   real outdoor ambience, and traffic-engine noise.
2. Start the drone at +20 dB above the floor (assumed to be the level at 10 m).
3. Weaken the drone step by step. Spherical spreading costs 6 dB per doubling of distance, and ISO 9613-1 air
   absorption removes high frequencies faster.

| Noise floor | Recall stays ≥ 90% until | Recall falls to 50% at | Proxy distance at 90% / 50% |
|---|---|---|---|
| Pink noise | SNR ≈ 0 dB | SNR ≈ −3 dB | ≈ 105 m / 140 m |
| Outdoor ambience | SNR ≈ +2 dB | SNR ≈ −2 dB | ≈ 80 m / 115 m |
| Traffic engine noise | SNR ≈ +3 dB | SNR ≈ −5 dB | ≈ 65 m / 180 m |

How to read this:
- **The shape is the robust finding.** Detection is near-perfect down to about 0 dB SNR, the point where the
  drone is as loud as the background, and then collapses within 3–8 dB. There is little graceful degradation.
- **The metres are illustrative.** They depend entirely on the assumed starting level; a louder drone or a
  quieter site scales them up.
- With the noise floor alone, false alarms stayed at 0–0.1% for pink noise and ambience, but reached **2.5% for
  traffic noise** at the standard threshold (0% at strict).
- External drones never exceed about 22% recall at any level, so this curve is an in-domain best case.

## Which spectral features drive detections

Details are in [results/detect_v1/explain_summary.json](results/detect_v1/explain_summary.json) and the figures beside it.

1. **Harmonicity does not predict detection.** Estimated rotor fundamentals are 125–315 Hz (median 250 Hz).
   But missed drones are *more* harmonic than detected ones (median cepstral peak prominence 0.111 against
   0.080), and drone probability falls slightly as harmonic clarity rises (Spearman ρ = −0.38).
2. **Causal notch test** ([harmonic_notch_test.png](results/detect_v1/harmonic_notch_test.png)), on 1,000
   clearly harmonic, correctly detected windows:
   - Removing the harmonic comb (k·f0, ±4%) strips **68% of the energy**, but recall only drops from 100% to 97.7%.
   - Removing equal-width bands between the harmonics leaves recall at 99.3%.
   - Keeping **only** the harmonics drops recall to **10.8%**.

   So the harmonic comb is neither necessary nor sufficient for this model.
3. **Band occlusion** ([band_importance.png](results/detect_v1/band_importance.png)). Hiding any single band
   below 2.3 kHz barely matters. The drops come from **2.7–7.7 kHz**: broadband motor, ESC and propeller noise.
   The 59 false alarms on hard negatives are driven by the *same* band, even more strongly.

**Implication.** The detector has learned "high-pitched motor whine". That is exactly what:
- air absorption removes first at range;
- changes most between microphones (hence the external failure);
- chainsaws and drills also produce (hence the worst false alarms).

The lower rotor harmonics carry further and are more drone-specific, so a range-oriented detector should be
pushed towards them (see next steps).

## How leakage was avoided

1. **DADS: split by recording, verified with fingerprints.**
   - DADS renames every clip `drone-N.wav` and ships no recording IDs.
   - A check ([results/dads_adjacency.png](results/dads_adjacency.png)) showed that consecutive IDs are
     near-duplicate chunks of one recording: median similarity 0.98 against 0.72 for random pairs, with runs
     of up to about 700 IDs. A random split would therefore leak heavily.
   - Every clip was fingerprinted. DADS is cut into contiguous blocks of ≥ 1,000 IDs that are **closed only at a
     break**, where consecutive clips stop being near-duplicates. Whole blocks go to one split, with an extra
     25-ID guard margin.
   - The audit ([results/dads_split_audit.json](results/dads_split_audit.json)) finds **0 near-duplicate pairs
     across splits**.
2. **The other datasets use their own recording groupings.** ESC-50 and UrbanSound8K use their official folds,
   which are grouped by original recording. AeroSonicDB is split by recording day. Every 0.5 s window inherits
   its recording's split.
3. **Cross-fold duplicates were removed.** Three recordings appear in two official ESC-50 or UrbanSound8K folds;
   their later-split copies were dropped. Exact-duplicate audio was removed across splits as well.
4. **Normalisation statistics, augmentation backgrounds and decision thresholds use training and validation
   data only.**
5. **Two dataset shortcuts were found and removed.**
   - DADS drone clips are exactly 0.5 s while negatives are about 5 s, so **0.5 s windows** are used with no
     tiling or padding. Otherwise only the drones would have been tiled.
   - DADS drone clips carry a **5× larger DC offset** than everything else, so DC is removed per window.
6. **The external set was never used for any choice.** One improvement attempt (v2) did worse there, and no
   further variants were tried against it.

An earlier version used fixed 1,000-ID blocks with a 100-ID guard, before the 700-ID runs were found. Its results
are kept in [results/archive_fixed_block_split/](results/archive_fixed_block_split/). They are almost identical
(test AUC 0.9997 against 0.9998 now), so residual DADS leakage was not inflating the numbers. The strongest
evidence that the in-domain numbers are optimistic is the external test, not leakage.

## Errors, restarts and thermal behaviour

- **GPU.** The power limit was set to 220 W (it needed one elevated UAC prompt and resets on reboot; every run
  re-checks it). Peak GPU temperature across all detection runs was **53 °C**. There were **no thermal pauses
  and no out-of-memory events**. Training: 20 epochs × 80k balanced samples, batch 128, fp16, about 25 s per
  epoch, checkpointed every epoch. Events are logged in `logs/gpu_events.jsonl`.
- **The first training run crashed** before epoch 0: Windows DataLoader workers tried to pickle the
  multi-GB memory-mapped data. Fixed by opening the file lazily in each worker.
- **The second run was stopped on purpose** after the DC-offset shortcut was found, then restarted with the fix.
- **v2 was rejected.** It added distance and microphone augmentation (air absorption, spectral tilt, band
  limits) and was meant to fix the external result. It made things worse: external AUC 0.82 → 0.52, in-domain
  AUC 0.9997 → 0.9949, hard-negative false alarms 0.26% → 2.6%. Those are old-split numbers, archived.
  The pre-declared selection rule could not tell the models apart (validation AUC saturated), so v1 was kept
  and the deviation is disclosed in DECISIONS.md.
- **v1 was retrained once more** after the DADS split was tightened. The reported numbers come from that model.
- **A range-proxy bug** (drone simulated about 30× louder than in training) gave nonsense in the first run. It
  was fixed and re-run. Lesson: **the model is sensitive to absolute input level.**
- **Download trouble.** Zenodo was slow (about 0.2–0.5 MB/s per connection, with intermittent 504 errors). Two
  download jobs collided on one file. Fixed with parallel ranged downloads.

## Next steps

1. **Get engine-driven and fixed-wing UAV recordings.** This is the biggest gap for early warning. Ideally
   record them at 50–500 m with the same microphones the sensor will use.
2. **Push the model towards cues that survive distance.** Options:
   - a level-invariant front-end (PCEN or per-window gain normalisation);
   - harmonic-aware inputs (cepstrum, harmonic-comb energy) over 50–1,500 Hz;
   - training that penalises reliance on the 3–8 kHz band, for example by randomly dropping that band.
3. **Make cross-dataset testing the headline metric.** Hold out whole drone models or sites, and add more
   external sets (e.g. DroneAudioset, the 32-class multiclass set) as they become available.
4. **Collect hard negatives deliberately:** chainsaws, motorcycles, leaf blowers, two-stroke garden tools,
   drills. These are the current weak spot, and none of the datasets has motorcycles.
5. **Use temporal smoothing and track logic** (e.g. 3 of 4 consecutive windows) to cut isolated false alarms,
   and pair alerts with the stage-2 bearing estimate.

## Where things are

- `results/`: data summary, DADS adjacency check and split audit, training histories.
  - `results/detect_v1/`: final model metrics, per-category table, ROC, false-alarm chart, range curves,
    spectral analysis.
  - `results/archive_fixed_block_split/`: earlier split, including the rejected v2.
- `scripts/`: `download_data.py` → `check_dads_adjacency.py --all` → `prepare_data.py` → `train.py` →
  `evaluate.py` → `robustness.py` → `explain.py` → `baselines.py`
- `src/acousticwatch/`: shared code (features, model, data store, GPU guard, training loop, plot style).
