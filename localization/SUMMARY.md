# AcousticWatch: localization stage summary

*Stage 2 of 2: which direction is the drone? Stage 1 (detection) is summarised in [../SUMMARY.md](../SUMMARY.md).
All decisions are logged in [../DECISIONS.md](../DECISIONS.md).*

## The short version

- From **100 ms of 8-microphone audio**, the system estimates the drone's **bearing**, plus a rough **range**.
- **When the drone is clearly audible it works well.** Typical errors are **2–8°** for drones 20–60 m away, for
  both the neural network and a classical beamformer.
- **When the drone is faint, it locks onto other sounds.** For the small DJI Mini 2 at about 96 m, and for a
  hover at 149 m, the array often points at a competing noise source at the site instead of the drone.
- Over all held-out test flights:

  | Method | Median bearing error | Mean | Within 20° |
  |---|---|---|---|
  | CNN (timing + spectral) | **15.5°** | 46° | 54% |
  | Classical SRP-PHAT beamformer | **10.0°** | 53° | 53% |

  Mean range error is **24 m** (median 16 m).
- **Bearing accuracy is governed by how clearly the drone stands out from the background.** The neural network
  is markedly more robust than the classical method **when a competing engine noise comes from another direction**.
- **Most of the network's skill comes from spectral and level cues rather than timing.** Inter-channel timing is
  where the classical method gets its precision, so the two are complementary.

## Dataset used

**DDL: Drone Detection and Localization** (Safavi et al., MLSP 2025; [Zenodo 6459183](https://zenodo.org/records/6459183),
CC BY 4.0). The requested dataset was available, so no fallback was needed.

- **What it contains:** real field recordings of a DJI Mini 2 and a DJI Phantom 4 Pro flying autopilot orbits
  and hovers around an 8-channel microphone array. Recorded on 29 and 31 March 2021, at 96 kHz, cut into 100 ms
  clips, with bearing, range and altitude labels from the drones' GPS logs.
- **Used:** only the real-recording archive (12.6 GB). The synthetic (rendered) archive was skipped so that every
  test number reflects real acoustics.
- **Stored:** streamed into 8-channel 48 kHz shards with every channel preserved. 62,103 clips; 60,769 remain
  after removing take-off and landing clips (range < 10 m or altitude < 20 m), whose GPS bearings are unreliable.
- **No drone-free recordings exist in the real data,** so all noise for the robustness tests had to be added
  synthetically.

**Array geometry (calibrated, not published).** The paper describes the array only as "eight channels,
microphone heads at the four corners of the box, in pairs". So the 3-D microphone positions were **fitted from
training clips only**, using the measured inter-channel delays and the GPS directions.
- **Result** ([results/array_geometry.png](results/array_geometry.png)): a flat array, about 37 cm across,
  with four pairs of microphones 7–8 cm apart at the corners of a roughly 30 cm square. That matches the paper.
- **Fit quality:** the median residual is 1.6 samples (33 µs).
- **Check:** a classical beamformer using this geometry, with no learning, reaches a 6° median bearing error on
  validation.

## Bearing and range error on the test set

The test set has 13,253 clips from 6 complete flights that never appear in training: 3 Mini 2 orbits, a Phantom
orbit, a Phantom hover and a short Mini 2 hover.

| Method | Mean error | Median | Within 10° | Within 20° | Within 45° |
|---|---|---|---|---|---|
| CNN, timing + spectral | 46.0° | **15.5°** | 42% | 54% | 67% |
| CNN, spectral only | 47.1° | 17.7° | 39% | 52% | 66% |
| CNN, timing only (GCC-PHAT) | 56.1° | 21.5° | 39% | 49% | 59% |
| SRP-PHAT classical beamformer | 52.9° | **10.0°** | 50% | 53% | 61% |
| CNN, averaged over 0.5 s (5 clips) | 44.0° | 14.4° | | | |

**Range** (combined CNN):
- Mean error 24.3 m, median 15.6 m, 30% relative error.
- For comparison, always guessing the typical training range gives 35 m.
- Per flight: 4–5 m at 30–50 m (good), 20 m at 96 m, and 75 m for the 149 m hover, where half the clips are corrupted.

**By flight** ([results/per_trial.csv](results/per_trial.csv)). This is where the picture becomes clear:

| Test flight | Drone, distance | CNN median | SRP-PHAT median | Clips > 90° off (CNN) |
|---|---|---|---|---|
| 310321-T006 | Phantom 4, 50 m orbit | **4.1°** | 5° | 0% |
| 310321-T017 | Mini 2, 30 m orbit | **6.3°** | 4° | 0% |
| 310321-T033 | Mini 2, 60 m hover | **6.1°** | 9° | 0% |
| 290321-T004 | Mini 2, 48 m orbit | 17.8° | **5°** | 21% |
| 290321-T007 | Mini 2, 96 m orbit | 36.7° | 76° | 27% |
| 310321-T012 | Phantom 4, 149 m hover | 2.0° | 126° | 44% |

**What goes wrong.** When the drone is faint, the array points at another sound source instead.
- **Mini 2 at 96 m:** the classical beamformer answers about 240° no matter where the drone is. A fixed noise
  source at that bearing dominates the quiet Mini 2 at that distance.
- **Phantom hover at 149 m:** about half the clips point almost exactly the opposite way (about 13° instead
  of 193°), for every method. A source on the far side of the array periodically drowns out the drone.
- **Confirmed by signal clarity.** Sorting the test clips into fifths by GCC-PHAT coherence, the CNN's median
  error goes from 5.5° (clearest fifth) to 78° (least clear), and the beamformer's from 4° to 107°.

**Comparison with the paper.** The paper reports about 11–12° mean bearing error and about 5 m range error.
Its train, validation and test sets appear to be drawn from the same flights (every distance bin is split
roughly 70/10/20). Neighbouring 100 ms clips of one flight are near-identical, so that protocol largely tests
memorisation of each flight. Our numbers come from *unseen flights*, so they are not directly comparable, and
should be closer to what a deployed sensor would see.

## How accuracy falls with noise and distance

**Real distance (no simulation)** ([results/error_vs_true_range.png](results/error_vs_true_range.png)):

| Distance | CNN median error | SRP-PHAT median error |
|---|---|---|
| 10–35 m | 5.5° | 2° |
| 35–65 m | 8° | 5° |
| ~96 m (Mini 2) | 28–82° | 64–96° |
| 149 m (Phantom hover) | 2° | 126° (half the clips flipped) |

The small Mini 2 becomes unreliable beyond about 65 m. The louder Phantom 4 is still found at 149 m when it
is not masked.

**Simulated weakening** ([results/error_vs_snr.png](results/error_vs_snr.png)). Noise was added to 3,000 test
clips at a controlled signal-to-noise ratio (SNR). Three kinds of noise were used:
- independent noise on each microphone;
- diffuse real outdoor ambience, incoherent across microphones;
- one traffic-engine recording arriving from a random direction, rendered with the calibrated array geometry.

| Noise type | CNN timing + spectral | CNN timing only | SRP-PHAT |
|---|---|---|---|
| Clean (same clips) | 15.6° | 22.6° | 12.0° |
| White, +10 dB | 32° | 30° | 40° |
| White, 0 dB | 32° | 35° | 60° |
| White, −10 dB | 59° | 72° | 79° |
| Diffuse ambience, +10 dB | 40° | 32° | 36° |
| Diffuse ambience, 0 dB | 74° | 50° | 54° |
| **Directional engine, +20 dB** | **25°** | 48° | 57° |
| **Directional engine, +10 dB** | **44°** | 71° | 79° |
| Directional engine, 0 dB | 76° | 100° | 94° |

All values are median bearing errors; chance level is 90°. How to read this:
- **Even small amounts of added noise hurt** (about 15° → 20° at +30 dB). The SNR is measured against each clip's
  total power, which in field recordings is mostly low-frequency wind and ambience. In the bands that carry the
  drone, the true SNR is much lower than the headline number.
- **Uncorrelated noise:** the CNN holds about 30° down to 0 dB, while the classical beamformer degrades steadily.
- **A competing directional source is the worst case,** and it is exactly the failure seen in the real test
  flights. The classical beamformer is pulled towards the interferer even at +20 dB. The combined CNN resists
  much better (25° vs 57°), because it can also use *what* each source sounds like, not just its delay.
- **Below 0 dB SNR every method is near chance.** That marks the practical range limit: the drone must be at
  least about as loud as the background in the array's band.

## Which input features matter

([results/feature_importance.png](results/feature_importance.png))

- **Ablation, three separately trained models.** Timing + spectral: 15.5° median. Spectral only: 17.7°.
  Timing only: 21.5°. Combining the two families helps, but spectral cues alone already beat timing alone.
- **Permutation importance on the combined model.** Breaking the link between a clip and its spectral input
  raises the mean error by **11.7°**. Doing the same to its timing (GCC-PHAT) input raises it by only **1.7°**.
  The combined network leans mostly on spectral cues: per-channel spectra and **inter-channel level differences**.
  The array box shadows the microphones facing away from the drone, so level differences carry direction.
- **The classical beamformer** uses *only* timing, together with the calibrated geometry, and achieves the best
  median error (10°). So timing information is precise. The CNN has not learned to exploit it fully from just
  22 training flights, because it must discover the array geometry implicitly.

**Practical conclusion.** The two approaches are complementary:
- **Timing with known geometry** gives precise bearings on a clear signal.
- **Learned spectral cues** give robustness to competing sources.

A hybrid is the obvious next step: feed the beamformer's steered-response map into the CNN.

## How leakage was avoided

- **The unit of splitting is a whole flight** (one autopilot trial folder: one continuous recording). Every
  100 ms clip of a flight goes to the same split, so near-identical neighbouring clips can never sit on both
  sides. The split check confirms no flight is shared.
- **Flights were stratified by drone model and distance:** sorted by median range within each model, then dealt
  out in turn to train, validation and test. This gives 22 training flights (31,799 clips), 7 validation
  flights (15,717) and 6 test flights (13,253), all covering similar distances.
- **Geometry calibration** (and so the SRP-PHAT baseline) and feature normalisation use **training flights only**.
- **Noise used in the robustness tests comes from the detection stage's test split** (outdoor ambience and
  traffic recordings), so no training audio is reused.
- **Model selection used validation flights only.** The test flights were evaluated once, after every choice
  was fixed.
- **Residual caveat.** Both days used the same array and the same two drones, and validation and test flights
  share recording sessions with training flights. The test therefore measures generalisation to *new flights*,
  not to a new site, array or drone type.

## Errors, restarts and thermal behaviour

- **GPU.** The power limit was 220 W, verified at the start of every run. Peak temperature was **55 °C**,
  with **no thermal pauses and no out-of-memory events**. Each model trained for 25 epochs at batch 64 in fp16
  (about 23 s per epoch), checkpointed every epoch.
- **Training stall.** Running a second GPU job (a geometry re-plot) alongside training filled the 8 GB card.
  Windows then paged GPU memory into system RAM instead of raising an out-of-memory error, so the automatic
  batch-halving never triggered and training froze for about 11 minutes. Both jobs were stopped and training
  **resumed from its epoch-2 checkpoint**. Rule adopted: one GPU job at a time.
- **Checkpoint selection changed from mean to median validation error.** The mean is dominated by the
  interferer tail and picked an epoch-0 model. Validation-only; disclosed in DECISIONS.md.
- **Download.** Zenodo served about 0.2–0.5 MB/s per connection, with intermittent 504 errors. The archive was
  streamed with 8, then 16, parallel range requests over about 1.5 hours, resuming after a restart, with 0
  corrupt clips.
- **The documented file-name format is off by one character** (the timestamp has 14 digits, not 13), and a few
  take-off clips carry negative altitudes. Parsing uses a regular expression, and those clips are excluded.

## Suggested next steps

1. **Hybrid model.** Feed the SRP-PHAT steered-response map (geometry-aware timing) into the CNN alongside the
   spectral branch, to get the beamformer's precision and the CNN's robustness to interferers.
2. **Handle competing sources explicitly.** Options: predict several directions or a full direction map
   instead of one bearing; use the stage-1 detector to weight time–frequency bins that sound like a drone
   before beamforming; track bearings over time (e.g. a Kalman filter) so brief lock-ons to an interferer are
   rejected.
3. **Treat range with caution.** Every flight was at about 30 m altitude, so "range" is largely the elevation
   angle, and recording gain differed by about 18 dB between days. Neither would hold elsewhere. A real range
   estimate needs varied altitudes, a calibrated gain, or a second array for triangulation.
4. **Test generalisation properly.** Hold out a whole recording day or site, add new drone types (especially
   engine-driven and fixed-wing), and record drone-free ambience at the deployment site for realistic noise tests.
5. **More data per trial geometry.** Only 22 training flights limit how well the CNN can learn the array's
   timing structure. The synthetic DDL archive, or simulated array data, could pre-train the timing branch.

## Files

- [results/metrics.json](results/metrics.json), [per_trial.csv](results/per_trial.csv),
  [per_range_bin.csv](results/per_range_bin.csv), and `predictions_test.parquet` (every test clip's predictions).
- [results/noise_curve.csv](results/noise_curve.csv), [noise_summary.json](results/noise_summary.json),
  `error_vs_snr.png`, `error_vs_true_range.png`.
- `bearing_test.png`, `range_test.png`, `feature_importance.png`, `array_geometry.png/.json`, and
  `training_history_*.json`.
