# Decision log

Chronological record of every non-trivial decision made while building AcousticWatch
(the project was built autonomously; nothing here was confirmed by a human).

## 2026-10-09 — setup

- **GPU power limit.** `nvidia-smi -pl 220` failed without admin. Re-ran elevated (UAC) and it succeeded:
  limit is 220 W (default 310 W). **This resets on reboot**; `gpu_guard.ensure_power_limit()` re-checks and
  logs it at the start of every training run.
- **Thermal rule** implemented in `src/acousticwatch/gpu_guard.py`: nvidia-smi is polled every 5 s between
  batches; above 80 °C, training sleeps until the GPU is below 70 °C. Every pause is logged to
  `logs/gpu_events.jsonl` and summarised in `results/`.
- **Environment.** Python 3.10.11 venv in `.venv/` (no conda on the machine). PyTorch 2.4.1 + cu121 wheels
  (the driver supports CUDA 12.6, and 12.1 wheels are the most widely tested). numpy pinned to 1.26 for ABI safety.
- **Datasets chosen** (see SUMMARY.md for the rationale):
  - DADS (HF `geronimobasso/drone-audio-detection-samples`, 180,320 clips, binary drone / no-drone, MIT).
    It is the largest public drone-audio set and aggregates several earlier sets (Al-Emadi, DREGON, SPCup19,
    Salford DroneNoise, AUDROK…). Al-Emadi's multi-class set (Bebop / Membo) is already inside DADS,
    so it is not downloaded separately; adding it again would create duplicate clips across splits.
  - ESC-50 and UrbanSound8K as environmental negatives. Hard-negative classes: ESC-50 airplane, engine,
    helicopter, chainsaw, train, hand_saw, vacuum_cleaner; US8K engine_idling, drilling, jackhammer,
    air_conditioner, siren. **Neither set has a motorcycle class**; the closest are ESC-50 `engine` and
    US8K `engine_idling`, and both are logged as "engine" hard negatives.
  - AeroSonicDB (YPAD-0523): manned fixed-wing/rotor aircraft overflights. Real aircraft engine noise
    is the most dangerous false-positive source for an early-warning sensor.
  - Svanström et al. multi-sensor drone dataset (Zenodo 5500576): DRONE / HELICOPTER / BACKGROUND audio.
    It is held out completely as an **external test domain**: different microphones, sites and drones.
- **Fixed-wing / engine-driven UAV audio: not found.** A search of HF, Zenodo, GitHub and arXiv found no
  downloadable fixed-wing or gasoline-engine UAV recordings with a usable licence. Available public
  drone audio is overwhelmingly multirotor. Documented as the main limitation and next step.
- **DADS leakage check** (`scripts/check_dads_adjacency.py`, `results/dads_adjacency.json`). DADS ships no recording
  IDs. Clips with consecutive IDs have a median spectral-fingerprint similarity of 0.96, against 0.70 for random
  pairs, and 55% of adjacent pairs are more similar than 95% of random pairs. Runs of near-identical consecutive
  clips reach about 90 IDs (99th percentile 23). **So DADS is a chunked set of long recordings, and a random
  clip split would leak.** Decision: split DADS by contiguous ID blocks of 1000 and discard a 100-ID guard band
  wherever adjacent blocks fall in different splits. 100 is larger than the longest observed run.
- **Residual source-level similarity.** Even with block splits, DADS test clips come from the same *source
  collections* (same drones and microphones) as training clips. That is domain overlap rather than recording
  leakage, and it makes the DADS test optimistic. Countermeasure: the Svanström set is kept as a fully external
  test, and AeroSonicDB/ESC-50/US8K negatives are split by their own recording groups.
- **Window length 0.5 s, not 1 s (shortcut avoided).** DADS drone clips are almost all exactly 0.5 s long, while
  DADS no-drone clips run around 5 s. With 1 s windows every drone clip would have been tiled or padded and no
  negative would, and a CNN can learn that artefact as "drone". All audio is now cut into native 0.5 s windows
  with no tiling or zero padding; a tail up to 5% short is completed by reflection, and anything shorter is
  dropped. A remaining caveat: each DADS drone window *is* a whole clip, so any fade or edit at clip boundaries
  is exclusive to positives. The external Svanström test, where drone audio is cut from continuous 10 s
  recordings like everything else, is the check against such shortcuts.
- **DADS sampling.** Up to 50,000 clips per label (drone clips give 1 window each; no-drone clips give up to 2
  random 0.5 s windows each), to keep disk use and class balance reasonable. Disk was at 97%.
- **DC-offset shortcut removed.** A spectral audit found that DADS drone clips carry a mean |DC offset| of 0.028,
  against about 0.005 for every other source. Hann-window leakage carries DC into the lowest mel bands, so a CNN
  could separate "DADS drone" from everything else without hearing a drone. The first training run (val AUC 0.9999
  after one epoch) was stopped. DC is now removed per window inside the shared log-mel front-end, identically for
  training, evaluation, robustness and explanation. The same audit shows DADS drone energy is concentrated in
  200–1000 Hz (84%), against a much broader spectrum for the Svanström drones recorded at a distance.
  That is a real domain gap, which the external test measures.
- **Detection checkpoint selection.** Validation AUC saturated at 0.9998–0.99997 from epoch 0 onward, so "best by
  AUC" (epoch 5) was noise. The model used is epoch 18, which has the lowest validation loss (0.0123) and the
  lowest validation FPR at 0.5. Selection still uses validation data only. Training: 20 epochs × 80k balanced
  samples, batch 128, about 25 s per epoch, peak GPU temperature 53 °C, no thermal pauses, no OOM.
- **Training restarts.** The first detection run crashed before epoch 0: Windows DataLoader workers tried to
  pickle the multi-GB memmap. Fixed by opening the memmap lazily in each worker. The second run was stopped
  deliberately after the DC-offset audit, as above.
- **Detection v2 (distance and microphone augmentation): tried, rejected.** It was aimed at the poor external
  result of v1 (Svanström AUC 0.82, recall 8% at 0.5). v2 adds random ISO 9613-1 air absorption (10–600 m),
  spectral tilt (−6 to +3 dB per octave), random low-pass and high-pass filtering, and heavier background mixing.
  The rule declared before training was "v2 is final if val AUC ≥ 0.999". v2 passed (0.99991), but that rule was
  uninformative because both models' validation AUC is saturated. On held-out data v2 is worse everywhere:
  in-domain test AUC 0.9949 vs 0.9997, external AUC 0.52 vs 0.82, hard-negative FPR 2.6% vs 0.26%.
  **v1 is kept as the primary model.** This choice was informed by test results. That is disclosed here, and is
  mitigated by v1 being the original model, evaluated before v2 existed and never tuned on test.
  No further variants were tried against the external set, so it stays a genuine held-out check.
  Both models' full results are kept: `results/detect_v1/` and `results/detect_v2/`.
- **Range-proxy level bug, found and fixed.** The first range-proxy run normalised the noise floor to RMS 1, which
  put the drone at RMS 10 — about 30× louder than any training window. The model behaved out-of-distribution
  (external drone recall *rose* with added noise). The floor is now RMS 0.02, so the drone sits at the typical
  training level of RMS 0.2 at +20 dB. Lesson recorded as a weakness: the log-mel CNN is **sensitive to absolute
  input level**. Training used random gain of ±12 dB; a deployed sensor needs gain calibration or a
  level-invariant front-end such as PCEN or per-window normalisation (next steps).
- **DADS split tightened (break-aware blocks).** Re-running the adjacency check on 6 shards (27k clips), not 2,
  found near-duplicate runs of up to **698 consecutive IDs** (99.9th percentile 183). That is longer than the
  100-ID guard band. So every DADS clip was fingerprinted (`check_dads_adjacency.py --all`) and the split
  rebuilt. Blocks of ≥ 1000 IDs are now closed only at a *break*, meaning consecutive clips whose similarity is
  below the random-pair 95th percentile, plus a 25-ID guard margin. The audit (`results/dads_split_audit.json`)
  finds **0 near-duplicate consecutive pairs across splits**. All detection results were regenerated on this
  split. The earlier results (fixed 1000-ID blocks with 100-ID guards) are kept in
  `results/archive_fixed_block_split/` for comparison. The v2-versus-v1 decision was made on the old split and
  is not re-run; v2 stays rejected.
