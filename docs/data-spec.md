# Data spec: harmonized forearm EMG → hand kinematics

Goal: one dataset in which **any forearm EMG activation maps to the correct hand pose**, pooled from public
datasets, to train a generic EMG2pose model that can transfer to transradial amputees.

This file is the contract. `src/femg/joints.py`, `src/femg/processor.py` and `params.yaml` implement it; change them together.

## 1. Requirements

| # | Requirement | How it is met |
|---|---|---|
| R1 | Forearm EMG only (a residual limb has no wrist) | Sources must record over the forearm muscle bellies. Wrist datasets (emg2pose) are excluded from supervised data |
| R2 | One EMG rate for every session | `processor.fs` (500 Hz). Polyphase anti-aliased resampling. **Never upsample**: datasets whose *effective* rate is below `fs` are skipped |
| R3 | Filtering a device can reproduce | Causal 10 Hz high-pass + mains notches (per dataset: 50/60 Hz) at build time. Anything stronger (band-pass, envelopes) is a model choice and belongs to the training repo |
| R4 | Physical units | EMG in volts (`Dataset.emg_unit_scale`), angles in radians |
| R5 | One label space | 22 canonical right-hand joints (`joints.CANONICAL_JOINTS`). Unmeasured joints are NaN, and the loader turns them into a `mask` for the loss |
| R6 | Know where labels come from | `label_source` ∈ measured / mirrored / cue; `kin_units` ∈ rad / rad_uncalibrated |
| R7 | Time alignment | Label `k` is the hand at EMG sample `k·fs/label_fs`. Known dataset lags are corrected at build time (`Dataset.kin_lag_s`, `Processor.align_labels`) |
| R8 | Normalization does not leak future data | Stats per session and channel are stored at build time. The default `calib_rms` uses only the first `calib_s` seconds, like a calibration at device startup |
| R9 | Subject-level splits | Every split is by subject (windows overlap). `percent` selects subjects, never windows |
| R10 | Scale from dev to full | `--percent` takes the first N% of subjects: nested and identical for downloading, building and loading |
| R11 | Reproducible + shareable | DVC stage per dataset → `s3://hri-control/forearm-emg-dataset`; `dvc.lock` pins the exact build |
| R12 | Simple, fast, low-memory build | Separate phases (download, then build). One worker process per subject, float32 throughout, resample before filtering, atomic and resumable writes |

## 2. Decisions and why

**fs = 500 Hz.** Keeps most of the surface-EMG power (< 250 Hz). Includes EMG-FK (500 Hz) and DB8 (effective
1111 Hz). Excludes Myo-based datasets (200 Hz) from the raw-signal track. If the target prosthesis hardware samples
lower, rebuild with that `fs`: the build is parametric and DVC tracks the change.
→ **Open question: what is the deployment hardware's rate and channel count?** That fixes `fs` and the montage.

**Labels at 100 Hz.** Hand kinematics have essentially no content above about 10 Hz. Storing them at the EMG rate
would waste disk and RAM.

**Store continuous sessions, window in the loader.** The training repo currently stores overlapping windows
(240/40 → 6× redundant). Here windows are cut from memory-mapped arrays on the fly (≈17 µs per item), so
window/stride/lag experiments need no rebuild.

**Channels: store native, select at load time.** Each session keeps all its channels and lists named
*montages*. `ring8` (8 electrodes evenly spaced around the forearm) exists in EMG-FK, NinaPro DB2/DB7/DB8, so it
is the common denominator and matches the current 8-channel TRR model. Rotation of the ring is **not** aligned
across datasets, so train with random circular channel shifts.

**Label normalization.** Calibrated angles stay in radians. Raw CyberGlove (DB2) has a per-session gain and
offset, so by default (`label_norm="uncalibrated_zscore"`) those sessions are z-scored per session and joint.
Mixing z-scored and radian targets in one head is a known compromise (see open questions).

## 3. Datasets

| Dataset | Status | EMG | Labels | Subjects | Notes |
|---|---|---|---|---|---|
| EMG-FK | ✅ reader, built | 8 dry monopolar, 500 Hz | 15 MediaPipe flexions (rad) | 20 intact | Angle order inferred, not confirmed. ~300 ms lag, direction unknown |
| NinaPro DB2 | ✅ reader | 12 Delsys, 2 kHz | CyberGlove 22, raw | 40 intact | E3 (force) has no glove, so it is skipped |
| NinaPro DB8 | ✅ reader, tested on S1_E1_A3 | 16 Delsys (2 rows), 1111 Hz eff. | CyberGlove 18, calibrated, mirrored | 10 intact + 2 amputees | Which channels form row 1 is not documented |
| NinaPro DB7 | ⏳ next | 12 Delsys + IMU | CyberGlove 18 | 20 + 2 amputees | Same format as DB8 |
| NinaPro DB3 | ⏳ | 12 Delsys | cue only | 11 amputees | `label_source=cue`: amputee evaluation / gesture head |
| NinaPro DB4 | ⏳ | 12 Cometa, 2 kHz | CyberGlove 22 | 10 | |
| SEEDS | ⏳ | 126 HD-EMG, 2048 Hz | CyberGlove 18 | 25 | Define ring8 by picking grid electrodes |
| KIN-MUS UJI | ⏳ | 7 ch | CyberGlove 18, daily activities | 22 | No ring8 → new montage or channel-agnostic model |
| Hyser | ⏳ | 256 HD-EMG | finger forces | 20 | Needs a force head |
| NinaPro DB5 | ❌ fs | Myo 200 Hz | glove | 10 | Only if `fs ≤ 200` |
| emg2pose | ❌ R1 | wrist | mocap | 193 | Optional self-supervised pretraining only |

## 4. Roadmap (in order)

1. **Confirm label semantics** (blocks trusting per-joint metrics)
   - EMG-FK angle order: check `figures/figure_angles.png` in the authors' repo and the paper's Fig. 1.
   - CyberGlove sign conventions (abduction, wrist deviation) across DB2/DB8 with a visual check.
   - DB8 ring rows: pick the 8 channels whose cross-correlation forms a ring.
2. **Estimate lags.** Cross-correlate EMG envelope with joint speed per session and set `kin_lag_s` (EMG-FK first:
   ~0.3 s). Add a `scripts/estimate_lag.py` and store the per-session estimate in meta.
3. **Full build of the three datasets** on Sky (`sky/build.yaml`), then `dvc push` and commit `dvc.lock`.
4. **Wire the training repo**: `dvc import` + `pip install` of this package; replace `EMGFK` with
   `EMGWindows(montage="ring8")`; check that EMG-FK-only results match the current baseline.
5. **QC report per session**: flat or saturated channels, glove glitches (e.g. DB2 sensor 11 hits ±800), stimulus
   coverage. Flag sessions in meta, and have the loader drop flagged ones.
6. **Add DB7, DB3, DB4, then SEEDS** (one module each; tests with a small real file).
7. **Channel-agnostic option**: electrode geometry (angle around the forearm, distance from the elbow) in meta for
   a set/attention encoder, instead of fixed montages.
8. **Evaluation protocol**: leave-one-subject-out, leave-one-dataset-out, and amputee-only (DB8 s11/s12, DB7
   s21/s22) reported separately.

## 5. Open questions

- Deployment hardware: sampling rate, channel count, electrode type (dry/monopolar like MindRove, or bipolar)?
- Target space: is a calibrated-radian head enough, or should every session be z-scored (relative pose)?
- Wrist joints: CyberGlove has wrist flex/deviation, MediaPipe does not. Do we want wrist outputs?
- Licenses: NinaPro asks for citation; check redistribution before sharing the built data outside the team.
