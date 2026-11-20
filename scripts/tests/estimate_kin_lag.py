"""Where the hand follows the EMG: the delay at which a linear decoder from the EMG envelope best predicts the hand.

The number we want is a dataset's clock error (`kin_lag_s`), not the physiological delay. So the delay is measured the
same way on the dataset in doubt and on datasets recorded on one clock (--reference), and the clock error is the
difference: whatever the estimator adds (electromechanical delay, filters, the kind of movement) cancels out.

Target (--target):
- speed (default): the mean joint speed of the hand. The EMG envelope is muscle activity; speed is its most direct
  kinematic trace (a movement starts when the muscles fire and stops when they relax), so the curve has a sharp peak.
  It is also what the EMG-FK authors synchronized on (EMG move commands against joint-velocity peaks).
- pose: the joints themselves (the previous method). A pose is held long after the muscles that made it relax, so a
  decoder fits nearly as well at any lag around the true one: a flat curve, a peak that noise can move.

A label frame is only paired with EMG of the same segment (continuous recording) and of the same part (--parts 2: the
first and second half of the session, which EMG-FK synchronized separately). Cross-validated R2 over 5 contiguous folds.

    python scripts/tests/estimate_kin_lag.py data/interim/emg_fk/*/* --reference data/interim/great/*/*
    python scripts/tests/estimate_kin_lag.py data/interim/great/p1/* --inject-s 0.3   # must find 300 ms more
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from femg.datasets import DATASETS
from femg.normalize import Normalizer, display_montage, has_kinematics
from femg.signal_utils import envelope

FOLDS = 5
MAX_WIDTH_S = 0.5  # a curve wider than this around its peak, or a peak on the edge of the range: no estimate


def segment_ids(meta: dict, n: int) -> np.ndarray:
    ids = np.full(n, -1)
    for i, (start, end) in enumerate(meta["segments"]):
        ids[start : min(end, n)] = i
    return ids


def shift_within_segments(kin: np.ndarray, segments: np.ndarray, frames: int) -> np.ndarray:
    """kin delayed by `frames` inside every segment (the first frames of a segment lose their label)."""
    out = np.full_like(kin, np.nan)
    for s in np.unique(segments[segments >= 0]):
        idx = np.flatnonzero(segments == s)
        if len(idx) > frames:
            out[idx[frames:]] = kin[idx[: len(idx) - frames]]
    return out


def load(folder: Path, target: str, inject_s: float = 0.0) -> dict:
    """Features (log EMG envelope + bias), target, segment id per label frame."""
    meta = json.loads((folder / "meta.json").read_text())
    channels = display_montage(meta)[1]
    label_fs = meta["label_fs"]
    window = round(meta["processor"]["rms_window_s"] * label_fs)
    rms = envelope(np.load(folder / "emg.npy", mmap_mode="r")[:, channels], meta["fs"] // label_fs, window)
    kin = Normalizer.kin(np.load(folder / "kin.npy"), meta["kin_stats"])[:, meta["measured_joints"]]
    n = min(len(rms), len(kin))
    segments = segment_ids(meta, n)
    kin = kin[:n].astype(np.float64)
    if inject_s:
        kin = shift_within_segments(kin, segments, round(inject_s * label_fs))

    if target == "speed":
        velocity = np.gradient(kin, axis=0) * label_fs
        joins = np.flatnonzero(np.diff(segments) != 0)
        velocity[np.r_[joins, joins + 1]] = np.nan  # a gradient across a join is not a movement
        y = np.abs(velocity).mean(axis=1, keepdims=True)  # NaN where any joint has no label
    else:
        y = kin
    log_rms = np.log(rms[:n].astype(np.float64) + 1e-9)
    x = np.c_[log_rms, np.ones(n)]
    if target == "speed":
        # a movement is a change of activation: a held grasp keeps the envelope high and the hand still
        change = np.abs(np.gradient(log_rms, axis=0)) * label_fs
        change[np.r_[np.flatnonzero(np.diff(segments) != 0), np.flatnonzero(np.diff(segments) != 0) + 1]] = 0
        x = np.c_[log_rms, uniform_filter1d(change, window, axis=0), np.ones(n)]
    return {"meta": meta, "x": x, "y": y, "segments": segments}


def lag_curve(x: np.ndarray, y: np.ndarray, groups: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Cross-validated R2 of x[t] -> y[t + lag] per lag (frames), pairing frames of the same group only."""
    n = len(x)
    fold = np.arange(n) * FOLDS // n  # contiguous blocks: neighbouring frames never split between train and test
    r2 = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        t = np.arange(max(0, -lag), min(n, n - lag))
        ok = (groups[t] >= 0) & (groups[t] == groups[t + lag]) & np.isfinite(y[t + lag]).all(axis=1)
        t = t[ok]
        if len(t) < 100 * FOLDS:
            continue
        features, target = x[t], y[t + lag]
        error = total = 0.0
        for f in range(FOLDS):
            train, test = fold[t] != f, fold[t] == f
            if test.sum() == 0 or train.sum() < features.shape[1]:
                continue
            weights = np.linalg.lstsq(features[train], target[train], rcond=None)[0]
            error += ((target[test] - features[test] @ weights) ** 2).sum()
            total += ((target[test] - target[train].mean(axis=0)) ** 2).sum()
        r2[i] = 1 - error / total if total > 0 else np.nan
    return r2


def peak(lags_s: np.ndarray, r2: np.ndarray) -> dict:
    """Best lag (refined by a parabola through the top 3 points) and how sharp the curve is: the width of the lags
    whose R2 is above the middle of [min, max] over the searched range (a flat curve: wide, its peak unreliable)."""
    if not np.isfinite(r2).any():
        return {"lag_s": np.nan, "r2": np.nan, "r2_at_0": np.nan, "width_s": np.nan}
    i = int(np.nanargmax(r2))
    lag = lags_s[i]
    if 0 < i < len(r2) - 1 and np.isfinite(r2[i - 1 : i + 2]).all():
        a, b, c = r2[i - 1 : i + 2]
        denominator = a - 2 * b + c
        if denominator < 0:
            lag += 0.5 * (a - c) / denominator * (lags_s[1] - lags_s[0])
    half = (np.nanmax(r2) + np.nanmin(r2)) / 2
    above = np.flatnonzero(r2 >= half)
    width = lags_s[above[-1]] - lags_s[above[0]]
    zero = np.flatnonzero(lags_s == 0)
    return {"lag_s": lag, "r2": float(r2[i]), "r2_at_0": float(r2[zero[0]]) if len(zero) else np.nan, "width_s": width}


def estimate(folder: Path, target: str, parts: int, max_s: float, step_s: float, inject_s: float) -> list[dict]:
    data = load(folder, target, inject_s)
    meta, label_fs = data["meta"], data["meta"]["label_fs"]
    step = max(1, round(step_s * label_fs))
    reach = round(max_s * label_fs)
    lags = np.arange(-reach, reach + 1, step)
    n = len(data["x"])
    part = np.arange(n) * parts // n
    rows = []
    for p in range(parts):
        groups = np.where(part == p, data["segments"], -1)
        r2 = lag_curve(data["x"], data["y"], groups, lags)
        row = {"key": meta["key"], "dataset": meta["dataset"], "part": p + 1, **peak(lags / label_fs, r2)}
        on_edge = abs(row["lag_s"]) >= max_s - step_s
        row["reliable"] = bool(np.isfinite(row["lag_s"]) and not on_edge and row["width_s"] <= MAX_WIDTH_S)
        rows.append(row)
    return rows


def print_summary(rows: list[dict], reference: set[str], parts: int):
    by_dataset = defaultdict(list)
    for row in rows:
        if row["reliable"]:
            by_dataset[row["dataset"]].append(row["lag_s"])
    medians = {name: float(np.nanmedian(v)) for name, v in by_dataset.items()}

    n_unreliable = sum(not r["reliable"] for r in rows)
    print("\nper dataset: median lag over sessions" + (" and parts" if parts > 1 else ""))
    if n_unreliable:
        print(f"  ({n_unreliable} of {len(rows)} left out: peak on the edge or curve wider than {MAX_WIDTH_S} s)")
    for name, values in by_dataset.items():
        values = np.array(values)
        q1, q3 = np.nanpercentile(values, [25, 75])
        role = " reference" if name in reference else ""
        print(
            f"  {name:16} {medians[name] * 1000:6.0f} ms  (IQR {q1 * 1000:.0f}..{q3 * 1000:.0f}, n {len(values)}){role}"
        )

    if not reference:
        return
    ref = float(np.nanmedian([r["lag_s"] for r in rows if r["dataset"] in reference and r["reliable"]]))
    print(f"\nreference lag (median of {', '.join(sorted(reference))}): {ref * 1000:.0f} ms")
    for name in by_dataset:
        if name in reference:
            continue
        excess = medians[name] - ref
        current = DATASETS[name]().config.kin_lag_s if name in DATASETS else 0.0
        print(
            f"  {name}: {excess * 1000:+.0f} ms against the reference -> kin_lag_s {current:+.2f} would become "
            f"{current - excess:+.2f} (rounded to the label rate)"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sessions", type=Path, nargs="+", help="interim session folders")
    parser.add_argument("--reference", type=Path, nargs="*", default=[], help="sessions on one clock (GREAT, DB2...)")
    parser.add_argument("--target", choices=["speed", "pose"], default="speed")
    parser.add_argument("--parts", type=int, default=2, help="estimate each part of a session (1 = whole)")
    parser.add_argument("--max-s", type=float, default=1.0, help="lags searched: -max-s .. +max-s")
    parser.add_argument("--step-s", type=float, default=0.02)
    parser.add_argument("--inject-s", type=float, default=0.0, help="self-check: delay the labels by this much")
    args = parser.parse_args()

    rows, reference = [], set()
    print(f"target {args.target}; lag > 0: the hand trails the EMG; width: how flat the curve is around its peak")
    for folder, is_reference in [(f, False) for f in args.sessions] + [(f, True) for f in args.reference]:
        meta = json.loads((folder / "meta.json").read_text())
        if not has_kinematics(meta):
            print(f"{meta['key']:32} no kinematics")
            continue
        if is_reference:
            reference.add(meta["dataset"])
        for row in estimate(folder, args.target, args.parts, args.max_s, args.step_s, args.inject_s):
            rows.append(row)
            print(
                f"{row['key']:32} part {row['part']}  lag {row['lag_s'] * 1000:5.0f} ms  R2 {row['r2']:.2f} "
                f"(at 0 ms {row['r2_at_0']:.2f})  width {row['width_s'] * 1000:4.0f} ms"
                + ("" if row["reliable"] else "  UNRELIABLE"),
                flush=True,
            )
    print_summary(rows, reference, args.parts)


if __name__ == "__main__":
    main()
