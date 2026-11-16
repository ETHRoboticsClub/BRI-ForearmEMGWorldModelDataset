import argparse
import json
from pathlib import Path

import numpy as np

from femg.normalize import display_montage, has_kinematics, normalize_kin
from femg.signal_utils import envelope

FOLDS = 5


def lag_curve(folder: Path, lags: np.ndarray) -> np.ndarray:
    """Cross-validated R2 of a linear decoder from the log EMG envelope (display montage) at frame t to the 0-1 pose
    at t + lag, per lag in frames. The pose trails the muscle activity, so the curve peaks at a positive lag."""
    meta = json.loads((folder / "meta.json").read_text())
    channels = display_montage(meta)[1]
    window = round(meta["processor"]["rms_window_s"] * meta["label_fs"])
    rms = envelope(np.load(folder / "emg.npy", mmap_mode="r")[:, channels], meta["fs"] // meta["label_fs"], window)
    kin = normalize_kin(np.load(folder / "kin.npy"), meta["kin_stats"])[:, meta["measured_joints"]]
    n = min(len(rms), len(kin))
    x = np.c_[np.log(rms[:n].astype(np.float64) + 1e-9), np.ones(n)]
    kin = kin[:n].astype(np.float64)
    fold = np.arange(n) * FOLDS // n  # contiguous blocks: neighbouring frames never split between train and test

    r2 = []
    for lag in lags:
        t = np.arange(max(0, -lag), min(n, n - lag))
        y, features, labelled = kin[t + lag], x[t], np.isfinite(kin[t + lag]).all(axis=1)
        error = total = 0.0
        for f in range(FOLDS):
            train, test = labelled & (fold[t] != f), labelled & (fold[t] == f)
            weights = np.linalg.lstsq(features[train], y[train], rcond=None)[0]
            error += ((y[test] - features[test] @ weights) ** 2).sum()
            total += ((y[test] - y[train].mean(axis=0)) ** 2).sum()
        r2.append(1 - error / total)
    return np.array(r2)


def main():
    parser = argparse.ArgumentParser(
        description="Where the pose best follows the EMG, per session. Compare a dataset with kin_lag_s in doubt to "
        "datasets recorded on one clock (NinaPro DB2/DB7: 240-300 ms, GREAT: 60-180 ms)."
    )
    parser.add_argument("sessions", type=Path, nargs="+", help="processed session folders")
    parser.add_argument("--max-s", type=float, default=1.5, help="lags searched: -max-s .. +max-s")
    args = parser.parse_args()

    for folder in args.sessions:
        meta = json.loads((folder / "meta.json").read_text())
        if not has_kinematics(meta):
            print(f"{meta['key']:32} no kinematics")
            continue
        step = max(1, round(0.02 * meta["label_fs"]))
        reach = round(args.max_s * meta["label_fs"])
        lags = np.arange(-reach, reach + 1, step)
        r2 = lag_curve(folder, lags)
        best_ms = lags[np.argmax(r2)] / meta["label_fs"] * 1000
        at_0 = r2[lags == 0][0]
        print(f"{meta['key']:32} pose trails the EMG by {best_ms:5.0f} ms (R2 {r2.max():.2f}, at 0 ms {at_0:.2f})")


if __name__ == "__main__":
    main()
