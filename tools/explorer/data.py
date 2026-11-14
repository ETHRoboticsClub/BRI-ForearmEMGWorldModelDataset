import json
from functools import cached_property
from pathlib import Path

import numpy as np

from femg.normalize import as_array, display_montage, hand_flexion, has_kinematics, normalize_kin
from femg.signal_utils import envelope

OPEN_SESSIONS = 8
SUMMARY = ["key", "dataset", "subject", "session", "condition", "label_source", "kin_units", "n_channels", "duration_s"]
DETAILS = SUMMARY + [
    "device",
    "electrode_type",
    "citation",
    "url",
    "native_fs",
    "effective_fs",
    "fs",
    "label_fs",
    "montages",
    "joints",
    "measured_joints",
    "kin_lag_s",
    "verified",
]


class Session:
    def __init__(self, folder: Path):
        self.folder = folder
        self.meta = json.loads((folder / "meta.json").read_text())
        self.fs = self.meta["fs"]
        self.label_fs = self.meta["label_fs"]
        self.emg_per_label = self.fs // self.label_fs
        self.rms_frames = round(self.meta["processor"].get("rms_window_s", 0.1) * self.label_fs)
        self.active = as_array(self.meta["emg_stats"]["active"])
        self.overviews = {}

    @cached_property
    def emg(self) -> np.ndarray:
        return np.load(self.folder / "emg.npy", mmap_mode="r")

    @cached_property
    def kin(self) -> np.ndarray:
        return np.load(self.folder / "kin.npy")

    @cached_property
    def stimulus(self) -> np.ndarray:
        return np.load(self.folder / "stimulus.npy")

    @cached_property
    def kin_unit(self) -> np.ndarray:
        return normalize_kin(self.kin, self.meta["kin_stats"])

    @cached_property
    def kin_center(self) -> np.ndarray:
        """The session's mean pose in 0-1 units, the neutral of the `neutral: mean` hand joints."""
        stats = self.meta["kin_stats"]
        return (as_array(stats["mean"]) - as_array(stats["low"])) / (as_array(stats["high"]) - as_array(stats["low"]))

    def ring(self) -> list[int]:
        """The default channels: ring8, or the dataset's small montage when it has no ring (display_montage)."""
        return display_montage(self.meta)[1]

    def details(self) -> dict:
        stimuli = sorted({int(s) for s in np.unique(self.stimulus)} - {0, -1})
        return {k: self.meta.get(k) for k in DETAILS} | {
            "display_montage": display_montage(self.meta)[0],
            "stimuli": stimuli,
            "stimulus_names": self.meta.get("stimulus_names", {}),
            "emg_stats": {k: self.meta["emg_stats"][k] for k in ("floor", "active")},
            "kin_stats": {k: self.meta["kin_stats"][k] for k in ("low", "high")},
        }

    def overview(self, points: int) -> dict:
        if points not in self.overviews:
            self.overviews[points] = self.compute_overview(points)
        return self.overviews[points]

    def compute_overview(self, points: int) -> dict:
        ring = self.ring()
        rms = envelope(self.emg[:, ring], self.emg_per_label, self.rms_frames).mean(axis=1)
        rms = rms / max(float(np.percentile(rms, 99)), 1e-12)
        flexion = np.nan_to_num(hand_flexion(self.kin_unit)) if has_kinematics(self.meta) else np.zeros(len(rms))
        step = max(1, len(rms) // points)
        return {
            "dt": step / self.label_fs,
            "envelope": rounded(bucket(rms, step, np.max), 3),
            "flexion": rounded(bucket(flexion[: len(rms)], step, np.mean), 3),
            "stimulus": segments(self.stimulus, 0, self.label_fs),
        }

    def window(self, start: float, end: float, points: int, emg_scale: str, kin_scale: str, channels: str) -> dict:
        picked = self.ring() if channels == "ring8" else list(range(self.emg.shape[1]))
        gain = 1e6 if emg_scale == "uv" else 1 / self.active[picked]

        i0, i1 = clamp(start * self.fs, len(self.emg)), clamp(end * self.fs, len(self.emg))
        emg = np.asarray(self.emg[i0:i1, picked], np.float32) * gain
        emg_step = max(1, len(emg) // points)
        low, high = bucket(emg, emg_step, np.min), bucket(emg, emg_step, np.max)

        k0, k1 = i0 // self.emg_per_label, i1 // self.emg_per_label
        padded = max(0, k0 - self.rms_frames)
        rms = (
            envelope(
                self.emg[padded * self.emg_per_label : k1 * self.emg_per_label, picked],
                self.emg_per_label,
                self.rms_frames,
            )[k0 - padded :]
            * gain
        )
        label_step = max(1, (k1 - k0) // (2 * points))

        kin = self.kin_unit if kin_scale == "unit" else self.kin
        joints = [int(j) for j in self.meta["measured_joints"]]
        return {
            "start": i0 / self.fs,
            "end": i1 / self.fs,
            "emg": {
                "t0": i0 / self.fs,
                "dt": emg_step / self.fs,
                "channels": picked,
                "min": rounded(low.T, 2),
                "max": rounded(high.T, 2),
            },
            "envelope": {
                "t0": k0 / self.label_fs,
                "dt": label_step / self.label_fs,
                "values": rounded(rms[::label_step].T, 3),
            },
            "kin": {
                "t0": k0 / self.label_fs,
                "dt": label_step / self.label_fs,
                "joints": joints,
                "values": rounded(kin[k0:k1:label_step, joints].T, 4),
            },
            "stimulus": segments(self.stimulus[k0:k1], k0, self.label_fs),
        }


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.folders = {
            f"{p.parent.parent.parent.name}/{p.parent.parent.name}/{p.parent.name}": p.parent
            for p in sorted(root.glob("*/*/*/meta.json"))
        }
        self.open: dict[str, Session] = {}

    def summaries(self) -> list[dict]:
        out = []
        for folder in self.folders.values():
            meta = json.loads((folder / "meta.json").read_text())
            out.append({k: meta.get(k) for k in SUMMARY})
        return out

    def session(self, key: str) -> Session:
        if key not in self.open:
            if len(self.open) >= OPEN_SESSIONS:
                del self.open[next(iter(self.open))]
            self.open[key] = Session(self.folders[key])
        return self.open[key]


def clamp(index: float, length: int) -> int:
    return int(min(max(index, 0), length))


def bucket(x: np.ndarray, step: int, reduce) -> np.ndarray:
    n = len(x) // step * step
    if n == 0:
        return x[:0]
    return reduce(x[:n].reshape(n // step, step, *x.shape[1:]), axis=1)


def rounded(x: np.ndarray, decimals: int) -> list:
    return np.round(np.nan_to_num(np.asarray(x, np.float64)), decimals).tolist()


def segments(stimulus: np.ndarray, first_frame: int, label_fs: float) -> list[list[float]]:
    """Runs of the same instructed movement: [start_s, end_s, id], rest and unknown left out."""
    if len(stimulus) == 0:
        return []
    edges = np.flatnonzero(np.diff(stimulus)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(stimulus)]])
    return [
        [(first_frame + s) / label_fs, (first_frame + e) / label_fs, int(stimulus[s])]
        for s, e in zip(starts, ends, strict=True)
        if stimulus[s] > 0
    ]
