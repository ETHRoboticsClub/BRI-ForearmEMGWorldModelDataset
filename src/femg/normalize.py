import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_closing

from femg.hand.joints import JOINT_INDEX
from femg.signal_utils import envelope, rms

FINGER_FLEX = [JOINT_INDEX[f"{f}_{j}_flex"] for f in ["index", "middle", "ring", "pinky"] for j in ["mcp", "pip"]]


# Montages the tools show and compare by default, in order of preference: the 8-electrode ring, else all channels.
DISPLAY_MONTAGES = ("ring8",)


def display_montage(meta: dict) -> tuple[str, list[int]]:
    montages = meta.get("montages") or {}
    for name in DISPLAY_MONTAGES:
        if name in montages:
            return name, list(montages[name])
    return "all", list(range(meta["n_channels"]))


def has_kinematics(meta: dict) -> bool:
    """False for cue-only sessions: no joint is measured."""
    return bool(meta.get("measured_joints"))


def as_array(values) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], np.float32)


@dataclass(frozen=True)
class Normalizer:
    """Puts every session on the same scale, from anchors measured on the session itself: each EMG channel divided by
    its active level (a high percentile of its moving RMS, ~ strong contraction), each joint mapped to 0 (~ open
    hand) - 1 (~ closed hand) by two percentiles. `emg_stats` / `kin_stats` measure the anchors (saved in meta.json),
    `emg` / `kin` apply them."""

    fs: int
    label_fs: int
    rms_window_s: float  # moving-RMS window of the EMG anchors
    calib_s: float  # calib_rms: the RMS of the first calib_s seconds
    emg_anchors: tuple[float, float]  # percentiles of the moving RMS: floor (~ rest), active (~ strong contraction)
    kin_anchors: tuple[float, float]  # percentiles per joint: 0 (~ open hand) and 1 (~ closed hand)

    def envelope(self, emg: np.ndarray) -> np.ndarray:
        """(T, C) -> (T / per label frame, C): the moving RMS the EMG anchors are taken from."""
        return envelope(emg, self.fs // self.label_fs, max(1, round(self.rms_window_s * self.label_fs)))

    def emg_stats(self, emg: np.ndarray) -> dict:
        floor, active = np.percentile(self.envelope(emg), self.emg_anchors, axis=0)
        absolute = np.abs(emg)
        return {
            "floor": floor.tolist(),
            "active": active.tolist(),
            "rms": rms(emg).tolist(),
            "calib_rms": rms(emg[: max(1, int(self.calib_s * self.fs))]).tolist(),
            "median_abs": np.median(absolute, axis=0).tolist(),
            "p99_abs": np.percentile(absolute, 99, axis=0).tolist(),
        }

    def kin_stats(self, kin: np.ndarray) -> dict:
        """None for unmeasured or constant joints."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            low, high = np.nanpercentile(kin, self.kin_anchors, axis=0)
            stats = {
                "low": low,
                "high": np.where(high - low > 1e-6, high, np.nan),
                "mean": np.nanmean(kin, axis=0),
                "std": np.nanstd(kin, axis=0),
                "p1": np.nanpercentile(kin, 1, axis=0),
                "p99": np.nanpercentile(kin, 99, axis=0),
            }
        return {name: [float(v) if np.isfinite(v) else None for v in values] for name, values in stats.items()}

    @staticmethod
    def emg(x: np.ndarray, emg_stats: dict, channels: list[int] | None = None) -> np.ndarray:
        """EMG (or its RMS) / active level; 0 for channels without one (a virtual electrode no real one covers)."""
        active = as_array(emg_stats["active"])
        if channels is not None:
            active = active[channels]
        x = np.asarray(x, np.float32)
        return np.divide(x, active, out=np.zeros_like(x), where=active > 0)

    @staticmethod
    def kin(kin: np.ndarray, kin_stats: dict) -> np.ndarray:
        """Joints in 0 (~ open hand) - 1 (~ closed hand); NaN where unmeasured or unlabelled."""
        open_hand, closed_hand = as_array(kin_stats["low"]), as_array(kin_stats["high"])
        return (kin - open_hand) / (closed_hand - open_hand)


def hand_flexion(kin_norm: np.ndarray) -> np.ndarray:
    """Mean finger flexion per frame; NaN where the frame has no label."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames
        return np.nanmean(kin_norm[:, FINGER_FLEX], axis=1)


def hand_speed(flexion: np.ndarray, label_fs: float) -> np.ndarray:
    return np.abs(np.gradient(flexion)) * label_fs  # the labels are already low-passed (processor kin_lowpass)


def scenarios(flexion: np.ndarray, speed: np.ndarray) -> dict[str, np.ndarray]:
    def percentile(x, q):
        return np.nanpercentile(x, q)

    still = speed <= percentile(speed, 40)
    moving = speed >= percentile(speed, 75)
    hand_open = flexion <= percentile(flexion, 25)
    hand_closed = flexion >= percentile(flexion, 75)
    return {
        "all": np.ones(len(flexion), bool),
        "rest": still & hand_open,
        "hold": still & hand_closed,
        "moving": moving,
    }


def bouts(mask: np.ndarray, label_fs: float, min_gap_s: float = 0.2) -> np.ndarray:
    gap = max(1, round(min_gap_s * label_fs))
    mask = binary_closing(np.pad(mask, gap), np.ones(gap + 1, bool))[gap:-gap]
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return (ends - starts) / label_fs
