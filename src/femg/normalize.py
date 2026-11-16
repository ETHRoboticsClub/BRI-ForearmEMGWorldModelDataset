import warnings

import numpy as np
from scipy.ndimage import binary_closing

from femg.hand.joints import JOINT_INDEX

FINGER_FLEX = [JOINT_INDEX[f"{f}_{j}_flex"] for f in ["index", "middle", "ring", "pinky"] for j in ["mcp", "pip"]]


# Montages the tools show and compare by default, in order of preference: the 8-electrode ring, else the small
# montage of a dataset without one (SEEDS extensors, KIN-MUS UJI spots), else all channels.
DISPLAY_MONTAGES = ("ring8", "extensor8", "spots7")


def display_montage(meta: dict) -> tuple[str, list[int]]:
    montages = meta.get("montages") or {}
    for name in DISPLAY_MONTAGES:
        if name in montages:
            return name, list(montages[name])
    return "all", list(range(meta["n_channels"]))


def has_kinematics(meta: dict) -> bool:
    """False for cue-only sessions (NinaPro DB3, DB4): no joint is measured."""
    return bool(meta.get("measured_joints"))


def as_array(values) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], np.float32)


def normalize_emg(rms: np.ndarray, emg_stats: dict, channels: list[int]) -> np.ndarray:
    active = as_array(emg_stats["active"])[channels]
    return rms / active


def normalize_kin(kin: np.ndarray, kin_stats: dict) -> np.ndarray:
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
