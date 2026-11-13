import numpy as np
from scipy.ndimage import binary_closing, uniform_filter1d

from femg.joints import JOINT_INDEX

FINGER_FLEX = [JOINT_INDEX[f"{f}_{j}_flex"] for f in ["index", "middle", "ring", "pinky"] for j in ["mcp", "pip"]]


def as_array(values) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], np.float32)


def normalize_emg(rms: np.ndarray, emg_stats: dict, channels: list[int]) -> np.ndarray:
    active = as_array(emg_stats["active"])[channels]
    return rms / active


def normalize_kin(kin: np.ndarray, kin_stats: dict) -> np.ndarray:
    open_hand, closed_hand = as_array(kin_stats["low"]), as_array(kin_stats["high"])
    return (kin - open_hand) / (closed_hand - open_hand)


def hand_flexion(kin_norm: np.ndarray) -> np.ndarray:
    return np.nanmean(kin_norm[:, FINGER_FLEX], axis=1)


def hand_speed(flexion: np.ndarray, label_fs: float, window: int) -> np.ndarray:
    smooth_flexion = uniform_filter1d(flexion, window)  # gloves slower than label_fs are held between samples
    return np.abs(np.gradient(smooth_flexion)) * label_fs


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
