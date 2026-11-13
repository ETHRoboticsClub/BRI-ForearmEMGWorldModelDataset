import math
import warnings
from fractions import Fraction

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d


def resample_ratio(fs_in: float, fs_out: float) -> Fraction:
    return Fraction(fs_out / fs_in).limit_denominator(1000)


def resample(x: np.ndarray, fs_in: float, fs_out: float, padtype: str = "constant") -> np.ndarray:
    """Anti-aliased polyphase resampling. padtype="line" avoids edge artifacts on slow signals (kinematics)."""
    if fs_in == fs_out:
        return np.ascontiguousarray(x, dtype=np.float32)
    ratio = resample_ratio(fs_in, fs_out)
    y = signal.resample_poly(x.astype(np.float32), ratio.numerator, ratio.denominator, axis=0, padtype=padtype)
    return np.ascontiguousarray(y, dtype=np.float32)


def resample_with_nans(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """All-NaN columns (unmeasured joints) stay NaN; isolated NaNs are interpolated."""
    ratio = resample_ratio(fs_in, fs_out)
    n_out = math.ceil(len(x) * ratio.numerator / ratio.denominator)
    out = np.full((n_out, x.shape[1]), np.nan, np.float32)
    for j in range(x.shape[1]):
        col = x[:, j]
        valid = np.isfinite(col)
        if not valid.any():
            continue
        if not valid.all():
            t = np.arange(len(col))
            col = np.interp(t, t[valid], col[valid])
        out[:, j] = resample(col[:, None], fs_in, fs_out, padtype="line")[:, 0]
    return out


def pick_nearest(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Downsample categorical ids (stimulus) without averaging them."""
    n_out = int(len(x) * fs_out / fs_in)
    idx = np.round(np.arange(n_out) * fs_in / fs_out).astype(np.int64)
    return x[np.minimum(idx, len(x) - 1)]


def causal_filter(x: np.ndarray, fs: float, highpass_hz: float | None, mains_hz: float | None) -> np.ndarray:
    """4th-order high-pass + notches at mains and its harmonics. Causal, like a filter on a device."""
    sections = []
    if highpass_hz:
        sections.append(signal.butter(4, highpass_hz, btype="highpass", fs=fs, output="sos"))
    if mains_hz:
        for f in np.arange(mains_hz, fs / 2 - 1, mains_hz):
            b, a = signal.iirnotch(f, Q=30, fs=fs)
            sections.append(signal.tf2sos(b, a))
    if not sections:
        return x
    sos = np.concatenate(sections)
    return signal.sosfilt(sos, x, axis=0).astype(np.float32)


def rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=0))


def envelope(emg: np.ndarray, samples_per_frame: int, window_frames: int) -> np.ndarray:
    """Moving RMS with one value per label frame: (T, C) -> (T // samples_per_frame, C)."""
    n_frames = len(emg) // samples_per_frame
    frames = np.asarray(emg[: n_frames * samples_per_frame], np.float32).reshape(n_frames, samples_per_frame, -1)
    power = (frames**2).mean(axis=1)
    return np.sqrt(uniform_filter1d(power, window_frames, axis=0))


def emg_stats(emg, fs, calib_s, samples_per_frame, window_frames, anchors) -> dict:
    floor, active = np.percentile(envelope(emg, samples_per_frame, window_frames), anchors, axis=0)
    calib = emg[: max(1, int(calib_s * fs))]
    absolute = np.abs(emg)
    return {
        "floor": floor.tolist(),
        "active": active.tolist(),
        "rms": rms(emg).tolist(),
        "calib_rms": rms(calib).tolist(),
        "median_abs": np.median(absolute, axis=0).tolist(),
        "p99_abs": np.percentile(absolute, 99, axis=0).tolist(),
    }


def kin_stats(kin: np.ndarray, anchors) -> dict:
    """None for unmeasured or constant joints."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        low, high = np.nanpercentile(kin, anchors, axis=0)
        stats = {
            "low": low,
            "high": np.where(high - low > 1e-6, high, np.nan),
            "mean": np.nanmean(kin, axis=0),
            "std": np.nanstd(kin, axis=0),
            "p1": np.nanpercentile(kin, 1, axis=0),
            "p99": np.nanpercentile(kin, 99, axis=0),
        }
    return {name: [float(v) if np.isfinite(v) else None for v in values] for name, values in stats.items()}
