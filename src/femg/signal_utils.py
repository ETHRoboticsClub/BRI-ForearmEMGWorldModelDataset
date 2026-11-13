import math
import warnings
from fractions import Fraction

import numpy as np
from scipy import signal


def resample_ratio(fs_in: float, fs_out: float) -> Fraction:
    """fs_out / fs_in as a fraction up/down, e.g. 500 / 2000 -> 1/4."""
    return Fraction(fs_out / fs_in).limit_denominator(1000)


def resample(x: np.ndarray, fs_in: float, fs_out: float, padtype: str = "constant") -> np.ndarray:
    """Polyphase resampling along time with an anti-aliasing filter.

    padtype="line" avoids edge artifacts on slow signals (kinematics) that do not start and end at zero.
    """
    if fs_in == fs_out:
        return np.ascontiguousarray(x, dtype=np.float32)
    ratio = resample_ratio(fs_in, fs_out)
    y = signal.resample_poly(x.astype(np.float32), ratio.numerator, ratio.denominator, axis=0, padtype=padtype)
    return np.ascontiguousarray(y, dtype=np.float32)


def resample_with_nans(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Resample each column separately. All-NaN columns (unmeasured joints) stay NaN; isolated NaNs are filled."""
    ratio = resample_ratio(fs_in, fs_out)
    n_out = math.ceil(len(x) * ratio.numerator / ratio.denominator)  # same length resample() returns
    out = np.full((n_out, x.shape[1]), np.nan, np.float32)
    for j in range(x.shape[1]):
        col = x[:, j]
        valid = np.isfinite(col)
        if not valid.any():
            continue  # unmeasured joint: stays NaN
        if not valid.all():
            t = np.arange(len(col))
            col = np.interp(t, t[valid], col[valid])
        out[:, j] = resample(col[:, None], fs_in, fs_out, padtype="line")[:, 0]
    return out


def pick_nearest(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Downsample a categorical signal (stimulus ids) by picking the nearest sample: no averaging of ids."""
    n_out = int(len(x) * fs_out / fs_in)
    idx = np.round(np.arange(n_out) * fs_in / fs_out).astype(np.int64)
    return x[np.minimum(idx, len(x) - 1)]


def causal_filter(x: np.ndarray, fs: float, highpass_hz: float | None, mains_hz: float | None) -> np.ndarray:
    """Causal 4th-order high-pass + notches at the mains frequency and its harmonics.

    Causal means each output sample only depends on past input, like a filter running on a real device.
    """
    sections = []
    if highpass_hz:
        sections.append(signal.butter(4, highpass_hz, btype="highpass", fs=fs, output="sos"))
    if mains_hz:
        for f in np.arange(mains_hz, fs / 2 - 1, mains_hz):  # 50, 100, 150, ... below Nyquist
            b, a = signal.iirnotch(f, Q=30, fs=fs)
            sections.append(signal.tf2sos(b, a))
    if not sections:
        return x
    sos = np.concatenate(sections)
    return signal.sosfilt(sos, x, axis=0).astype(np.float32)


def rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=0))


def emg_stats(emg: np.ndarray, fs: float, calib_s: float) -> dict:
    """Per-channel statistics, stored in meta.json so the loader can normalize without reading whole sessions.

    calib_rms only uses the first calib_s seconds, like a calibration when the device starts.
    """
    calib = emg[: max(1, int(calib_s * fs))]
    absolute = np.abs(emg)
    return {
        "rms": rms(emg).tolist(),
        "calib_rms": rms(calib).tolist(),
        "median_abs": np.median(absolute, axis=0).tolist(),
        "p99_abs": np.percentile(absolute, 99, axis=0).tolist(),
    }


def kin_stats(kin: np.ndarray) -> dict:
    """Per-joint statistics (None for unmeasured joints), used for per-session label normalization."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "mean of empty slice" on all-NaN columns
        stats = {
            "mean": np.nanmean(kin, axis=0),
            "std": np.nanstd(kin, axis=0),
            "p1": np.nanpercentile(kin, 1, axis=0),
            "p99": np.nanpercentile(kin, 99, axis=0),
        }
    return {name: [float(v) if np.isfinite(v) else None for v in values] for name, values in stats.items()}
