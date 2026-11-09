"""Signal processing used by the builder. Everything stays float32 and time-major (T, C)."""

import warnings
from fractions import Fraction

import numpy as np
from scipy import signal


def resample(x: np.ndarray, fs_in: float, fs_out: float, padtype: str = "constant") -> np.ndarray:
    """Polyphase resampling along time (axis 0) with an anti-aliasing FIR. Returns float32.

    padtype "line" avoids edge artifacts on slow signals (kinematics) that do not start and end at zero.
    """
    if fs_in == fs_out:
        return np.ascontiguousarray(x, dtype=np.float32)
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    y = signal.resample_poly(
        x.astype(np.float32, copy=False), ratio.numerator, ratio.denominator, axis=0, padtype=padtype
    )
    return np.ascontiguousarray(y, dtype=np.float32)


def resample_nan(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Resample columns that may be all-NaN (unmeasured joints). Isolated NaNs are linearly filled first."""
    out = np.full((_resampled_len(len(x), fs_in, fs_out), x.shape[1]), np.nan, np.float32)
    for j in range(x.shape[1]):
        col = x[:, j]
        valid = np.isfinite(col)
        if not valid.any():
            continue
        if not valid.all():
            idx = np.arange(len(col))
            col = np.interp(idx, idx[valid], col[valid])
        out[:, j] = resample(col[:, None], fs_in, fs_out, padtype="line")[: len(out), 0]
    return out


def _resampled_len(n: int, fs_in: float, fs_out: float) -> int:
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return -(-n * ratio.numerator // ratio.denominator)


def decimate_labels(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Pick-nearest downsampling for categorical signals (stimulus ids)."""
    n = int(len(x) * fs_out / fs_in)
    idx = np.minimum(np.round(np.arange(n) * fs_in / fs_out).astype(np.int64), len(x) - 1)
    return x[idx]


class Filter:
    """Causal high-pass + mains notches, the same kind of filter a real-time device can run."""

    def __init__(self, fs: float, highpass: float | None, mains: float | None) -> None:
        sos = []
        if highpass:
            sos.append(signal.butter(4, highpass, btype="highpass", fs=fs, output="sos"))
        if mains:
            for f in np.arange(mains, fs / 2 - 1, mains):
                sos.append(signal.tf2sos(*signal.iirnotch(f, Q=30, fs=fs)))
        self.sos = np.concatenate(sos).astype(np.float32) if sos else None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.sos is None:
            return x
        return signal.sosfilt(self.sos, x, axis=0).astype(np.float32, copy=False)


def emg_stats(emg: np.ndarray, fs: float, calib_s: float) -> dict:
    """Per-channel statistics stored in meta.json so the loader can normalize without reading whole sessions."""
    calib = emg[: max(1, int(calib_s * fs))]
    absx = np.abs(emg)
    return {
        "rms": _rms(emg).tolist(),
        "calib_rms": _rms(calib).tolist(),
        "median_abs": np.median(absx, axis=0).tolist(),
        "p99_abs": np.percentile(absx, 99, axis=0).tolist(),
    }


def _rms(x: np.ndarray) -> np.ndarray:
    """Per-column RMS accumulated in float64 without materializing a float64 copy."""
    return np.sqrt(np.einsum("tc,tc->c", x, x, dtype=np.float64) / len(x))


def kin_stats(kin: np.ndarray) -> dict:
    """Per-joint statistics (None for unmeasured joints), used for per-session label normalization."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        return {
            "mean": _nan_list(np.nanmean(kin, axis=0)),
            "std": _nan_list(np.nanstd(kin, axis=0)),
            "p1": _nan_list(np.nanpercentile(kin, 1, axis=0)),
            "p99": _nan_list(np.nanpercentile(kin, 99, axis=0)),
        }


def _nan_list(x: np.ndarray) -> list:
    return [float(v) if np.isfinite(v) else None for v in x]
