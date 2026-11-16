import math
import warnings
from fractions import Fraction

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d


def resample_ratio(fs_in: float, fs_out: float) -> Fraction:
    return Fraction(fs_out / fs_in).limit_denominator(1000)


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Anti-aliased polyphase resampling (EMG)."""
    if fs_in == fs_out:
        return np.ascontiguousarray(x, dtype=np.float32)
    ratio = resample_ratio(fs_in, fs_out)
    y = signal.resample_poly(x.astype(np.float32), ratio.numerator, ratio.denominator, axis=0)
    return np.ascontiguousarray(y, dtype=np.float32)


def pick_nearest(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Downsample categorical ids (stimulus) without averaging them."""
    n_out = int(len(x) * fs_out / fs_in)
    idx = np.round(np.arange(n_out) * fs_in / fs_out).astype(np.int64)
    return x[np.minimum(idx, len(x) - 1)]


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of every run of True."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def bridge_gaps(valid: np.ndarray, max_gap: int) -> np.ndarray:
    """valid, with every run of False of at most max_gap samples between two valid samples set to True."""
    bridged = valid.copy()
    for a, b in runs(~valid):
        if a > 0 and b < len(valid) and b - a <= max_gap:
            bridged[a:b] = True
    return bridged


def frame_knots(x: np.ndarray, rows: np.ndarray, joints: np.ndarray, max_repeat: int) -> np.ndarray:
    """The rows of x (sample indices, all finite on `joints`) that carry a new measurement.

    Gloves and cameras slower than the file rate repeat their last frame until the next one arrives (held, or
    linearly interpolated by the authors), and repeat it again when a frame is late or the value did not cross a
    quantization step: the hand then steps instead of moving (DB8 ~29 Hz, GREAT ~32 Hz, emg_fk camera). The new
    frames are the corners of that piecewise-linear signal, minus the corners that repeat the previous one: those
    are dropped unless the repeat lasts more than max_repeat samples (then the hand really was still)."""
    corner = np.zeros(len(rows), bool)
    corner[[0, -1]] = True
    for j in joints:
        corner[1:-1] |= np.abs(np.diff(x[rows, j], 2)) > 1e-6
    corners = np.flatnonzero(corner)
    same = np.ones(len(corners) - 1, bool)
    for j in joints:
        values = x[rows[corners], j]
        same &= values[1:] == values[:-1]
    repeat = same & (np.diff(rows[corners]) <= max_repeat)
    return rows[corners[np.concatenate([[True], ~repeat])]]


def resample_kin(kin, fs_in, fs_out, segments, lowpass_hz, max_gap_s, max_repeat_s) -> np.ndarray:
    """Kinematics (T, J) at fs_out, label frame k at time k / fs_out, one segment (continuous recording, from each
    start in `segments` to the next) at a time: nothing is interpolated or filtered across the join of two trials.

    A row with NaN on a measured joint has no measurement: gaps up to max_gap_s are interpolated, longer ones stay
    NaN (no label). Repeated frames are interpolated over (frame_knots), then a zero-phase low-pass at lowpass_hz,
    capped at 0.4 fs_out against aliasing, smooths the remaining quantization steps and tracking jitter. Labels
    are targets, not inputs, so they need not be causal."""
    n_out = math.ceil(len(kin) * fs_out / fs_in)
    out = np.full((n_out, kin.shape[1]), np.nan, np.float32)
    joints = np.flatnonzero(np.isfinite(kin).any(axis=0))
    if not len(joints):
        return out
    sos = signal.butter(2, min(lowpass_hz or np.inf, 0.4 * fs_out), fs=fs_in, output="sos")
    valid = np.isfinite(kin[:, joints]).all(axis=1)
    for start, end in zip(segments, [*segments[1:], len(kin)], strict=True):
        for a, b in runs(bridge_gaps(valid[start:end], round(max_gap_s * fs_in))):
            a, b = start + a, start + b  # a measured stretch, [a, b) in kin samples
            x = kin[a:b]
            knots = frame_knots(x, np.flatnonzero(valid[a:b]), joints, round(max_repeat_s * fs_in))
            frames = np.arange(math.ceil(a * fs_out / fs_in), math.floor((b - 1) * fs_out / fs_in) + 1)
            times = frames * fs_in / fs_out - a  # in samples of x
            samples = np.arange(len(x))
            for j in joints:
                dense = np.interp(samples, knots, x[knots, j])
                if len(x) > 3 * (2 * len(sos) + 1):  # sosfiltfilt's padding
                    dense = signal.sosfiltfilt(sos, dense)
                out[frames, j] = np.interp(times, samples, dense)
    return out


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
