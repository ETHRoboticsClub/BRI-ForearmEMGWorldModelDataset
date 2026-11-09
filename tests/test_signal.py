import numpy as np
import pytest

from femg.signal import Filter, decimate_labels, resample, resample_nan


def test_resample_keeps_in_band_sine():
    fs_in, fs_out, f = 2000, 500, 40
    t = np.arange(4000) / fs_in
    y = resample(np.sin(2 * np.pi * f * t)[:, None], fs_in, fs_out)
    assert y.dtype == np.float32 and y.shape == (1000, 1)
    ref = np.sin(2 * np.pi * f * np.arange(1000) / fs_out)
    assert np.abs(y[100:-100, 0] - ref[100:-100]).max() < 1e-2


def test_resample_nan_keeps_unmeasured_columns_nan():
    x = np.zeros((1000, 3), np.float32)
    x[:, 1] = np.nan
    x[10, 2] = np.nan  # isolated gap is filled
    y = resample_nan(x, 1000, 100)
    assert np.isnan(y[:, 1]).all() and np.isfinite(y[:, [0, 2]]).all()


def test_filter_removes_dc_and_mains():
    fs = 1000
    t = np.arange(20 * fs) / fs
    x = (5.0 + np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)).astype(np.float32)[:, None]
    y = Filter(fs, highpass=10, mains=50)(x)[5 * fs :, 0]
    assert abs(y.mean()) < 1e-2
    spectrum = np.abs(np.fft.rfft(y)) / len(y)
    freqs = np.fft.rfftfreq(len(y), 1 / fs)
    assert spectrum[np.argmin(abs(freqs - 50))] < 0.02
    assert spectrum[np.argmin(abs(freqs - 120))] == pytest.approx(0.5, rel=0.05)


def test_decimate_labels_picks_existing_values():
    x = np.repeat(np.arange(10, dtype=np.int16), 20)
    y = decimate_labels(x, 2000, 100)
    assert len(y) == 10 and set(y) <= set(x)
