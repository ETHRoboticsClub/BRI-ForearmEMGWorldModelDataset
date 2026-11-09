"""End-to-end checks on a synthetic source: build -> index -> windows, with exact time alignment."""

from pathlib import Path

import numpy as np
import pytest

from femg import build, store
from femg.build import BuildConfig, build_hash, process_session
from femg.loader import EMGWindows
from femg.schema import JOINT_INDEX, N_JOINTS, Recording, SessionRef
from femg.sources.base import Source
from femg.subset import select_subjects


class Synthetic(Source):
    name = "synthetic"
    citation = license = url = device = "test"
    electrode_type = "bipolar"
    native_fs = effective_fs = 1000.0
    mains_hz = None
    montages = {"ring8": list(range(8)), "first4": [0, 1, 2, 3]}
    kin_lag_s = 0.0

    def subjects(self):
        return ["s01", "s02", "s03", "s04"]

    def sessions(self, subject):
        info = {"condition": "intact", "label_source": "measured", "kin_units": "rad"}
        return [SessionRef(self.name, subject, "a", [], info)]

    def read(self, ref, raw_dir):
        t = np.arange(10_000, dtype=np.float32)  # 10 s; every channel holds the sample index
        emg = np.repeat(t[:, None], 8, axis=1)
        kin = np.full((len(t), N_JOINTS), np.nan, np.float32)
        kin[:, JOINT_INDEX["index_mcp_flex"]] = t / self.native_fs  # the time in seconds
        return Recording(emg=emg, fs=self.native_fs, kin=kin, kin_fs=self.native_fs)


@pytest.fixture
def built(tmp_path, monkeypatch):
    src = Synthetic()
    monkeypatch.setattr(build, "get_source", lambda name: src)
    cfg = BuildConfig(fs=1000, label_fs=100, highpass=None, notch=False, calib_s=1)
    for s in src.subjects():
        for ref in src.sessions(s):
            assert process_session(ref, tmp_path, tmp_path, cfg, build_hash(cfg, "x"))["status"] == "built"
    store.write_index(tmp_path / "synthetic")
    return tmp_path, src


def test_window_target_alignment(built):
    root, _ = built
    ds = EMGWindows(root, window_s=0.2, stride_s=0.05, normalize="none", montage="ring8")
    assert ds.window == 200 and ds.ratio == 10 and ds.stride == 5
    for i in (0, 1, len(ds) - 1):
        item = ds[i]
        last_sample = item["emg"][0, -1]
        t = item["kin"][JOINT_INDEX["index_mcp_flex"]]
        # The target is the hand at the window's last EMG sample.
        assert last_sample / 1000.0 == pytest.approx(t, abs=1e-4)
        assert item["emg"].shape == (8, 200)
        assert np.diff(item["emg"][0]).tolist() == [1.0] * 199  # contiguous samples
    assert ds[0]["mask"].sum() == 1


def test_montage_and_fraction(built):
    root, _ = built
    assert EMGWindows(root, montage="first4")[0]["emg"].shape[0] == 4
    small = EMGWindows(root, fraction=0.5)
    assert len(small.subjects) == 2
    assert set(small.subjects) <= set(EMGWindows(root, fraction=0.75).subjects)


def test_split_is_subject_disjoint(built):
    root, _ = built
    train, val, test = EMGWindows(root).split_subjects(val=0.25, test=0.25)
    sets = [set(d.subjects) for d in (train, val, test)]
    assert all(sets) and not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_lag_shift(tmp_path, monkeypatch):
    src = Synthetic()
    src.kin_lag_s = 0.1  # labels describe the hand 100 ms after their timestamp
    monkeypatch.setattr(build, "get_source", lambda name: src)
    cfg = BuildConfig(fs=1000, label_fs=100, highpass=None, notch=False, calib_s=1)
    process_session(src.sessions("s01")[0], tmp_path, tmp_path, cfg, "h")
    d = tmp_path / "synthetic/s01/a"
    emg, kin = np.load(d / "emg.npy"), np.load(d / "kin.npy")
    k = 50
    # After the shift, EMG sample k * r is the instant label k describes: its timestamp + lag.
    assert emg[k * 10, 0] / 1000.0 == pytest.approx(kin[k, JOINT_INDEX["index_mcp_flex"]] + 0.1, abs=1e-6)


def test_rebuild_is_skipped_when_unchanged(built):
    root, _ = built
    cfg = BuildConfig(fs=1000, label_fs=100, highpass=None, notch=False, calib_s=1)
    assert store.is_built(root / "synthetic/s01/a", build_hash(cfg, "x"))
    assert not store.is_built(root / "synthetic/s01/a", build_hash(cfg, "other-code"))


def test_skips_sessions_below_target_rate(tmp_path, monkeypatch):
    src = Synthetic()
    monkeypatch.setattr(build, "get_source", lambda name: src)
    cfg = BuildConfig(fs=2000, label_fs=100)
    assert process_session(src.sessions("s01")[0], tmp_path, tmp_path, cfg, "h")["status"] == "skipped"
    assert not Path(tmp_path / "synthetic").exists()


def test_subject_selection_is_nested_and_stable():
    subs = [f"s{i:02d}" for i in range(40)]
    a, b, c = (select_subjects("ds", subs, f) for f in (0.1, 0.5, 1.0))
    assert len(a) == 4 and len(b) == 20 and c == subs
    assert set(a) <= set(b) and a == select_subjects("ds", subs, 0.1)
    assert len(select_subjects("ds", subs[:3], 0.01)) == 1
