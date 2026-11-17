import math

import numpy as np

from femg.hand.joints import N_JOINTS
from femg.normalize import Normalizer
from femg.signal_utils import causal_filter, pick_nearest, resample, resample_kin
from femg.virtual_array import VirtualArray


class Processor:
    """Raw session (native rates, volts, canonical joints) -> training-ready session: EMG at fs, filtered; labels at
    label_fs, aligned to the EMG; the normalization anchors; the EMG on the virtual band, normalized; the joints in
    0-1. `process` runs every step in order."""

    def __init__(
        self,
        fs=500,
        label_fs=100,
        highpass=10,
        notch=True,
        calib_s=30,
        rms_window_s=0.1,
        emg_anchors=(10, 95),
        kin_anchors=(2, 98),
        kin_lowpass=6,
        kin_max_repeat_s=0.1,
        kin_max_gap_s=0.1,
        settle_s=0.2,
        virtual_array: VirtualArray | None = None,
    ):
        if fs % label_fs != 0:
            raise ValueError(f"fs ({fs}) must be a multiple of label_fs ({label_fs})")
        self.fs = fs
        self.label_fs = label_fs
        self.highpass = highpass
        self.notch = notch
        self.normalizer = Normalizer(fs, label_fs, rms_window_s, calib_s, tuple(emg_anchors), tuple(kin_anchors))
        self.virtual_array = virtual_array or VirtualArray.load()
        self.kin_lowpass = kin_lowpass
        self.kin_max_repeat_s = kin_max_repeat_s
        self.kin_max_gap_s = kin_max_gap_s
        self.settle_s = settle_s
        self.emg_per_label = int(fs // label_fs)

    def settings(self) -> dict:
        return {
            "processor": type(self).__name__,
            "fs": self.fs,
            "label_fs": self.label_fs,
            "highpass": self.highpass,
            "notch": self.notch,
            "calib_s": self.normalizer.calib_s,
            "rms_window_s": self.normalizer.rms_window_s,
            "emg_anchors": list(self.normalizer.emg_anchors),
            "kin_anchors": list(self.normalizer.kin_anchors),
            "kin_lowpass": self.kin_lowpass,
            "kin_max_repeat_s": self.kin_max_repeat_s,
            "kin_max_gap_s": self.kin_max_gap_s,
            "settle_s": self.settle_s,
            "virtual_array": self.virtual_array.settings(),
        }

    def process(self, session, dataset):
        session = self.resample_emg(session)  # before filtering: fewer samples to filter
        session = self.filter_emg(session, dataset.config.mains_hz)
        session = self.resample_labels(session)
        session = self.align_labels(session, dataset.config.kin_lag_s)
        session = self.drop_settling(session)
        session = self.normalize(session)
        session = self.to_virtual(session, self.virtual_array.inputs(dataset.config.electrodes))
        return session

    def resample_emg(self, session):
        session.emg = resample(session.emg, session.fs, self.fs)
        session.fs = self.fs
        return session

    def filter_emg(self, session, mains_hz):
        mains = mains_hz if self.notch else None
        session.emg = causal_filter(session.emg, self.fs, self.highpass, mains)
        return session

    def resample_labels(self, session):
        if session.kin is None:
            session.kin = np.full((len(session.emg) // self.emg_per_label, N_JOINTS), np.nan, np.float32)
        else:
            session.kin = resample_kin(
                session.kin,
                session.kin_fs,
                self.label_fs,
                session.segments,
                self.kin_lowpass,
                self.kin_max_gap_s,
                self.kin_max_repeat_s,
            )

        if session.stimulus is None:
            session.stimulus = np.full(len(session.kin), -1, np.int16)
        else:
            session.stimulus = pick_nearest(session.stimulus, session.kin_fs, self.label_fs).astype(np.int16)

        session.segments = [math.ceil(start * self.label_fs / session.kin_fs) for start in session.segments]
        session.kin_fs = self.label_fs
        return session

    def align_labels(self, session, kin_lag_s):
        lag = round(kin_lag_s * self.label_fs)  # > 0: kin[k] shows the hand kin_lag_s after its timestamp
        if lag and len(session.segments) > 1:
            raise ValueError("kin_lag_s would shift the joins of the EMG and the labels apart")
        if lag > 0:
            session.emg = session.emg[lag * self.emg_per_label :]
        elif lag < 0:
            session.kin = session.kin[-lag:]
            session.stimulus = session.stimulus[-lag:]

        n = min(len(session.emg) // self.emg_per_label, len(session.kin), len(session.stimulus))
        session.emg = session.emg[: n * self.emg_per_label]
        session.kin = session.kin[:n]
        session.stimulus = session.stimulus[:n]
        session.segments = [start for start in session.segments if start < n]
        return session

    def drop_settling(self, session):
        """No label for the first settle_s of every segment: the causal high-pass rings after each start or join."""
        n = round(self.settle_s * self.label_fs)
        for start in session.segments:
            session.kin[start : start + n] = np.nan
        return session

    def normalize(self, session):
        """The anchors (meta emg_stats, kin_stats) and the joints in 0-1 (session.labels)."""
        session.info["emg_stats"] = self.normalizer.emg_stats(session.emg)
        session.info["kin_stats"] = self.normalizer.kin_stats(session.kin)
        session.labels = self.normalizer.kin(session.kin, session.info["kin_stats"]).astype(np.float32)
        return session

    def to_virtual(self, session, positions):
        """The EMG / active level on the virtual band, each virtual channel / its own active level (session.virtual,
        (T, H * A)). Its anchors: meta virtual_stats; virtual electrodes no real one covers stay 0."""
        emg = self.normalizer.emg(session.emg, session.info["emg_stats"])
        virtual = self.virtual_array(emg, positions).reshape(len(emg), -1)
        session.info["virtual_stats"] = self.normalizer.emg_stats(virtual)
        session.info["virtual_covered"] = self.virtual_array.covered(positions).ravel().tolist()
        session.virtual = self.normalizer.emg(virtual, session.info["virtual_stats"])
        return session
