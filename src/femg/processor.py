import math

import numpy as np

from femg.joints import N_JOINTS
from femg.signal_utils import causal_filter, emg_stats, kin_stats, pick_nearest, resample, resample_kin


class Processor:
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
    ):
        if fs % label_fs != 0:
            raise ValueError(f"fs ({fs}) must be a multiple of label_fs ({label_fs})")
        self.fs = fs
        self.label_fs = label_fs
        self.highpass = highpass
        self.notch = notch
        self.calib_s = calib_s
        self.rms_window_s = rms_window_s
        self.emg_anchors = list(emg_anchors)
        self.kin_anchors = list(kin_anchors)
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
            "calib_s": self.calib_s,
            "rms_window_s": self.rms_window_s,
            "emg_anchors": self.emg_anchors,
            "kin_anchors": self.kin_anchors,
            "kin_lowpass": self.kin_lowpass,
            "kin_max_repeat_s": self.kin_max_repeat_s,
            "kin_max_gap_s": self.kin_max_gap_s,
            "settle_s": self.settle_s,
        }

    def process(self, session, dataset):
        session = self.resample_emg(session)  # before filtering: fewer samples to filter
        session = self.filter_emg(session, dataset.mains_hz)
        session = self.resample_labels(session)
        session = self.align_labels(session, dataset.kin_lag_s)
        session = self.drop_settling(session)
        session = self.add_stats(session)
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

    def add_stats(self, session):
        window = max(1, round(self.rms_window_s * self.label_fs))
        session.info["emg_stats"] = emg_stats(
            session.emg, self.fs, self.calib_s, self.emg_per_label, window, self.emg_anchors
        )
        session.info["kin_stats"] = kin_stats(session.kin, self.kin_anchors)
        return session


PROCESSORS = {
    "default": Processor,
}
