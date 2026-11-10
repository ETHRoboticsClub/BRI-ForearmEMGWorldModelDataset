import numpy as np

from femg.joints import N_JOINTS
from femg.signal_utils import causal_filter, emg_stats, kin_stats, pick_nearest, resample, resample_with_nans


class Processor:
    def __init__(self, fs=500, label_fs=100, highpass=10, notch=True, calib_s=30):
        if fs % label_fs != 0:
            raise ValueError(f"fs ({fs}) must be a multiple of label_fs ({label_fs})")
        self.fs = fs  # common EMG rate (Hz)
        self.label_fs = label_fs  # common kinematics rate (Hz)
        self.highpass = highpass  # high-pass cut-off (Hz), None = off
        self.notch = notch  # remove mains noise at the dataset's mains_hz
        self.calib_s = calib_s  # seconds used for the calib_rms normalization stat
        self.ratio = int(fs // label_fs)  # EMG samples per label

    def settings(self) -> dict:
        """Stored in meta.json. A session built with different settings is rebuilt."""
        return {
            "processor": type(self).__name__,
            "fs": self.fs,
            "label_fs": self.label_fs,
            "highpass": self.highpass,
            "notch": self.notch,
            "calib_s": self.calib_s,
        }

    def process(self, session, dataset):
        session = self.resample_emg(session)  # first, so the filter runs on fewer samples
        session = self.filter_emg(session, dataset.mains_hz)
        session = self.resample_labels(session)
        session = self.align_labels(session, dataset.kin_lag_s)
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
        """Bring kin and stimulus to label_fs. Missing ones become NaN / -1 so every session has both."""
        if session.kin is None:
            session.kin = np.full((len(session.emg) // self.ratio, N_JOINTS), np.nan, np.float32)
        else:
            session.kin = resample_with_nans(session.kin, session.kin_fs, self.label_fs)

        if session.stimulus is None:
            session.stimulus = np.full(len(session.kin), -1, np.int16)
        else:
            session.stimulus = pick_nearest(session.stimulus, session.kin_fs, self.label_fs).astype(np.int16)

        session.kin_fs = self.label_fs
        return session

    def align_labels(self, session, kin_lag_s):
        """Correct the dataset's lag, then crop so that label k matches EMG sample k * ratio.

        kin_lag_s > 0 means kin[k] describes the hand kin_lag_s after its timestamp, so we drop the start of the EMG.
        """
        lag = round(kin_lag_s * self.label_fs)  # in labels
        if lag > 0:
            session.emg = session.emg[lag * self.ratio :]
        elif lag < 0:
            session.kin = session.kin[-lag:]
            session.stimulus = session.stimulus[-lag:]

        n = min(len(session.emg) // self.ratio, len(session.kin), len(session.stimulus))
        session.emg = session.emg[: n * self.ratio]
        session.kin = session.kin[:n]
        session.stimulus = session.stimulus[:n]
        return session

    def add_stats(self, session):
        session.info["emg_stats"] = emg_stats(session.emg, self.fs, self.calib_s)
        session.info["kin_stats"] = kin_stats(session.kin)
        return session


# Processors selectable with `processor.name` in params.yaml.
PROCESSORS = {
    "default": Processor,
}
