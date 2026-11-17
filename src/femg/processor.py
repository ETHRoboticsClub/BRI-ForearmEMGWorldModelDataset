import math

import numpy as np

from femg.electrodes import ElectrodeLayout
from femg.hand.joints import N_JOINTS
from femg.normalize import Normalizer
from femg.signal_utils import causal_filter, pick_nearest, resample, resample_kin
from femg.virtual_array import VirtualArray


class Processor:
    """Two stages, each a list of steps run in order:
    - `process`: raw session (native rates, volts, canonical joints) -> interim (data/interim): EMG at fs, filtered;
      joints at label_fs, aligned to the EMG; the normalization anchors (meta emg_stats, kin_stats);
    - `prepare`: interim -> processed (data/processed), what training reads: the EMG on the virtual band, each channel
      / its own active level; the joints in 0-1."""

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
        }

    def prepare_settings(self) -> dict:
        """The settings of a processed session: the interim ones and the virtual array's."""
        return {**self.settings(), "virtual_array": self.virtual_array.settings()}

    def process(self, session, dataset):
        session = self.resample_emg(session)  # before filtering: fewer samples to filter
        session = self.filter_emg(session, dataset.config.mains_hz)
        session = self.resample_labels(session)
        session = self.align_labels(session, dataset.config.kin_lag_s)
        session = self.drop_settling(session)
        session = self.add_stats(session)
        return session

    def prepare(self, emg: np.ndarray, kin: np.ndarray, meta: dict) -> tuple[dict[str, np.ndarray], dict]:
        """An interim session (emg (T, C) volts, kin (T / r, J) rad, its meta) -> the processed arrays (virtual,
        labels) and meta."""
        virtual, info = self.to_virtual(emg, meta)
        labels = self.normalizer.kin(kin, meta["kin_stats"]).astype(np.float32)
        return {"virtual": virtual, "labels": labels}, {**meta, **info, "processor": self.prepare_settings()}

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
        """The normalization anchors: meta emg_stats (per channel), kin_stats (per joint)."""
        session.info["emg_stats"] = self.normalizer.emg_stats(session.emg)
        session.info["kin_stats"] = self.normalizer.kin_stats(session.kin)
        return session

    def to_virtual(self, emg: np.ndarray, meta: dict) -> tuple[np.ndarray, dict]:
        """The EMG / active level on the virtual band (T, H * A), each virtual channel / its own active level (meta
        virtual_stats); virtual electrodes no real one covers stay 0."""
        positions = self.virtual_array.inputs(ElectrodeLayout.from_meta(meta))
        virtual = self.virtual_array(self.normalizer.emg(emg, meta["emg_stats"]), positions).reshape(len(emg), -1)
        stats = self.normalizer.emg_stats(virtual)
        info = {
            "virtual_stats": stats,
            "virtual_covered": self.virtual_array.covered(positions).ravel().tolist(),
            "n_virtual": virtual.shape[1],
        }
        return self.normalizer.emg(virtual, stats), info
