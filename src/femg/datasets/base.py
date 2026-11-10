"""The Dataset base class. Every public dataset is one subclass (emg_fk.py is a short example to copy)."""

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from femg.download import RemoteFile, is_downloaded


@dataclass
class Session:
    """One continuous recording. Arrays are time-major: one row per time sample.

    A dataset's `load_session` returns it in the dataset's native rates; the Processor converts it to the
    common rates in place.
    """

    subject: str  # e.g. "s01"
    name: str  # e.g. "A1"
    emg: np.ndarray  # (T, C) float32, volts
    fs: float  # EMG sampling rate (Hz)
    kin: np.ndarray | None  # (Tk, N_JOINTS) float32 canonical joints in radians, NaN = not measured
    kin_fs: float  # sampling rate of kin and stimulus (Hz)
    stimulus: np.ndarray | None = None  # (Tk,) instructed movement id, 0 = rest
    info: dict = field(default_factory=dict)  # condition, label_source, kin_units, ... (copied to meta.json)


class Dataset:
    """One public dataset.

    To add a dataset: subclass this, fill in the metadata and implement the four methods. The metadata is copied
    into every session's meta.json, so keep it factual and say where each value comes from. Unknown = None.
    """

    name = ""  # folder name in data/raw and data/processed
    citation = ""
    license = ""
    url = ""
    device = ""
    electrode_type = ""  # "bipolar" (double-differential) or "monopolar" (common reference)
    native_fs = 0.0  # EMG rate of the files
    effective_fs = 0.0  # EMG rate before any upsampling done by the dataset authors
    mains_hz = None  # power-line frequency where it was recorded (50 or 60)
    emg_unit_scale = 1.0  # multiply the file's EMG by this to get volts
    hand = "right"  # hand whose EMG is recorded
    # Channel subsets the loader can pick to get the same channel count across datasets.
    # ring8: 8 electrodes evenly spaced around the forearm. Their rotation is NOT aligned across datasets.
    montages = {}
    kin_lag_s = 0.0  # kin[t] describes the hand at t + kin_lag_s; the processor corrects it
    verified = {}  # what was checked against the real files, and how

    def subjects(self) -> list[str]:
        """All subject ids, e.g. ["s01", "s02", ...]."""
        raise NotImplementedError

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        """The raw files of one subject."""
        raise NotImplementedError

    def sessions(self, subject: str) -> list[str]:
        """Names of the subject's sessions, e.g. ["A1", "A2"]."""
        raise NotImplementedError

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        """Read one session from raw_dir: EMG in volts, kinematics as canonical joints in radians."""
        raise NotImplementedError

    def is_downloaded(self, subject: str, raw_dir: Path) -> bool:
        return all(is_downloaded(f, raw_dir) for f in self.files_to_download(subject))

    def metadata(self) -> dict:
        return {
            "citation": self.citation,
            "license": self.license,
            "url": self.url,
            "device": self.device,
            "electrode_type": self.electrode_type,
            "native_fs": self.native_fs,
            "effective_fs": self.effective_fs,
            "mains_hz": self.mains_hz,
            "hand": self.hand,
            "montages": self.montages,
            "kin_lag_s": self.kin_lag_s,
            "verified": self.verified,
        }


def first_percent(subjects: list[str], percent: float) -> list[str]:
    """The first `percent`% of the subjects (at least one).

    Used by pull_data, build_dataset and the loader, so they all agree. The subsets are nested:
    the subjects of a 10% build are also in the 30% build.
    """
    if not 0 < percent <= 100:
        raise ValueError(f"percent must be in (0, 100], got {percent}")
    n = max(1, math.ceil(len(subjects) * percent / 100))
    return subjects[:n]
