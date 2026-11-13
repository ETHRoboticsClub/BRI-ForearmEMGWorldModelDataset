import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from femg.download import RemoteFile, is_downloaded


@dataclass
class Session:
    subject: str
    name: str
    emg: np.ndarray  # (T, C) volts
    fs: float
    kin: np.ndarray | None  # (Tk, N_JOINTS) canonical joints
    kin_fs: float  # rate of kin and stimulus
    stimulus: np.ndarray | None = None  # (Tk,) instructed movement id, 0 = rest
    info: dict = field(default_factory=dict)  # copied to meta.json


class Dataset:
    name = ""
    citation = ""
    license = ""
    url = ""
    device = ""
    electrode_type = ""  # "bipolar" or "monopolar"
    native_fs = 0.0
    effective_fs = 0.0  # before any upsampling by the authors
    mains_hz = None
    emg_unit_scale = 1.0  # file units -> volts
    hand = "right"
    montages = {}  # e.g. ring8: 8 electrodes around the forearm (rotation not aligned across datasets)
    kin_lag_s = 0.0  # kin[t] shows the hand at t + kin_lag_s
    verified = {}  # what was checked against the real files

    def subjects(self) -> list[str]:
        raise NotImplementedError

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        raise NotImplementedError

    def sessions(self, subject: str) -> list[str]:
        raise NotImplementedError

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
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
    """At least one subject. Nested: the 10% subjects are also in the 30%."""
    if not 0 < percent <= 100:
        raise ValueError(f"percent must be in (0, 100], got {percent}")
    n = max(1, math.ceil(len(subjects) * percent / 100))
    return subjects[:n]
