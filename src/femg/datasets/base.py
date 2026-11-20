import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from femg.download import RemoteFile, is_downloaded
from femg.electrodes import ElectrodeLayout
from femg.gestures import check_cues


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
    # where each continuous recording starts (kin samples, label frames once processed): trials joined end to end
    # are not continuous in time, so nothing is filtered across a join and training windows must not span one
    segments: list[int] = field(default_factory=lambda: [0])


CONFIG_DIR = Path(__file__).with_name("config")


@dataclass(frozen=True)
class DatasetConfig:
    """What a dataset is, read from config/<name>.yaml: source, recording, electrodes and notes."""

    name: str
    description: str
    citation: str
    license: str
    url: str
    device: str
    device_group: str  # recording setups that give alike signals (myo, delsys_trigno, ...): the first level of sampling
    native_fs: float
    effective_fs: float  # before any upsampling by the authors
    mains_hz: float | None
    electrodes: ElectrodeLayout
    hand: str = "right"
    emg_unit_scale: float = 1.0  # file units -> volts
    kin_lag_s: float = 0.0  # kin[t] shows the hand at t + kin_lag_s
    kin_source: str = "native"  # where the labels come from: the dataset's own files, or another (ninapro_db2: db9)
    # native cue id -> (its name in the dataset, shared gesture of src/femg/config/gestures.yaml or None); 0 is rest
    cues: dict[int, tuple[str, str | None]] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)  # what was checked against the real files, and how

    @classmethod
    def load(cls, name: str) -> "DatasetConfig":
        spec = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
        electrodes = ElectrodeLayout.from_config(spec.pop("electrodes"), spec.pop("montages", None))
        numbers = {k: float(spec[k]) for k in ("native_fs", "effective_fs", "emg_unit_scale", "kin_lag_s") if k in spec}
        cues = load_cues(spec.pop("cues", None) or {}, name)
        return cls(name=name, electrodes=electrodes, cues=cues, **(spec | numbers))

    def metadata(self) -> dict:
        """The meta.json fields of every session of the dataset."""
        return {
            "description": self.description,
            "citation": self.citation,
            "license": self.license,
            "url": self.url,
            "device": self.device,
            "native_fs": self.native_fs,
            "effective_fs": self.effective_fs,
            "mains_hz": self.mains_hz,
            "hand": self.hand,
            **self.electrodes.to_meta(),
            "kin_lag_s": self.kin_lag_s,
            "kin_source": self.kin_source,
            "stimulus_names": {cue: label for cue, (label, _) in self.cues.items()},
            "notes": self.notes,
        }


def load_cues(spec: dict | str, dataset: str) -> dict[int, tuple[str, str | None]]:
    """A dataset's `cues:`, inline ({id: [name, gesture]}) or the name of a table in config/cues/ shared by datasets
    with the same protocol."""
    if isinstance(spec, str):
        spec = yaml.safe_load((CONFIG_DIR / "cues" / f"{spec}.yaml").read_text())
    cues = {}
    for cue, value in spec.items():
        label, gesture = value
        cues[int(cue)] = (str(label), gesture)
    check_cues(cues, dataset)
    return cues


class Dataset:
    """Reads one dataset's files. What the dataset is (rates, electrodes, ...) is in `self.config`."""

    name = ""

    def __init__(self):
        self.config = DatasetConfig.load(self.name)

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

    def condition(self, subject: str) -> str:
        """intact, amputee_transradial, ... without reading the files (the stratum of `splits`)."""
        return "intact"

    def splits(self, test_percent: float, seed: int) -> dict[str, str]:
        """Subject -> train / test, stratified by condition: test_percent of each condition's subjects (rounded up, at
        least one when the condition has two or more, none when it has one), drawn with a seed of its own per dataset.
        Drawn from every subject, so a subject keeps its split whatever `percent` is built. A dataset with an official
        split overrides this."""
        subjects = self.subjects()
        rng = np.random.default_rng([seed, zlib.crc32(self.name.encode())])
        test = set()
        for condition in sorted({self.condition(s) for s in subjects}):
            group = [s for s in subjects if self.condition(s) == condition]
            n_test = math.ceil(len(group) * test_percent / 100) if len(group) > 1 and test_percent > 0 else 0
            test |= {group[i] for i in rng.permutation(len(group))[:n_test]}
        return {s: "test" if s in test else "train" for s in subjects}


def concat_trials(trials, emg_per_label: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Joins short trials (emg (Te, C), kin (Tk, J), stimulus (Tk,)) end to end, each cut to whole kin samples
    (Te = emg_per_label * Tk). Also returns where every trial starts, in kin samples: the Session's segments."""
    emgs, kins, stimuli, starts, at = [], [], [], [], 0
    for emg, kin, stimulus in trials:
        n = min(len(emg) // emg_per_label, len(kin), len(stimulus))
        emgs.append(emg[: n * emg_per_label])
        kins.append(kin[:n])
        stimuli.append(stimulus[:n])
        starts.append(at)
        at += n
    return np.concatenate(emgs), np.concatenate(kins), np.concatenate(stimuli), starts


def first_percent(subjects: list[str], percent: float) -> list[str]:
    """At least one subject. Nested: the 10% subjects are also in the 30%."""
    if not 0 < percent <= 100:
        raise ValueError(f"percent must be in (0, 100], got {percent}")
    n = max(1, math.ceil(len(subjects) * percent / 100))
    return subjects[:n]
