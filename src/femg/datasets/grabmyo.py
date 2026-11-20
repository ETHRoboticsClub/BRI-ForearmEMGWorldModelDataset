from pathlib import Path

import numpy as np

from femg.datasets.base import Dataset, Session, concat_trials
from femg.download import RemoteFile

# PhysioNet's open S3 mirror of the project: plain https, one file at a time
URL = "https://physionet-open.s3.amazonaws.com/grabmyo/1.1.0"
DAYS = (1, 2, 3)
GESTURES = range(1, 18)  # 16 gestures, then rest (MotionSequence.txt)
REST = 17  # stored as cue 0
TRIALS = range(1, 8)
# 32 columns: F1-F16 (forearm rings 1, 2), U1, W1-W6 (ring 3), U2, U3, W7-W12 (ring 4), U4. U: unused, all zero
CHANNELS = [*range(0, 16), *range(17, 23), *range(25, 31)]


def record_name(day: int, subject: str, gesture: int, trial: int) -> str:
    return f"session{day}_participant{int(subject[1:])}_gesture{gesture}_trial{trial}"


class GRABMyo(Dataset):
    name = "grabmyo"

    def subjects(self) -> list[str]:
        return [f"s{i:02d}" for i in range(1, 44)]

    def sessions(self, subject: str) -> list[str]:
        return [f"day{d}" for d in DAYS]

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        files = []
        for day in DAYS:
            folder = f"Session{day}/session{day}_participant{int(subject[1:])}"
            for gesture in GESTURES:
                for trial in TRIALS:
                    name = record_name(day, subject, gesture, trial)
                    files += [RemoteFile(url=f"{URL}/{folder}/{name}.{e}", name=f"{name}.{e}") for e in ("hea", "dat")]
        return files

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        """The 119 trials of one day (17 gestures incl. rest x 7, 5 s each), joined; each its own segment. The gesture
        is held through the whole trial (config notes trial_timing)."""
        import wfdb

        day = int(session.removeprefix("day"))
        trials = []
        for gesture in GESTURES:
            cue = 0 if gesture == REST else gesture
            for trial in TRIALS:
                record = wfdb.rdrecord(str(raw_dir / record_name(day, subject, gesture, trial)))
                if set(record.units) != {"mV"} or record.fs != self.config.native_fs:
                    raise ValueError(f"{record.record_name}: units {set(record.units)}, fs {record.fs}")
                emg = record.p_signal[:, CHANNELS].astype(np.float32) * np.float32(self.config.emg_unit_scale)
                trials.append((emg, np.zeros((len(emg), 0)), np.full(len(emg), cue, np.int16)))
        emg, _, stimulus, starts = concat_trials(trials, 1)
        info = {
            "condition": "intact",
            "label_source": "cue",
            "kin_units": None,
            "day": day,
            "task": "16 gestures and rest held 5 s, 7 trials each, trials joined",
        }
        fs = self.config.native_fs
        return Session(subject, session, emg=emg, fs=fs, kin=None, kin_fs=fs, stimulus=stimulus, info=info,
                       segments=starts)  # fmt: skip
