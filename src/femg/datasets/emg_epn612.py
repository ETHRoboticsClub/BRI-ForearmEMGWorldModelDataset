import json
from pathlib import Path

import numpy as np

from femg.datasets.base import Dataset, Session, concat_trials
from femg.download import RemoteFile

URL = "https://zenodo.org/api/records/4421500/files/EMG-EPN612%20Dataset.zip/content"
ARCHIVE_ROOT = "EMG-EPN612 Dataset"
GROUPS = {"tr": "trainingJSON", "te": "testingJSON"}  # the dataset's two groups of 306 users: its official split
USERS = 306
# the file's samples -> session. samples1: "trainingSamples" (labelled in both groups); samples2: "testingSamples"
# (labelled in the training group, labels withheld in the testing group: EMG only)
SESSIONS = {"samples1": "trainingSamples", "samples2": "testingSamples"}
# generalInfo.myoPredictionLabel of the files: the cue ids (cues: in config/emg_epn612.yaml)
CUES = {"noGesture": 0, "fist": 1, "waveIn": 2, "waveOut": 3, "open": 4, "pinch": 5}


class EMGEPN612(Dataset):
    name = "emg_epn612"

    def subjects(self) -> list[str]:
        return [f"{group}{i:03d}" for group in GROUPS for i in range(1, USERS + 1)]

    def sessions(self, subject: str) -> list[str]:
        return list(SESSIONS)

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        """The user's JSON only, read out of the 5.5 GB zip (not its photo.png)."""
        group, user = GROUPS[subject[:2]], int(subject[2:])
        member = f"{ARCHIVE_ROOT}/{group}/user{user}/user{user}.json"
        return [RemoteFile(url=URL, name=file_name(subject), member=member)]

    def splits(self, test_percent: float, seed: int) -> dict[str, str]:
        """The official split: the testing group's users are the test subjects (test_percent does not apply)."""
        return {s: "test" if s.startswith("te") else "train" for s in self.subjects()}

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        """The 150 samples (5 s each) of one half, joined, each its own segment. A gesture sample is its cue where
        groundTruth marks the movement and rest around it; a noGesture sample rest throughout; unlabelled: -1."""
        data = json.loads((raw_dir / file_name(subject)).read_text())
        fs = data["generalInfo"]["samplingFrequencyInHertz"]
        if fs != self.config.native_fs:
            raise ValueError(f"{subject}: {fs} Hz")
        samples = data[SESSIONS[session]]
        labelled = all("gestureName" in s for s in samples.values())
        trials = []
        for key in sorted(samples, key=lambda k: int(k.split("_")[1])):
            sample = samples[key]
            emg = np.array([sample["emg"][f"ch{i}"] for i in range(1, 9)], np.float32).T
            stimulus = np.full(len(emg), -1, np.int16)
            if labelled:
                cue = CUES[sample["gestureName"]]
                stimulus[:] = 0
                if cue:
                    stimulus[np.asarray(sample["groundTruth"][: len(emg)], bool)] = cue
            trials.append((emg * np.float32(self.config.emg_unit_scale), np.zeros((len(emg), 0)), stimulus))
        emg, _, stimulus, starts = concat_trials(trials, 1)
        info = {
            # the file's flag, a string ("False" in the users opened); what damage it means is not documented
            "condition": "arm_damage" if str(data["userInfo"].get("ArmDamage")).lower() == "true" else "intact",
            "label_source": "cue" if labelled else "none",
            "kin_units": None,
            "group": GROUPS[subject[:2]],
            "myo_from_elbow_cm": data["userInfo"].get("distanceFromElbowToMyoInCm"),
            "task": "5 gestures and relaxed, 5 s per sample, 25 each, samples joined",
        }  # nothing else of userInfo (age, gender, ethnic group, date) is copied
        return Session(subject, session, emg=emg, fs=fs, kin=None, kin_fs=fs, stimulus=stimulus, info=info,
                       segments=starts)  # fmt: skip


def file_name(subject: str) -> str:
    return f"{GROUPS[subject[:2]]}_user{int(subject[2:])}.json"
