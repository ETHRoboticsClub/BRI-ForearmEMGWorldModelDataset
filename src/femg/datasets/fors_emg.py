from pathlib import Path

import numpy as np

from femg.datasets.base import Dataset, Session, concat_trials
from femg.download import KAGGLE, RemoteFile

HANDLE = "ummerummanchaity/fors-emg-a-novel-semg-dataset"
ROOT = Path("FORS-EMG Dataset") / "FORS-EMG Dataset" / "FORS-EMG"  # inside the extracted archive
# session -> the orientation folder (its case varies between subjects)
ORIENTATIONS = {"pronation": "pronation", "neutral": "rest", "supination": "supination"}
# file name -> cue id, the order of gestures_name.txt (cues: in config/fors_emg.yaml)
GESTURES = [
    "Thumb_UP", "Index", "Right_Angle", "Peace", "Index_Little", "Thumb_Little",
    "Hand_Close", "Hand_Open", "Wrist_Extension", "Wrist_Flexion", "Ulner_Deviation", "Radial_Deviation",
]  # fmt: skip
TRIALS = range(1, 6)
REST_S, ONSET_S = 0.5, 1.0  # each trial: rest until 0.5 s, the movement until 1 s, then held (config notes)


class FORSEMG(Dataset):
    name = "fors_emg"

    def subjects(self) -> list[str]:
        return [f"s{i:02d}" for i in range(1, 20)]

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        return [RemoteFile(url=f"{KAGGLE}{HANDLE}", name=ROOT.parts[0])]  # one archive for every subject

    def sessions(self, subject: str) -> list[str]:
        return list(ORIENTATIONS)

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        """The 60 trials (12 gestures x 5) of one forearm orientation, joined; each its own segment."""
        import scipy.io

        folder = orientation_folder(raw_dir / ROOT / f"Subject{int(subject[1:])}", ORIENTATIONS[session])
        fs = self.config.native_fs
        trials = []
        for cue, gesture in enumerate(GESTURES, 1):
            for trial in TRIALS:
                emg = scipy.io.loadmat(folder / f"{gesture}-{trial}.mat")["value"].T.astype(np.float32)
                stimulus = np.full(len(emg), cue, np.int16)
                stimulus[: round(ONSET_S * fs)] = -1
                stimulus[: round(REST_S * fs)] = 0
                trials.append((emg * np.float32(self.config.emg_unit_scale), np.zeros((len(emg), 0)), stimulus))
        emg, _, stimulus, starts = concat_trials(trials, 1)
        info = {
            "condition": "intact",
            "label_source": "cue",
            "kin_units": None,
            "forearm_orientation": session,
            "task": "12 gestures held 7 s, 5 trials each, forearm in one orientation, trials joined",
        }
        return Session(subject, session, emg=emg, fs=fs, kin=None, kin_fs=fs, stimulus=stimulus, info=info,
                       segments=starts)  # fmt: skip


def orientation_folder(subject_folder: Path, name: str) -> Path:
    for folder in subject_folder.iterdir():
        if folder.is_dir() and folder.name.lower() == name:
            return folder
    raise FileNotFoundError(f"{subject_folder}: no {name} folder")
