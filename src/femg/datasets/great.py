from pathlib import Path

import numpy as np

from femg.datasets.base import Dataset, Session, concat_trials
from femg.datasets.ninapro import CYBERGLOVE_18, glove_to_canonical
from femg.download import RemoteFile

# Dryad (doi:10.5061/dryad.8sf7m0czv) only serves one 3.8 GB zip and refuses scripted downloads. The same files are
# in the authors' GitHub repository, one folder per participant, day and block; the commit is pinned.
REPO = "https://raw.githubusercontent.com/MoveR-Digital-Health-and-Care-Hub/posture_dataset_collection"
COMMIT = "d1a7bdcc15075516262216c3c1c223ccbe5bea1a"
FILES = ["emg_data.hdf5", "glove_data.hdf5", "trials.csv"]  # finger_data.hdf5 (5 finger positions) not used
FS = 2000.0
# trials.csv grasp ids (posture_data_collection.py of the repository); 5 is rest, stored as stimulus 0
GRASPS = {1: "lateral", 2: "open", 3: "power", 4: "pointer", 5: "rest", 6: "tripod"}
REST = 5


class GREAT(Dataset):
    """6 grasps held at 5 arm positions: the dataset to learn invariance to arm posture."""

    name = "great"
    citation = "GREAT, Scientific Data 2024, doi:10.1038/s41597-024-04296-8, doi:10.5061/dryad.8sf7m0czv"
    license = "CC0-1.0"
    url = "https://datadryad.org/dataset/doi:10.5061/dryad.8sf7m0czv"
    device = "16 Delsys Trigno in two rows of 8 around the forearm"
    electrode_type = "bipolar"
    native_fs = FS
    effective_fs = FS
    mains_hz = 50.0  # recorded in the UK
    montages = {"ring8": list(range(8))}  # assumed: first row = channels 1-8 (not documented)
    verified = {
        "files": "participant1_day1_block1 opened: emg_data (150 trials, each (16, ~10000) float32 V, std 7-40 uV), "
        "glove_data (150, (18, same T) float64, upsampled to 2 kHz), trials.csv (target_position, grasp per trial)",
        "emg_preprocessing": "recording_parameters.txt: 4th-order 20-450 Hz Butterworth applied while recording",
        "glove": "uncalibrated (abduction sensors sit near -60); MCP/PIP pairs correlate 0.61-0.96 -> CYBERGLOVE_18",
        "ring8": "row membership of channels not documented",
        "glove_hand": "assumed the EMG hand (intact participants)",
        "glove_signs": "p1 d1b1, normalized medians: open > rest on the 3 spreads (+0.45..+0.84), power raises thumb "
        "roll (+0.67) and lowers thumb abduction (-0.81) vs open: canonical as is (unlike the NinaPro raw glove); "
        "wrist_dev not checked (no wrist movement)",
    }

    def subjects(self) -> list[str]:
        return [f"p{i}" for i in range(1, 9)]

    def sessions(self, subject: str) -> list[str]:
        return [f"day{d}_block{b}" for d in (1, 2) for b in (1, 2)]

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        i = subject[1:]
        return [
            RemoteFile(
                url=f"{REPO}/{COMMIT}/data/participant_{i}/participant{i}_{session}/{file}",
                name=f"{subject}_{session}_{file}",
            )
            for session in self.sessions(subject)
            for file in FILES
        ]

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        """The 150 trials of a block (5 s each, 3 s apart, not recorded in between) joined in order."""
        import h5py
        import pandas as pd

        prefix = raw_dir / f"{subject}_{session}_"
        trials_table = pd.read_csv(f"{prefix}trials.csv", index_col=0)
        trials = []
        with h5py.File(f"{prefix}emg_data.hdf5") as emg_file, h5py.File(f"{prefix}glove_data.hdf5") as glove_file:
            for row in trials_table.itertuples():
                key = str(row.row_number)
                emg = np.asarray(emg_file[key], np.float32).T
                kin = glove_to_canonical(np.asarray(glove_file[key], np.float32).T, CYBERGLOVE_18)
                stimulus = 0 if row.grasp == REST else row.grasp
                trials.append((emg, kin, np.full(len(kin), stimulus, np.int16)))

        emg, kin, stimulus, starts = concat_trials(trials, 1)
        info = {
            "condition": "intact",
            "label_source": "measured",
            "kin_units": "rad_uncalibrated",
            "task": "5 grasps + rest held at 5 arm positions, trials joined",
            "stimulus_names": {k: v for k, v in GRASPS.items() if k != REST},
            "trial_starts_s": [s / FS for s in starts],
            "trial_arm_positions": trials_table["target_position"].astype(int).tolist(),
        }
        return Session(
            subject, session, emg=np.ascontiguousarray(emg), fs=FS, kin=kin, kin_fs=FS, stimulus=stimulus, info=info
        )
