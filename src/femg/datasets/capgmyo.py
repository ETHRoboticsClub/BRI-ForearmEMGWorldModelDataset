from pathlib import Path

import numpy as np

from femg.datasets.base import Dataset, Session, concat_trials
from femg.download import RemoteFile

# figshare article 7210397 (CapgMyo DB-a): one zip per subject, its file id
FILE_IDS = [13277105, 13275896, 13275953, 13275962, 13275965, 13275983, 13276016, 13276019, 13276022, 13276934,
            13277147, 13277000, 13277015, 13277027, 13276136, 13276139, 13276142, 13276148]  # fmt: skip
GESTURES = range(1, 9)
TRIALS = range(1, 11)
COLUMNS, ROWS = 16, 8  # 8 modules of 2 x 8 around the forearm: 16 columns around, 8 electrodes along each
# The files hold a column's 8 electrodes as consecutive channels (config notes channel_order); the electrodes config
# lists ring by ring (8 rings of 16): channel `row * 16 + column` <- file channel `column * 8 + row`
RING_ORDER = [column * ROWS + row for row in range(ROWS) for column in range(COLUMNS)]


def mat_name(subject: str, gesture: int, trial: int) -> str:
    return f"{int(subject[1:]):03d}-{gesture:03d}-{trial:03d}.mat"


class CapgMyoDBa(Dataset):
    name = "capgmyo_dba"

    def subjects(self) -> list[str]:
        return [f"s{i:02d}" for i in range(1, len(FILE_IDS) + 1)]

    def sessions(self, subject: str) -> list[str]:
        return ["dba"]

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        i = int(subject[1:])
        members = tuple(mat_name(subject, g, t) for g in GESTURES for t in TRIALS)
        url = f"https://ndownloader.figshare.com/files/{FILE_IDS[i - 1]}"
        return [RemoteFile(url=url, name=f"dba-s{i}.zip", unzip=members)]

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        """The 80 trials (8 gestures x 10, 1 s each as published), joined, each its own segment; no rest."""
        import scipy.io

        trials = []
        for gesture in GESTURES:
            for trial in TRIALS:
                emg = scipy.io.loadmat(raw_dir / mat_name(subject, gesture, trial))["data"][:, RING_ORDER]
                emg = emg.astype(np.float32) * np.float32(self.config.emg_unit_scale)
                trials.append((emg, np.zeros((len(emg), 0)), np.full(len(emg), gesture, np.int16)))
        emg, _, stimulus, starts = concat_trials(trials, 1)
        info = {
            "condition": "intact",
            "label_source": "cue",
            "kin_units": None,
            "task": "8 finger gestures held 3-10 s, 10 trials each; 1 s of each trial published, trials joined",
        }
        fs = self.config.native_fs
        return Session(subject, session, emg=emg, fs=fs, kin=None, kin_fs=fs, stimulus=stimulus, info=info,
                       segments=starts)  # fmt: skip
