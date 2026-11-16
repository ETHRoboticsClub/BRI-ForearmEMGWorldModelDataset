from pathlib import Path

import numpy as np

from femg.datasets.base import Dataset, Session
from femg.download import RemoteFile
from femg.hand.joints import FINGER_FLEXIONS, JOINT_INDEX, N_JOINTS

ZENODO = "https://zenodo.org/records/19453843/files"
MD5 = {
    1: "1e8bcd72c41b1d7865ffdbba00fc7c0b",
    2: "49cd84e2fed21653d20e31e6ce91e0d9",
    3: "cf882cbbee92630ee81223add9bd8fce",
    4: "f87039ce5ca7d9f98b4f7e874da7cbae",
    5: "5579a3895988bcaa3644937ea13f7278",
    6: "5f35da3570314670e3310d30d9b042ff",
    7: "dc77e20cc58a08d1c0d208339ff46770",
    8: "e6a88d3bfdb17d38fdacc936a64e9fc6",
    9: "d16b1d1f588976c6aa525609ff64b24b",
    10: "ad49207c48703a4e621b3263c61e1a7e",
    11: "2eebae7cdda63633e34a99d774897332",
    12: "41d7ca636c32aeab2eee4bb523351764",
    13: "c7a6f19e5dc9c9a08f1f086e7e1fbaf9",
    14: "8f99a6f47168da8be7271f121fd31927",
    15: "a797602f00403d5703771e258d98e3fe",
    16: "3b78d13a37bbbc310423e1c56208186d",
    17: "3cb9b3e1e9d486ecfce1a541522e0940",
    18: "cb8dac87caad73be0f37c78e731817b2",
    19: "b1785d48d0e1e7886bd8d383249e0bc3",
    20: "3e64b600e7e7617b19666ec2ad8b997c",
}

# "Angle 1..15" (MediaPipe, degrees, 180 = straight): the 15 finger flexions, thumb to pinky, base to tip. Order not
# documented: inferred from correlations on Subject_01. To confirm with figures/figure_angles.png of the authors' repo.
ANGLES = FINGER_FLEXIONS


class EMGFK(Dataset):
    name = "emg_fk"

    def subjects(self) -> list[str]:
        return [f"s{i:02d}" for i in range(1, 21)]

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        i = int(subject[1:])
        name = f"Subject_{i:02d}.fif"
        return [RemoteFile(url=f"{ZENODO}/{name}", name=name, md5=MD5[i])]

    def sessions(self, subject: str) -> list[str]:
        return ["free"]

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        import mne

        path = raw_dir / f"Subject_{int(subject[1:]):02d}.fif"
        raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
        fs = raw.info["sfreq"]
        emg = raw.get_data(picks=[f"EMG {i}" for i in range(1, 9)]).T.astype(np.float32) * self.config.emg_unit_scale
        angles = raw.get_data(picks=[f"Angle {i}" for i in range(1, 16)]).T.astype(np.float32)

        kin = np.full((len(angles), N_JOINTS), np.nan, np.float32)
        for column, joint in enumerate(ANGLES):
            kin[:, JOINT_INDEX[joint]] = np.deg2rad(180.0 - angles[:, column])

        info = {"condition": "intact", "label_source": "measured", "kin_units": "rad", "task": "free finger motion"}
        return Session(subject, session, emg=np.ascontiguousarray(emg), fs=fs, kin=kin, kin_fs=fs, info=info)
