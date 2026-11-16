from functools import cache
from pathlib import Path

import numpy as np

from femg.datasets.base import Dataset, Session, concat_trials
from femg.download import RemoteFile
from femg.hand.joints import JOINT_INDEX, N_JOINTS

ZENODO = "https://zenodo.org/records/3469380/files"
MD5 = {"KIN_MUS_UJI.mat": "4fc49241d6a887f6f1e9dbbcfc6aef12", "RAW_EMG.mat": "c0045aac1480480c265501a7c6f3192c"}
KIN_FS = 100.0

# Kinematic_data columns (paper Table 2) -> canonical joint
ANGLES = {
    "CMC1_A": "thumb_cmc_flex",  # abduction: the thumb lifting in front of the palm
    "CMC1_F": "thumb_cmc_abd",  # flexion, negated (FLIPPED): the thumb going across the palm
    "MCP1_F": "thumb_mcp_flex",
    "IP1_F": "thumb_ip_flex",
    "MCP2-3_A": "index_middle_abd",
    "MCP2_F": "index_mcp_flex",
    "PIP2_F": "index_pip_flex",
    "MCP3_F": "middle_mcp_flex",
    "PIP3_F": "middle_pip_flex",
    "MCP3-4_A": "middle_ring_abd",
    "MCP4_F": "ring_mcp_flex",
    "PIP4_F": "ring_pip_flex",
    "CMC5_F": "palm_arch",
    "MCP4-5_A": "ring_pinky_abd",
    "MCP5_F": "pinky_mcp_flex",
    "PIP5_F": "pinky_pip_flex",
    "WRIST_F": "wrist_flex",
    "WRIST_A": "wrist_dev",
}


FLIPPED = ("thumb_cmc_abd",)


@cache
def load_records(raw_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """(kinematics: one record per subject x ADL x phase, raw EMG: one record per subject x ADL). ~0.5 GB."""
    import scipy.io

    kin = scipy.io.loadmat(Path(raw_dir) / "KIN_MUS_UJI.mat")["EMG_KIN_v4"].ravel()
    raw = scipy.io.loadmat(Path(raw_dir) / "RAW_EMG.mat")["RAW_EMG"].ravel()
    return kin, raw


class KinMusUJI(Dataset):
    name = "kin_mus_uji"

    def subjects(self) -> list[str]:
        return [f"s{i:02d}" for i in range(1, 23)]

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        return [RemoteFile(url=f"{ZENODO}/{name}", name=name, md5=md5) for name, md5 in MD5.items()]

    def sessions(self, subject: str) -> list[str]:
        return ["adl"]

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        """The 26 ADLs of a subject joined, one segment each. Kinematics cover reach -> manipulate -> release;
        before and after (the still periods the authors trimmed from the kinematics, not from the EMG) they are NaN
        (no label) and the stimulus is 0."""
        kin_records, raw_records = load_records(str(raw_dir))
        number = int(subject[1:])
        fs = self.config.native_fs
        ratio = round(fs / KIN_FS)

        trials = []
        for raw in sorted((r for r in raw_records if r["Subject"][0, 0] == number), key=lambda r: r["ADL"][0, 0]):
            adl = int(raw["ADL"][0, 0])
            emg = raw["Raw_EMG"].astype(np.float32) * np.float32(self.config.emg_unit_scale)
            n = len(emg) // ratio
            kin = np.full((n, N_JOINTS), np.nan, np.float32)
            stimulus = np.zeros(n, np.int16)
            for phase in (r for r in kin_records if r["Subject"][0, 0] == number and r["ADL"][0, 0] == adl):
                if phase["Kinematic_data"].shape[1:] != (len(ANGLES),):
                    continue  # 3 empty placeholders (1, 1): s21 ADL25 phase 1, s22 ADL25 phases 2-3
                # EMG sample i is at (i + 1) ms, so label k (EMG sample 10 k) is at 10 k + 1 ms
                k = np.round((phase["time"][:, 0] - 1 / fs) * KIN_FS).astype(int)
                keep = (k >= 0) & (k < n)
                angles = np.deg2rad(phase["Kinematic_data"][keep].astype(np.float32))
                for column, joint in enumerate(ANGLES.values()):
                    sign = -1.0 if joint in FLIPPED else 1.0
                    kin[k[keep], JOINT_INDEX[joint]] = sign * angles[:, column]
                stimulus[k[keep]] = adl
            trials.append((emg, kin, stimulus))

        emg, kin, stimulus, starts = concat_trials(trials, ratio)
        info = {
            "condition": "intact",
            "label_source": "measured",
            "kin_units": "rad",
            "task": "26 activities of daily living (stimulus = ADL id while the hand is recorded)",
        }
        return Session(
            subject,
            session,
            emg=np.ascontiguousarray(emg),
            fs=fs,
            kin=kin,
            kin_fs=KIN_FS,
            stimulus=stimulus,
            info=info,
            segments=starts,
        )
