from pathlib import Path

import numpy as np

from femg.datasets.base import Dataset, Session
from femg.download import RemoteFile
from femg.joints import JOINT_INDEX, N_JOINTS

BASE_URL = "https://ninapro.hevs.ch/files"

# CyberGlove II sensor order. Checked on DB8 S1_E1_A3: MCP/PIP pairs correlate at 0.9.
CYBERGLOVE_22 = [
    "thumb_cmc_flex",  # thumb rotation across the palm
    "thumb_mcp_flex",
    "thumb_ip_flex",
    "thumb_cmc_abd",
    "index_mcp_flex",
    "index_pip_flex",
    "index_dip_flex",
    "middle_mcp_flex",
    "middle_pip_flex",
    "middle_dip_flex",
    "index_middle_abd",
    "ring_mcp_flex",
    "ring_pip_flex",
    "ring_dip_flex",
    "middle_ring_abd",
    "pinky_mcp_flex",
    "pinky_pip_flex",
    "pinky_dip_flex",
    "ring_pinky_abd",
    "palm_arch",
    "wrist_flex",
    "wrist_dev",
]
CYBERGLOVE_18 = [name for name in CYBERGLOVE_22 if not name.endswith("_dip_flex")]


def load_mat(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    import scipy.io

    try:
        data = scipy.io.loadmat(path, variable_names=names)
        return {k: data[k] for k in names if k in data}
    except NotImplementedError:  # v7.3 files are HDF5
        import h5py

        with h5py.File(path, "r") as f:
            return {k: np.asarray(f[k]).T for k in names if k in f}


def glove_to_canonical(glove: np.ndarray, sensors: list[str]) -> np.ndarray:
    kin = np.full((len(glove), N_JOINTS), np.nan, np.float32)
    for column, joint in enumerate(sensors):
        kin[:, JOINT_INDEX[joint]] = np.deg2rad(glove[:, column])
    return kin


def read_ninapro_session(path: Path, fs: float, sensors: list[str], subject: str, session: str, info: dict) -> Session:
    data = load_mat(path, ["emg", "glove", "restimulus"])
    n = min(len(v) for v in data.values())  # restimulus is sometimes one sample shorter

    emg = np.ascontiguousarray(data["emg"][:n], np.float32)  # volts (DB2 S1: std ~4e-5)
    kin = None
    if "glove" in data:
        kin = glove_to_canonical(data["glove"][:n].astype(np.float32), sensors)
    stimulus = None
    if "restimulus" in data:
        stimulus = data["restimulus"][:n, 0].astype(np.int16)

    return Session(subject, session, emg=emg, fs=fs, kin=kin, kin_fs=fs, stimulus=stimulus, info=info)


class NinaProDB2(Dataset):
    name = "ninapro_db2"
    citation = "Atzori et al. 2014, Scientific Data 1:140053"
    license = "see https://ninapro.hevs.ch (cite the paper)"
    url = "https://ninapro.hevs.ch/instructions/DB2.html"
    device = (
        "12 Delsys Trigno: 8 evenly spaced around the forearm at the radio-humeral joint + FDS, EDC, biceps, triceps"
    )
    electrode_type = "bipolar"
    native_fs = 2000.0
    effective_fs = 2000.0
    mains_hz = 50.0
    montages = {"ring8": list(range(8))}
    verified = {
        "mat_variables": "DB2_s1 E1/E3 opened: emg (T,12) float32 V, glove (T,22), restimulus; E3 has no glove",
        "glove_order": "manufacturer order; not checked on DB2 data",
    }
    EXERCISES = ["E1", "E2"]  # E3 (force patterns) has no glove

    def subjects(self) -> list[str]:
        return [f"s{i:02d}" for i in range(1, 41)]

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        i = int(subject[1:])
        members = tuple(f"S{i}_{e}_A1.mat" for e in self.EXERCISES)
        return [RemoteFile(url=f"{BASE_URL}/DB2_Preproc/DB2_s{i}.zip", name=f"DB2_s{i}.zip", unzip=members)]

    def sessions(self, subject: str) -> list[str]:
        return self.EXERCISES

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        path = raw_dir / f"S{int(subject[1:])}_{session}_A1.mat"
        info = {"condition": "intact", "label_source": "measured", "kin_units": "rad_uncalibrated"}
        return read_ninapro_session(path, self.native_fs, CYBERGLOVE_22, subject, session, info)


class NinaProDB8(Dataset):
    name = "ninapro_db8"
    citation = "Krasoulis et al. 2019, Frontiers in Neuroscience 13:891"
    license = "see https://ninapro.hevs.ch (cite the paper)"
    url = "https://ninapro.hevs.ch/instructions/DB8.html"
    device = "16 Delsys Trigno IM in two rows of 8 around the right forearm at the radio-humeral joint"
    electrode_type = "bipolar"
    native_fs = 2000.0
    effective_fs = 1111.0  # "sampled at a rate of 1111 Hz ... upsampled to 2 kHz"
    mains_hz = 50.0
    montages = {"ring8": list(range(8))}  # assumed: first row = channels 1-8 (not documented)
    verified = {
        "mat_variables": "S1_E1_A3 opened: emg (T,16) float32 V, glove (T,18) no NaN, restimulus 0-9",
        "glove_order": "correlation structure matches CYBERGLOVE_18",
        "ring8": "row membership of channels not documented",
    }
    AMPUTEES = ["s11", "s12"]

    def subjects(self) -> list[str]:
        return [f"s{i:02d}" for i in range(1, 13)]

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        files = []
        for session in self.sessions(subject):
            name = f"S{int(subject[1:])}_E1_{session}.mat"
            files.append(RemoteFile(url=f"{BASE_URL}/DB8/{name}", name=name))
        return files

    def sessions(self, subject: str) -> list[str]:
        return ["A1", "A2", "A3"]

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        path = raw_dir / f"S{int(subject[1:])}_E1_{session}.mat"
        condition = "amputee_transradial" if subject in self.AMPUTEES else "intact"
        # calibrated glove on the left hand, mirroring the movement (the phantom hand for amputees)
        info = {"condition": condition, "label_source": "mirrored", "kin_units": "rad"}
        return read_ninapro_session(path, self.native_fs, CYBERGLOVE_18, subject, session, info)
