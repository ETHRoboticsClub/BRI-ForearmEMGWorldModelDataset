from pathlib import Path

import numpy as np

from femg.datasets.base import Dataset, Session
from femg.download import RemoteFile
from femg.hand.joints import CYBERGLOVE_18, CYBERGLOVE_22, JOINT_INDEX, N_JOINTS

BASE_URL = "https://ninapro.hevs.ch/files"

# Raw NinaPro CyberGlove sensors that run against the canonical sign (spread > 0, radial deviation > 0). Checked on the
# instructed E1 movements, per repetition: abduction of all fingers (5) reads lower than adduction (8) on all three
# spread sensors (6/6 reps in DB2 S1, S2 and DB7 S1), radial deviation (15) lower than ulnar (16) (5/6 reps).
NINAPRO_FLIPPED = ("index_middle_abd", "middle_ring_abd", "ring_pinky_abd", "wrist_dev")


def load_mat(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    import scipy.io

    try:
        data = scipy.io.loadmat(path, variable_names=names)
        return {k: data[k] for k in names if k in data}
    except NotImplementedError:  # v7.3 files are HDF5
        import h5py

        with h5py.File(path, "r") as f:
            return {k: np.asarray(f[k]).T for k in names if k in f}


def glove_to_canonical(
    glove: np.ndarray, sensors: dict[str, str], flipped: tuple[str, ...] = (), scale: float = np.pi / 180
) -> np.ndarray:
    """Glove columns -> canonical joints. sensors: column name -> joint, in column order (CYBERGLOVE_22, ...).
    flipped: joints whose sensor runs against the canonical sign. scale: file units -> rad (degrees by default; 1 keeps
    unitless values such as SEEDS' 0-1)."""
    kin = np.full((len(glove), N_JOINTS), np.nan, np.float32)
    for column, joint in enumerate(sensors.values()):
        sign = -1.0 if joint in flipped else 1.0
        kin[:, JOINT_INDEX[joint]] = sign * scale * glove[:, column]
    return kin


def read_ninapro_session(
    path: Path,
    fs: float,
    sensors: dict[str, str] | None,
    subject: str,
    session: str,
    info: dict,
    emg_scale: float = 1.0,
    flipped: tuple[str, ...] = (),
) -> Session:
    """sensors: glove column name -> canonical joint, or None to ignore the glove (cue-only datasets)."""
    data = load_mat(path, ["emg", "restimulus"] + (["glove"] if sensors else []))
    n = min(len(v) for v in data.values())  # restimulus is sometimes one sample shorter

    emg = np.ascontiguousarray(data["emg"][:n] * np.float32(emg_scale), np.float32)  # volts (DB2 S1: std ~4e-5)
    kin = None
    if sensors and "glove" in data:
        kin = glove_to_canonical(data["glove"][:n].astype(np.float32), sensors, flipped)
    stimulus = None
    if "restimulus" in data:
        stimulus = data["restimulus"][:n, 0].astype(np.int16)

    return Session(subject, session, emg=emg, fs=fs, kin=kin, kin_fs=fs, stimulus=stimulus, info=info)


class NinaProDB2(Dataset):
    name = "ninapro_db2"

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
        return read_ninapro_session(
            path, self.config.native_fs, CYBERGLOVE_22, subject, session, info, flipped=NINAPRO_FLIPPED
        )


class NinaProDB8(Dataset):
    name = "ninapro_db8"

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
        return read_ninapro_session(path, self.config.native_fs, CYBERGLOVE_18, subject, session, info)


class NinaProDB7(Dataset):
    name = "ninapro_db7"

    AMPUTEES = ["s21", "s22"]  # right hand amputated; s21 lacks the last two movements of E2

    def subjects(self) -> list[str]:
        return [f"s{i:02d}" for i in range(1, 23)]

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        i = int(subject[1:])
        members = tuple(f"S{i}_{e}_A1.mat" for e in self.sessions(subject))
        return [RemoteFile(url=f"{BASE_URL}/DB7_Preproc/Subject_{i}.zip", name=f"DB7_s{i}.zip", unzip=members)]

    def sessions(self, subject: str) -> list[str]:
        return ["E1", "E2"]

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        path = raw_dir / f"S{int(subject[1:])}_{session}_A1.mat"
        amputee = subject in self.AMPUTEES
        info = {
            "condition": "amputee_transradial" if amputee else "intact",
            "label_source": "mirrored" if amputee else "measured",
            "kin_units": "rad_uncalibrated",
        }
        return read_ninapro_session(
            path, self.config.native_fs, CYBERGLOVE_18, subject, session, info, flipped=NINAPRO_FLIPPED
        )


class NinaProDB3(Dataset):
    name = "ninapro_db3"

    LEFT_AMPUTATED = ["s05", "s06"]  # the others are right-hand amputees

    def subjects(self) -> list[str]:
        return [f"s{i:02d}" for i in range(1, 12)]

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        i = int(subject[1:])
        members = tuple(f"S{i}_{e}_A1.mat" for e in self.sessions(subject))
        return [RemoteFile(url=f"{BASE_URL}/db3_Preproc/s{i}_0.zip", name=f"DB3_s{i}.zip", unzip=members)]

    def sessions(self, subject: str) -> list[str]:
        return ["E1", "E2", "E3"]  # E3: finger force patterns

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        path = raw_dir / f"S{int(subject[1:])}_{session}_A1.mat"
        info = {
            "condition": "amputee_transradial",
            "amputated_side": "left" if subject in self.LEFT_AMPUTATED else "right",
            "label_source": "cue",
            "kin_units": None,
        }
        return read_ninapro_session(path, self.config.native_fs, None, subject, session, info)


class NinaProDB4(Dataset):
    name = "ninapro_db4"

    def subjects(self) -> list[str]:
        return [f"s{i:02d}" for i in range(1, 11)]

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        i = int(subject[1:])
        members = tuple(f"S{i}_{e}_A1.mat" for e in self.sessions(subject))
        return [RemoteFile(url=f"{BASE_URL}/DB4_Preproc/s{i}.zip", name=f"DB4_s{i}.zip", unzip=members)]

    def sessions(self, subject: str) -> list[str]:
        return ["E1", "E2", "E3"]

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        path = raw_dir / f"S{int(subject[1:])}_{session}_A1.mat"
        info = {"condition": "intact", "label_source": "cue", "kin_units": None}
        return read_ninapro_session(
            path, self.config.native_fs, None, subject, session, info, self.config.emg_unit_scale
        )
