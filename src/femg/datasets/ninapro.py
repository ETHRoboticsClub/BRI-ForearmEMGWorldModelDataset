from pathlib import Path

import numpy as np

from femg.datasets import ninapro_db9
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
    flipped: joints whose sensor runs against the canonical sign. scale: file units -> rad (degrees by default)."""
    kin = np.full((len(glove), N_JOINTS), np.nan, np.float32)
    for column, joint in enumerate(sensors.values()):
        sign = -1.0 if joint in flipped else 1.0
        kin[:, JOINT_INDEX[joint]] = sign * scale * glove[:, column]
    return kin


def read_ninapro_session(
    path: Path,
    fs: float,
    sensors: dict[str, str],
    subject: str,
    session: str,
    info: dict,
    flipped: tuple[str, ...] = (),
    glove: bool = True,
) -> Session:
    """sensors: glove column name -> canonical joint. glove=False: cues only, a glove in the file is not read."""
    data = load_mat(path, ["emg", "restimulus", "glove"] if glove else ["emg", "restimulus"])
    n = min(len(v) for v in data.values())  # restimulus is sometimes one sample shorter

    emg = np.ascontiguousarray(data["emg"][:n], np.float32)  # volts (DB2 S1: std ~4e-5)
    kin = None
    if "glove" in data:
        kin = glove_to_canonical(data["glove"][:n].astype(np.float32), sensors, flipped)
    stimulus = None
    if "restimulus" in data:
        stimulus = data["restimulus"][:n, 0].astype(np.int16)

    return Session(subject, session, emg=emg, fs=fs, kin=kin, kin_fs=fs, stimulus=stimulus, info=info)


class NinaProDB2(Dataset):
    name = "ninapro_db2"

    EXERCISES = ["E1", "E2", "E3"]
    GLOVE_EXERCISES = ["E1", "E2"]  # E3 (force patterns, exercise D) has no glove: cues only

    def subjects(self) -> list[str]:
        return [f"s{i:02d}" for i in range(1, 41)]

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        i = int(subject[1:])
        members = tuple(f"S{i}_{e}_A1.mat" for e in self.EXERCISES)
        files = [RemoteFile(url=f"{BASE_URL}/DB2_Preproc/DB2_s{i}.zip", name=f"DB2_s{i}.zip", unzip=members)]
        if self.config.kin_source == "db9":
            files.append(ninapro_db9.remote_file(subject))
        return files

    def sessions(self, subject: str) -> list[str]:
        return self.EXERCISES

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        path = raw_dir / f"S{int(subject[1:])}_{session}_A1.mat"
        if self.config.kin_source == "db9" and session in self.GLOVE_EXERCISES:
            return self.load_with_db9(subject, session, path, raw_dir)
        info = {"condition": "intact", "label_source": "measured", "kin_units": "rad_uncalibrated"}
        if session not in self.GLOVE_EXERCISES:
            info = {"condition": "intact", "label_source": "cue", "kin_units": None}
        return read_ninapro_session(
            path, self.config.native_fs, CYBERGLOVE_22, subject, session, info, flipped=NINAPRO_FLIPPED
        )

    def load_with_db9(self, subject: str, session: str, path: Path, raw_dir: Path) -> Session:
        """DB2's EMG with DB9's calibrated angles (degrees, canonical signs) as labels, and DB9's restimulus. Refuses a
        DB9 file that does not line up with the DB2 one (scripts/tests/check_db9.py)."""
        data = load_mat(path, ["emg", "restimulus"])
        emg = np.ascontiguousarray(data["emg"], np.float32)
        stimulus = data["restimulus"][: len(emg), 0]
        db9_path, alignment, db9 = ninapro_db9.pick(stimulus, ninapro_db9.mat_files(subject, raw_dir), session)
        if alignment["offset"] != 0 or alignment["match"] < 0.999 or alignment["overlap"] < 0.999:
            raise ValueError(f"{db9_path.name} does not line up with {path.name}: {alignment}")

        ratio = alignment["ratio"]
        kin = ninapro_db9.to_canonical(db9["angles"], db9["names"])
        n = min(len(kin), len(emg) // ratio)
        info = {"condition": "intact", "label_source": "measured", "kin_units": "rad", "kin_file": db9_path.name}
        fs = self.config.native_fs
        return Session(
            subject,
            session,
            emg=np.ascontiguousarray(emg[: n * ratio]),
            fs=fs,
            kin=kin[:n],
            kin_fs=fs / ratio,
            stimulus=db9["restimulus"][:n],
            info=info,
        )


# NinaPro DB4 and DB5 (exercises A, B, C) number each exercise's cues from 1. Added to them, the 52 movements are
# numbered across exercises (A 1-12, B 13-29, C 30-52: config/cues/ninapro_abc.yaml)
ABC_CUE_OFFSETS = {"E1": 0, "E2": 12, "E3": 29}


def number_across_exercises(session: Session) -> Session:
    moving = session.stimulus > 0
    session.stimulus[moving] += ABC_CUE_OFFSETS[session.name]
    return session


class NinaProDB3(Dataset):
    """11 transradial amputees, DB2's protocol (exercises B, C, D) and electrodes; cues only."""

    name = "ninapro_db3"

    EXERCISES = ["E1", "E2", "E3"]
    # s07 (0 % of the forearm left): EMG at the noise floor, no channel stronger during the cues than at rest (config
    # notes s07); its two forearm spots are flat
    EXCLUDED = ["s07"]
    LEFT = ["s02", "s05", "s06"]  # the left hand amputated (the DB3 page); the others the right

    def subjects(self) -> list[str]:
        return [f"s{i:02d}" for i in range(1, 12) if f"s{i:02d}" not in self.EXCLUDED]

    def condition(self, subject: str) -> str:
        return "amputee_transradial"

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        i = int(subject[1:])
        members = tuple(f"S{i}_{e}_A1.mat" for e in self.EXERCISES)
        return [RemoteFile(url=f"{BASE_URL}/db3_Preproc/s{i}_0.zip", name=f"DB3_s{i}.zip", unzip=members)]

    def sessions(self, subject: str) -> list[str]:
        return self.EXERCISES

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        """The glove is not read: it cannot sit on the amputated hand (config notes glove)."""
        path = raw_dir / f"S{int(subject[1:])}_{session}_A1.mat"
        info = {
            "condition": self.condition(subject),
            "label_source": "cue",
            "kin_units": None,
            "hand": "left" if subject in self.LEFT else "right",
            "task": "movements imagined with the amputated hand",
        }
        out = read_ninapro_session(path, self.config.native_fs, CYBERGLOVE_22, subject, session, info, glove=False)
        dead = np.flatnonzero(out.emg.std(axis=0) < 1e-9)
        if len(dead):  # the virtual array would take a flat channel for a silent electrode
            raise ValueError(f"{path.name}: channels {dead.tolist()} are flat (exclude the subject, see s07)")
        return out


class NinaProDB4(Dataset):
    """10 intact subjects, 12 Cometa electrodes placed as DB2's, exercises A, B, C; cues only (no glove in files)."""

    name = "ninapro_db4"

    EXERCISES = ["E1", "E2", "E3"]

    def subjects(self) -> list[str]:
        return [f"s{i:02d}" for i in range(1, 11)]

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        i = int(subject[1:])
        members = tuple(f"S{i}_{e}_A1.mat" for e in self.EXERCISES)
        return [RemoteFile(url=f"{BASE_URL}/DB4_Preproc/s{i}.zip", name=f"DB4_s{i}.zip", unzip=members)]

    def sessions(self, subject: str) -> list[str]:
        return self.EXERCISES

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        path = raw_dir / f"S{int(subject[1:])}_{session}_A1.mat"
        info = {"condition": "intact", "label_source": "cue", "kin_units": None}
        out = read_ninapro_session(path, self.config.native_fs, CYBERGLOVE_22, subject, session, info, glove=False)
        out.emg = out.emg * np.float32(self.config.emg_unit_scale)
        return number_across_exercises(out)


class NinaProDB5(Dataset):
    """Two Myo armbands (16 channels, 200 Hz) and the raw CyberGlove II, exercises A, B and C."""

    name = "ninapro_db5"

    EXERCISES = ["E1", "E2", "E3"]
    NOISY_SENSORS = ("index_middle_abd",)  # sensor 11: noise (config notes glove_sensor_11), left NaN

    def subjects(self) -> list[str]:
        return [f"s{i:02d}" for i in range(1, 11)]

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        i = int(subject[1:])
        members = tuple(f"S{i}_{e}_A1.mat" for e in self.EXERCISES)
        return [RemoteFile(url=f"{BASE_URL}/DB5_Preproc/s{i}.zip", name=f"DB5_s{i}.zip", unzip=members)]

    def sessions(self, subject: str) -> list[str]:
        return self.EXERCISES

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        path = raw_dir / f"S{int(subject[1:])}_{session}_A1.mat"
        info = {"condition": "intact", "label_source": "measured", "kin_units": "rad_uncalibrated"}
        out = read_ninapro_session(
            path, self.config.native_fs, CYBERGLOVE_22, subject, session, info, flipped=NINAPRO_FLIPPED
        )
        out.emg = out.emg * np.float32(self.config.emg_unit_scale)
        for joint in self.NOISY_SENSORS:
            out.kin[:, JOINT_INDEX[joint]] = np.nan
        return number_across_exercises(out)


class NinaProDB8(Dataset):
    name = "ninapro_db8"

    AMPUTEES = ["s11", "s12"]

    def condition(self, subject: str) -> str:
        return "amputee_transradial" if subject in self.AMPUTEES else "intact"

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
        # calibrated glove on the left hand, mirroring the movement (the phantom hand for amputees)
        info = {"condition": self.condition(subject), "label_source": "mirrored", "kin_units": "rad"}
        return read_ninapro_session(path, self.config.native_fs, CYBERGLOVE_18, subject, session, info)


class NinaProDB7(Dataset):
    name = "ninapro_db7"

    AMPUTEES = ["s21", "s22"]  # right hand amputated; s21 lacks the last two movements of E2

    def condition(self, subject: str) -> str:
        return "amputee_transradial" if subject in self.AMPUTEES else "intact"

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
            "condition": self.condition(subject),
            "label_source": "mirrored" if amputee else "measured",
            "kin_units": "rad_uncalibrated",
        }
        return read_ninapro_session(
            path, self.config.native_fs, CYBERGLOVE_18, subject, session, info, flipped=NINAPRO_FLIPPED
        )
