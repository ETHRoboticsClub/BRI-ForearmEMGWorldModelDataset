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
    glove: np.ndarray, sensors: list[str], flipped: tuple[str, ...] = (), scale: float = np.pi / 180
) -> np.ndarray:
    """Glove columns -> canonical joints. flipped: joints whose sensor runs against the canonical sign.
    scale: file units -> rad (degrees by default; 1 keeps unitless values such as SEEDS' 0-1)."""
    kin = np.full((len(glove), N_JOINTS), np.nan, np.float32)
    for column, joint in enumerate(sensors):
        sign = -1.0 if joint in flipped else 1.0
        kin[:, JOINT_INDEX[joint]] = sign * scale * glove[:, column]
    return kin


def read_ninapro_session(
    path: Path,
    fs: float,
    sensors: list[str] | None,
    subject: str,
    session: str,
    info: dict,
    emg_scale: float = 1.0,
    flipped: tuple[str, ...] = (),
) -> Session:
    """sensors: glove column -> canonical joint, or None to ignore the glove (cue-only datasets)."""
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
        "glove_signs": "spreads and wrist_dev flipped (NINAPRO_FLIPPED), from the instructed E1 movements of S1, S2",
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
        return read_ninapro_session(
            path, self.native_fs, CYBERGLOVE_22, subject, session, info, flipped=NINAPRO_FLIPPED
        )


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
        "glove_signs": "NOT flipped: calibrated glove and no spread/deviation movement to check against. DB2/DB7 "
        "(raw) needed NINAPRO_FLIPPED; check DB8 visually in the explorer",
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


class NinaProDB7(Dataset):
    name = "ninapro_db7"
    citation = "Krasoulis et al. 2017, Journal of NeuroEngineering and Rehabilitation 14:71"
    license = "see https://ninapro.hevs.ch (cite the paper)"
    url = "https://ninapro.hevs.ch/instructions/DB7.html"
    device = (
        "12 Delsys Trigno IM: 8 evenly spaced around the forearm at the radio-humeral joint + FDS, EDC, biceps, triceps"
    )
    electrode_type = "bipolar"
    native_fs = 2000.0
    effective_fs = 1111.0  # same Trigno IM as DB8; S1_E1: 0.8 % of the EMG power above 555 Hz
    mains_hz = 50.0
    montages = {"ring8": list(range(8))}  # assumed as in DB2/DB3/DB4 (the DB7 page gives no column order)
    verified = {
        "zip_members": "Subject_1.zip, Subject_21.zip listed: S{i}_E1_A1.mat, S{i}_E2_A1.mat (~0.6-0.9 GB each)",
        "mat_variables": "S1_E1_A1 opened: emg (T,12) float32 V (rest RMS 5-30 uV), glove (T,18) no NaN, restimulus",
        "glove": "uncalibrated per the page (abduction sensors sit near -120); MCP/PIP pairs correlate 0.76-0.87",
        "glove_signs": "spreads and wrist_dev flipped as in DB2 (instructed E1 movements of S1); thumb as DB2",
        "effective_fs": "inferred from the spectrum, not documented on the DB7 page",
        "glove_hand": "not documented; assumed the EMG hand for intact subjects, the other hand for amputees",
        "ring8": "column order not documented; assumed channels 1-8 as in the other NinaPro databases",
    }
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
            path, self.native_fs, CYBERGLOVE_18, subject, session, info, flipped=NINAPRO_FLIPPED
        )


class NinaProDB3(Dataset):
    """Transradial amputees, cue labels only: the glove columns exist but barely move (see `verified`)."""

    name = "ninapro_db3"
    citation = "Atzori et al. 2014, Scientific Data 1:140053"
    license = "see https://ninapro.hevs.ch (cite the paper)"
    url = "https://ninapro.hevs.ch/instructions/DB3.html"
    device = (
        "12 Delsys Trigno: 8 evenly spaced around the forearm at the radio-humeral joint + FDS, EDC, biceps, triceps"
    )
    electrode_type = "bipolar"
    native_fs = 2000.0
    effective_fs = 2000.0
    mains_hz = 50.0
    montages = {"ring8": list(range(8))}  # page: columns 1-8 are the ring
    verified = {
        "zip_members": "s1_0.zip, s11_0.zip listed: DB3_s{i}/S{i}_E{1,2,3}_A1.mat (+ __MACOSX entries, ignored)",
        "mat_variables": "S1_E1/E3 opened: emg (T,12) float32 V (rest RMS 2-5 uV), restimulus; glove (T,22) in E1/E2",
        "glove": "S1: E2 glove std 0.05 raw units and identical across movements, E1 barely moves -> not used",
        "channels": "page says 'up to 12' electrodes: some subjects may have flat channels (QC)",
    }
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
        return read_ninapro_session(path, self.native_fs, None, subject, session, info)


class NinaProDB4(Dataset):
    """Intact subjects, cue labels only: the DB4 page lists a glove, but the files have none."""

    name = "ninapro_db4"
    citation = "Pizzolato et al. 2017, PLoS ONE 12(10):e0186132"
    license = "see https://ninapro.hevs.ch (cite the paper)"
    url = "https://ninapro.hevs.ch/instructions/DB4.html"
    device = "12 Cometa: 8 evenly spaced around the forearm at the radio-humeral joint + FDS, EDC, biceps, triceps"
    electrode_type = "bipolar"
    native_fs = 2000.0
    effective_fs = 2000.0
    mains_hz = 50.0
    emg_unit_scale = 1e-6  # assumed microvolts: S1 rest RMS 60-250, movement 200-1700 (not documented)
    montages = {"ring8": list(range(8))}  # page: columns 1-8 are the ring
    verified = {
        "zip_members": "s1.zip, s10.zip listed: s{i}/S{i}_E{1,2,3}_A1.mat",
        "mat_variables": "S1 E1-E3 opened: emg (T,12) float32, restimulus, frequency=2000, sensor='Cometa'; NO glove",
        "emg_unit": "assumed uV from amplitude; rest RMS 60-250 uV is high, to confirm",
        "exercise_variable": "the 'exercise' variable of E1/E2 files is swapped (2/1); the file names match the "
        "movement counts of the page (12/17/23), so the file names are used",
    }

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
        return read_ninapro_session(path, self.native_fs, None, subject, session, info, self.emg_unit_scale)
