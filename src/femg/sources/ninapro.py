"""NinaPro databases (https://ninapro.hevs.ch). Downloads are public, no login."""

from pathlib import Path

import numpy as np

from femg.schema import JOINT_INDEX, N_JOINTS, Recording, RemoteFile, SessionRef
from femg.sources.base import Source

BASE = "https://ninapro.hevs.ch/files"

# CyberGlove II sensor order (manufacturer numbering). The 18-sensor model drops the DIP sensors.
# Checked on DB8 S1_E1_A3: MCP/PIP pairs correlate at 0.9 and the three abduction sensors anti-correlate with flexion.
CYBERGLOVE_22 = (
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
)
CYBERGLOVE_18 = tuple(n for n in CYBERGLOVE_22 if not n.endswith("_dip_flex"))


def glove_to_canonical(glove: np.ndarray, sensors: tuple[str, ...]) -> np.ndarray:
    """CyberGlove readings (degree-like) -> canonical joints in radians, NaN for joints the glove lacks."""
    kin = np.full((len(glove), N_JOINTS), np.nan, np.float32)
    for j, name in enumerate(sensors):
        kin[:, JOINT_INDEX[name]] = np.deg2rad(glove[:, j])
    return kin


def load_mat(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    """Read selected variables from a MATLAB file (v5 via scipy, v7.3/HDF5 via h5py)."""
    import scipy.io as sio

    try:
        m = sio.loadmat(path, variable_names=names)
        return {k: m[k] for k in names if k in m}
    except NotImplementedError:  # v7.3
        import h5py

        with h5py.File(path, "r") as f:
            return {k: np.asarray(f[k]).T for k in names if k in f}


class NinaPro(Source):
    mains_hz = 50.0
    emg_unit_scale = 1.0  # volts (DB2 S1: std ~4e-5)
    sensors: tuple[str, ...]
    kin_units: str
    label_source: str

    def _read(self, path: Path, ref: SessionRef) -> Recording:
        m = load_mat(path, ["emg", "glove", "restimulus"])
        n = min(len(v) for v in m.values())  # restimulus is sometimes one sample shorter
        emg = np.ascontiguousarray(m["emg"][:n], np.float32) * self.emg_unit_scale
        kin = glove_to_canonical(m["glove"][:n].astype(np.float32), self.sensors) if "glove" in m else None
        stim = m["restimulus"][:n, 0].astype(np.int16) if "restimulus" in m else None
        return Recording(emg=emg, fs=self.native_fs, kin=kin, kin_fs=self.native_fs, stimulus=stim)


class NinaProDB2(NinaPro):
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
    montages = {"ring8": list(range(8))}
    sensors = CYBERGLOVE_22
    kin_units = "rad_uncalibrated"  # "uncalibrated signal from the 22 sensors of the cyberglove"
    label_source = "measured"
    verified = {
        "mat_variables": "DB2_s1 E1/E3 opened: emg (T,12) float32 V, glove (T,22), restimulus; E3 has no glove",
        "glove_order": "manufacturer order; not checked on DB2 data",
    }
    EXERCISES = ("E1", "E2")  # E3 (force patterns) has no glove

    def subjects(self) -> list[str]:
        return [f"s{i:02d}" for i in range(1, 41)]

    def sessions(self, subject: str) -> list[SessionRef]:
        i = int(subject[1:])
        members = tuple(f"S{i}_{e}_A1.mat" for e in self.EXERCISES)
        zipf = RemoteFile(f"{BASE}/DB2_Preproc/DB2_s{i}.zip", f"DB2_s{i}.zip", extract=members)
        info = {"condition": "intact", "label_source": self.label_source, "kin_units": self.kin_units}
        # Both sessions share one zip; the downloader skips it once its members exist.
        return [
            SessionRef(self.name, subject, e, [zipf], {**info, "file": m})
            for e, m in zip(self.EXERCISES, members, strict=True)
        ]

    def read(self, ref: SessionRef, raw_dir: Path) -> Recording:
        return self._read(raw_dir / ref.info["file"], ref)


class NinaProDB8(NinaPro):
    name = "ninapro_db8"
    citation = "Krasoulis et al. 2019, Frontiers in Neuroscience 13:891"
    license = "see https://ninapro.hevs.ch (cite the paper)"
    url = "https://ninapro.hevs.ch/instructions/DB8.html"
    device = "16 Delsys Trigno IM in two rows of 8 around the right forearm at the radio-humeral joint"
    electrode_type = "bipolar"
    native_fs = 2000.0
    effective_fs = 1111.0  # "sampled at a rate of 1111 Hz ... upsampled to 2 kHz"
    montages = {"ring8": list(range(8))}  # assumed: first row = channels 1-8 (not documented)
    sensors = CYBERGLOVE_18
    kin_units = "rad"  # "calibrated signal from the sensors of the cyberglove"
    label_source = "mirrored"  # glove on the left hand, bilateral mirrored movements (phantom hand for amputees)
    AMPUTEES = {"s11", "s12"}
    verified = {
        "mat_variables": "S1_E1_A3 opened: emg (T,16) float32 V, glove (T,18) no NaN, restimulus 0-9",
        "glove_order": "correlation structure matches CYBERGLOVE_18",
        "ring8": "row membership of channels not documented",
    }

    def subjects(self) -> list[str]:
        return [f"s{i:02d}" for i in range(1, 13)]

    def sessions(self, subject: str) -> list[SessionRef]:
        i = int(subject[1:])
        condition = "amputee_transradial" if subject in self.AMPUTEES else "intact"
        info = {"condition": condition, "label_source": self.label_source, "kin_units": self.kin_units}
        refs = []
        for a in (1, 2, 3):
            name = f"S{i}_E1_A{a}.mat"
            refs.append(SessionRef(self.name, subject, f"A{a}", [RemoteFile(f"{BASE}/DB8/{name}", name)], info))
        return refs

    def read(self, ref: SessionRef, raw_dir: Path) -> Recording:
        return self._read(raw_dir / ref.files[0].name, ref)
