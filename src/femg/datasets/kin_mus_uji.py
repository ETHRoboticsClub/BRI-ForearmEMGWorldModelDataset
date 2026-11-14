from functools import cache
from pathlib import Path

import numpy as np

from femg.datasets.base import Dataset, Session, concat_trials
from femg.download import RemoteFile
from femg.joints import JOINT_INDEX, N_JOINTS

ZENODO = "https://zenodo.org/records/3469380/files"
MD5 = {"KIN_MUS_UJI.mat": "4fc49241d6a887f6f1e9dbbcfc6aef12", "RAW_EMG.mat": "c0045aac1480480c265501a7c6f3192c"}
EMG_FS = 1000.0
KIN_FS = 100.0

# Kinematic_data columns (paper Table 2), degrees; Table 3: flexion, fingers separated, radial deviation and palmar
# arch flexion > 0, as the canonical joints. The thumb base is anatomical, not the CyberGlove pair the canonical
# thumb labels follow (src/femg/hands/README.md): grasping (phase 2) vs the open hand at the start of reaching raises
# CMC1_A in 22/22 subjects (+5.4 deg), as the glove's thumb roll (thumb_cmc_flex) rises in fists, and raises CMC1_F,
# the thumb going across the palm, which lowers the glove's thumb abduction (thumb_cmc_abd). Hence the swap and sign.
ANGLES = [
    "thumb_cmc_flex",  # CMC1_A: abduction, the thumb lifting in front of the palm
    "thumb_cmc_abd",  # CMC1_F, negated (FLIPPED): flexion, the thumb going across the palm
    "thumb_mcp_flex",  # MCP1_F
    "thumb_ip_flex",  # IP1_F
    "index_middle_abd",  # MCP2-3_A
    "index_mcp_flex",  # MCP2_F
    "index_pip_flex",  # PIP2_F
    "middle_mcp_flex",  # MCP3_F
    "middle_pip_flex",  # PIP3_F
    "middle_ring_abd",  # MCP3-4_A
    "ring_mcp_flex",  # MCP4_F
    "ring_pip_flex",  # PIP4_F
    "palm_arch",  # CMC5_F
    "ring_pinky_abd",  # MCP4-5_A
    "pinky_mcp_flex",  # MCP5_F
    "pinky_pip_flex",  # PIP5_F
    "wrist_flex",  # WRIST_F
    "wrist_dev",  # WRIST_A
]


FLIPPED = ("thumb_cmc_abd",)


@cache
def load_records(raw_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """(kinematics: one record per subject x ADL x phase, raw EMG: one record per subject x ADL). ~0.5 GB."""
    import scipy.io

    kin = scipy.io.loadmat(Path(raw_dir) / "KIN_MUS_UJI.mat")["EMG_KIN_v4"].ravel()
    raw = scipy.io.loadmat(Path(raw_dir) / "RAW_EMG.mat")["RAW_EMG"].ravel()
    return kin, raw


class KinMusUJI(Dataset):
    """26 activities of daily living with real objects: the most natural movements of the pool."""

    name = "kin_mus_uji"
    citation = "Jarque-Bou et al. 2019, Scientific Data 6:270, doi:10.5281/zenodo.3469380"
    license = "CC-BY-4.0"
    url = "https://zenodo.org/records/3469380"
    device = "Biometrics DataLOG, 7 SX230 dry bipolar electrodes on 7 forearm spots (gain 1000, 20-460 Hz)"
    electrode_type = "bipolar"
    native_fs = EMG_FS
    effective_fs = EMG_FS
    mains_hz = 50.0  # recorded in Spain
    emg_unit_scale = 1e-3  # millivolts: std 0.04-0.09, quantization step 0.0007
    montages = {"spots7": list(range(7))}  # 7 muscle spots along the forearm, no ring
    verified = {
        "md5": "both files downloaded, md5 matches Zenodo",
        "structure": "EMG_KIN_v4 (1716 = 22 subjects x 26 ADLs x 3 phases): Subject, ADL, Phase, time, "
        "Kinematic_data (T,18) deg at 100 Hz, EMG_data (envelope, not used); RAW_EMG (572): Subject, ADL, time "
        "(from 0.001 s at 1 kHz), Raw_EMG (T,7)",
        "angle_order": "paper Table 2; S1: MCP2/MCP3 flexion correlate 0.92, p2-p98 ranges anatomical",
        "angle_signs": "Table 3 (fingers separated+, radial+, flexion+); thumb CMC pair mapped from the grasp test "
        "(see ANGLES)",
        "time_axis": "kinematic 'time' is on the raw EMG clock (S1 ADL1: kin 2.6-5.94 s, raw 0.001-5.94 s)",
        "emg_unit": "inferred from amplitude",
        "gaps": "s21 ADL25 phase 1 and s22 ADL25 phases 2-3 are empty (1x1) records: NaN kinematics there",
    }

    def subjects(self) -> list[str]:
        return [f"s{i:02d}" for i in range(1, 23)]

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        return [RemoteFile(url=f"{ZENODO}/{name}", name=name, md5=md5) for name, md5 in MD5.items()]

    def sessions(self, subject: str) -> list[str]:
        return ["adl"]

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        """The 26 ADLs of a subject joined. Kinematics cover reach -> manipulate -> release. Before the first
        and after the last kinematic sample of an ADL (the still periods the authors trimmed) the hand holds that
        pose and the stimulus is 0. Left as NaN, the processor would interpolate across the join between ADLs."""
        kin_records, raw_records = load_records(str(raw_dir))
        number = int(subject[1:])
        ratio = round(EMG_FS / KIN_FS)

        trials = []
        for raw in sorted((r for r in raw_records if r["Subject"][0, 0] == number), key=lambda r: r["ADL"][0, 0]):
            adl = int(raw["ADL"][0, 0])
            emg = raw["Raw_EMG"].astype(np.float32) * np.float32(self.emg_unit_scale)
            n = len(emg) // ratio
            kin = np.full((n, N_JOINTS), np.nan, np.float32)
            stimulus = np.zeros(n, np.int16)
            for phase in (r for r in kin_records if r["Subject"][0, 0] == number and r["ADL"][0, 0] == adl):
                if phase["Kinematic_data"].shape[1:] != (len(ANGLES),):
                    continue  # 3 empty placeholders (1, 1): s21 ADL25 phase 1, s22 ADL25 phases 2-3
                # EMG sample i is at (i + 1) ms, so label k (EMG sample 10 k) is at 10 k + 1 ms
                k = np.round((phase["time"][:, 0] - 1 / EMG_FS) * KIN_FS).astype(int)
                keep = (k >= 0) & (k < n)
                angles = np.deg2rad(phase["Kinematic_data"][keep].astype(np.float32))
                for column, joint in enumerate(ANGLES):
                    sign = -1.0 if joint in FLIPPED else 1.0
                    kin[k[keep], JOINT_INDEX[joint]] = sign * angles[:, column]
                stimulus[k[keep]] = adl
            recorded = np.flatnonzero(np.isfinite(kin).any(axis=1))
            if len(recorded):
                kin[: recorded[0]] = kin[recorded[0]]
                kin[recorded[-1] + 1 :] = kin[recorded[-1]]
            trials.append((emg, kin, stimulus))

        emg, kin, stimulus, starts = concat_trials(trials, ratio)
        info = {
            "condition": "intact",
            "label_source": "measured",
            "kin_units": "rad",
            "task": "26 activities of daily living (stimulus = ADL id while the hand is recorded)",
            "trial_starts_s": [s / KIN_FS for s in starts],
        }
        return Session(
            subject,
            session,
            emg=np.ascontiguousarray(emg),
            fs=EMG_FS,
            kin=kin,
            kin_fs=KIN_FS,
            stimulus=stimulus,
            info=info,
        )
