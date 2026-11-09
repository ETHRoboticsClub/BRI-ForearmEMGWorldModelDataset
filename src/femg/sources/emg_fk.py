from pathlib import Path

import numpy as np

from femg.schema import JOINT_INDEX, N_JOINTS, Recording, RemoteFile, SessionRef
from femg.sources.base import Source

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

# "Angle 1..15": MediaPipe relative angles between consecutive landmark triplets, in degrees, 180 = straight.
# The order is not documented; it is inferred from the correlation structure of Subject_01 (1-3 are independent
# of the rest = thumb; 4-15 form four correlated triplets = index..pinky, MCP/PIP/DIP following MediaPipe's
# landmark order). Verify against figures/figure_angles.png of the authors' repo before trusting per-joint metrics.
ANGLES = (
    "thumb_cmc_flex",
    "thumb_mcp_flex",
    "thumb_ip_flex",
    "index_mcp_flex",
    "index_pip_flex",
    "index_dip_flex",
    "middle_mcp_flex",
    "middle_pip_flex",
    "middle_dip_flex",
    "ring_mcp_flex",
    "ring_pip_flex",
    "ring_dip_flex",
    "pinky_mcp_flex",
    "pinky_pip_flex",
    "pinky_dip_flex",
)


class EMGFK(Source):
    name = "emg_fk"
    citation = "Colot et al. 2026, arXiv:2604.22499, doi:10.5281/zenodo.19453843"
    license = "CC-BY-4.0"
    url = "https://zenodo.org/records/19453843"
    device = "MindRove 8-channel dry-electrode armband, 20 cm from the wrist"
    electrode_type = "monopolar"
    native_fs = 500.0
    effective_fs = 500.0
    mains_hz = 50.0  # recorded in Belgium
    emg_unit_scale = 1e-6  # files claim volts but values are ~60 (std): microvolts, consistent with NinaPro levels
    montages = {"ring8": list(range(8))}
    # Zenodo: "a constant shift of approximately 300 ms can be observed between the two signal types".
    # Direction not documented: estimate it (docs/data-spec.md, roadmap) before setting it here.
    kin_lag_s = 0.0
    verified = {
        "angle_order": "inferred from correlations, not confirmed",
        "emg_unit": "inferred from amplitude",
    }

    def subjects(self) -> list[str]:
        return [f"s{i:02d}" for i in range(1, 21)]

    def sessions(self, subject: str) -> list[SessionRef]:
        i = int(subject[1:])
        name = f"Subject_{i:02d}.fif"
        f = RemoteFile(f"{ZENODO}/{name}", name, md5=MD5[i])
        info = {"condition": "intact", "label_source": "measured", "kin_units": "rad", "task": "free finger motion"}
        return [SessionRef(self.name, subject, "free", [f], info)]

    def read(self, ref: SessionRef, raw_dir: Path) -> Recording:
        import mne

        raw = mne.io.read_raw_fif(raw_dir / ref.files[0].name, preload=False, verbose="ERROR")
        fs = raw.info["sfreq"]
        emg = raw.get_data(picks=[f"EMG {i}" for i in range(1, 9)]).T.astype(np.float32) * self.emg_unit_scale
        angles = raw.get_data(picks=[f"Angle {i}" for i in range(1, 16)]).T.astype(np.float32)
        kin = np.full((len(angles), N_JOINTS), np.nan, np.float32)
        for j, name in enumerate(ANGLES):
            kin[:, JOINT_INDEX[name]] = np.deg2rad(180.0 - angles[:, j])
        return Recording(emg=np.ascontiguousarray(emg), fs=fs, kin=kin, kin_fs=fs)
