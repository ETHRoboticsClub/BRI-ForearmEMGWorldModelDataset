"""The data contract every built session follows. See docs/data-spec.md for the rationale."""

from dataclasses import dataclass, field

import numpy as np

# Canonical right-hand joint set. Units: radians. Flexion > 0, 0 = fully extended (anatomical neutral).
# Abductions are spreads between neighbouring fingers (> 0 = spread). Every dataset maps what it has onto
# this list; joints it does not measure are NaN and masked out of the loss.
CANONICAL_JOINTS = (
    "thumb_cmc_flex",  # 0  thumb rotation/opposition across the palm
    "thumb_cmc_abd",  # 1  thumb-index spread
    "thumb_mcp_flex",  # 2
    "thumb_ip_flex",  # 3
    "index_mcp_flex",  # 4
    "index_pip_flex",  # 5
    "index_dip_flex",  # 6
    "middle_mcp_flex",  # 7
    "middle_pip_flex",  # 8
    "middle_dip_flex",  # 9
    "ring_mcp_flex",  # 10
    "ring_pip_flex",  # 11
    "ring_dip_flex",  # 12
    "pinky_mcp_flex",  # 13
    "pinky_pip_flex",  # 14
    "pinky_dip_flex",  # 15
    "index_middle_abd",  # 16
    "middle_ring_abd",  # 17
    "ring_pinky_abd",  # 18
    "palm_arch",  # 19
    "wrist_flex",  # 20
    "wrist_dev",  # 21 radial (+) / ulnar (-) deviation
)
JOINT_INDEX = {name: i for i, name in enumerate(CANONICAL_JOINTS)}
N_JOINTS = len(CANONICAL_JOINTS)

# How the kinematic targets of a session were obtained.
LABEL_SOURCES = (
    "measured",  # sensor on the same hand as the EMG
    "mirrored",  # sensor on the contralateral hand doing mirrored movements (works for amputees)
    "cue",  # only the instructed movement (stimulus), no kinematics
)
# Units of the canonical targets before any normalization.
KIN_UNITS = (
    "rad",  # calibrated joint angles
    "rad_uncalibrated",  # proportional to the joint angle, gain/offset differ per session (raw CyberGlove)
)


@dataclass
class RemoteFile:
    """A file to download. `extract` lists archive members to keep (by basename) when url is a zip."""

    url: str
    name: str
    size: int | None = None
    md5: str | None = None
    extract: tuple[str, ...] = ()


@dataclass
class SessionRef:
    """One continuous recording. `files` are downloaded into the source's raw dir before `Source.read`."""

    dataset: str
    subject: str
    session: str
    files: list[RemoteFile]
    info: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.subject}/{self.session}"


@dataclass
class Recording:
    """A session as read from disk, in the source's native rates. Arrays are time-major."""

    emg: np.ndarray  # (T, C) float32, volts
    fs: float
    kin: np.ndarray | None  # (Tk, N_JOINTS) float32 canonical joints, NaN where not measured
    kin_fs: float
    stimulus: np.ndarray | None = None  # (Tk,) int, instructed movement id (0 = rest), at kin_fs
    meta: dict = field(default_factory=dict)  # per-session extras merged into meta.json
