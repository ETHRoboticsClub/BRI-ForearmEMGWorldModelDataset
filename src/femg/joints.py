"""The common label space: every dataset maps its hand kinematics onto these 22 joints.

Units: radians. Flexion > 0, 0 = fully extended (anatomical neutral). Abductions are spreads between neighbouring
fingers (> 0 = spread). Joints a dataset does not measure are NaN and get masked out of the loss.
"""

CANONICAL_JOINTS = [
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
]
N_JOINTS = len(CANONICAL_JOINTS)
JOINT_INDEX = {name: i for i, name in enumerate(CANONICAL_JOINTS)}  # "index_mcp_flex" -> 4

# Allowed values of session.info["label_source"]: how the kinematic targets were obtained.
#   "measured": sensor on the same hand as the EMG
#   "mirrored": sensor on the other hand doing mirrored movements (works for amputees)
#   "cue":      only the instructed movement (stimulus), no kinematics
LABEL_SOURCES = ["measured", "mirrored", "cue"]

# Allowed values of session.info["kin_units"]:
#   "rad":              calibrated joint angles
#   "rad_uncalibrated": proportional to the angle, gain/offset differ per session (raw CyberGlove)
KIN_UNITS = ["rad", "rad_uncalibrated"]
