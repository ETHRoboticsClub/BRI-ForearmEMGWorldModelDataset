# Right hand, radians. 0 = extended; flexion and finger spread > 0; wrist_dev > 0 = radial. NaN = not measured.
CANONICAL_JOINTS = [
    "thumb_cmc_flex",  # thumb rotation across the palm
    "thumb_cmc_abd",
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
    "index_middle_abd",
    "middle_ring_abd",
    "ring_pinky_abd",
    "palm_arch",
    "wrist_flex",
    "wrist_dev",
]
N_JOINTS = len(CANONICAL_JOINTS)
JOINT_INDEX = {name: i for i, name in enumerate(CANONICAL_JOINTS)}

# measured: sensor on the EMG hand; mirrored: sensor on the other hand doing the same movement; cue: stimulus only
LABEL_SOURCES = ["measured", "mirrored", "cue"]

# rad_uncalibrated: proportional to the angle, with a per-session gain and offset (raw CyberGlove)
KIN_UNITS = ["rad", "rad_uncalibrated"]
