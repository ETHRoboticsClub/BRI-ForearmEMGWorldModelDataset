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

# The joints a dataset can measure, by device. A dict maps the device's column names, in file order, to the joints.

# CyberGlove II/III, 22 sensors (inner = MCP, middle = PIP, outer = DIP; abductions between neighbouring fingers)
CYBERGLOVE_22 = {
    "Thumb_Roll": "thumb_cmc_flex",  # thumb rotation across the palm
    "Thumb_Inner": "thumb_mcp_flex",
    "Thumb_Outer": "thumb_ip_flex",
    "TI_Abduction": "thumb_cmc_abd",
    "Index_Inner": "index_mcp_flex",
    "Index_Middle": "index_pip_flex",
    "Index_Outer": "index_dip_flex",
    "Middle_Inner": "middle_mcp_flex",
    "Middle_Middle": "middle_pip_flex",
    "Middle_Outer": "middle_dip_flex",
    "MI_Abduction": "index_middle_abd",
    "Ring_Inner": "ring_mcp_flex",
    "Ring_Middle": "ring_pip_flex",
    "Ring_Outer": "ring_dip_flex",
    "RM_Abduction": "middle_ring_abd",
    "Pinky_Inner": "pinky_mcp_flex",
    "Pinky_Middle": "pinky_pip_flex",
    "Pinky_Outer": "pinky_dip_flex",
    "PR_Abduction": "ring_pinky_abd",
    "Palm_Arch": "palm_arch",
    "Wrist_Flex": "wrist_flex",
    "Wrist_Abduction": "wrist_dev",
}
# The 18-sensor glove: the same without the DIP sensors
CYBERGLOVE_18 = {sensor: joint for sensor, joint in CYBERGLOVE_22.items() if not joint.endswith("_dip_flex")}

# The 15 finger flexions, thumb to pinky, base to tip (camera-based datasets)
FINGER_FLEXIONS = [j for j in CANONICAL_JOINTS if j.endswith("_flex") and not j.startswith("wrist")]
