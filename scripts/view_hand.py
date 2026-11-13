import argparse
import json
from pathlib import Path

import mujoco
import mujoco_menagerie
import numpy as np
import viser
from mjviser import Viewer

from femg.joints import JOINT_INDEX

# canonical joint -> Shadow Hand joint. Shadow J1/J2/J3 = DIP/PIP/MCP flexion, J4 = spread, LFJ5 = palm arch.
TO_SHADOW = {
    "wrist_dev": "WRJ2",
    "wrist_flex": "WRJ1",
    "index_middle_abd": "FFJ4",
    "index_mcp_flex": "FFJ3",
    "index_pip_flex": "FFJ2",
    "index_dip_flex": "FFJ1",
    "middle_mcp_flex": "MFJ3",
    "middle_pip_flex": "MFJ2",
    "middle_dip_flex": "MFJ1",
    "middle_ring_abd": "RFJ4",
    "ring_mcp_flex": "RFJ3",
    "ring_pip_flex": "RFJ2",
    "ring_dip_flex": "RFJ1",
    "palm_arch": "LFJ5",
    "ring_pinky_abd": "LFJ4",
    "pinky_mcp_flex": "LFJ3",
    "pinky_pip_flex": "LFJ2",
    "pinky_dip_flex": "LFJ1",
    "thumb_cmc_flex": "THJ5",  # rotation across the palm
    "thumb_cmc_abd": "THJ4",
    "thumb_mcp_flex": "THJ2",
    "thumb_ip_flex": "THJ1",
}
SPREADS = {"index_middle_abd", "middle_ring_abd", "ring_pinky_abd"}  # Shadow J4 > 0 closes the finger: flip sign
FINGERS = {"thumb": "T", "index": "I", "middle": "M", "ring": "R", "pinky": "P"}


def short(name: str) -> str:
    """index_pip_flex -> "I pip", index_middle_abd -> "IM abd", wrist_flex -> "wrist flex"."""
    parts = name.split("_")
    if parts[0] not in FINGERS:
        return " ".join(parts)
    parts = name.removesuffix("_flex").split("_")
    fingers = "".join(FINGERS[p] for p in parts if p in FINGERS)
    return " ".join([fingers] + [p for p in parts if p not in FINGERS])


def load_hand():
    model = mujoco_menagerie.load("shadow_hand", "right_hand")
    model.stat.center[:], model.stat.extent = (0.33, 0.0, 0.0), 0.1  # mjviser frames the camera on the fingers
    joints = {name: model.joint(f"rh_{shadow}").id for name, shadow in TO_SHADOW.items()}
    return model, mujoco.MjData(model), joints


def set_pose(model, data, joints, kin_row):
    """Write one frame of canonical joints into the hand (kinematics only, no physics)."""
    for name, j in joints.items():
        value = np.nan_to_num(kin_row[JOINT_INDEX[name]]) * (-1 if name in SPREADS else 1)
        data.qpos[model.jnt_qposadr[j]] = np.clip(value, *model.jnt_range[j])
    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path, help="a processed session folder (with kin.npy and meta.json)")
    parser.add_argument("--start", type=float, default=0.0, help="start time (s)")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    meta = json.loads((args.session / "meta.json").read_text())
    kin = np.load(args.session / "kin.npy")
    stimulus = np.load(args.session / "stimulus.npy")
    label_fs = meta["label_fs"]
    model, data, joints = load_hand()

    def frame() -> int:  # label frame shown at the current session time, looping at the end
        return int(data.time * label_fs) % len(kin)

    def step(model, data):  # the "simulation" is the recording: advance its clock and pose the hand
        data.time += model.opt.timestep
        set_pose(model, data, joints, kin[frame()])

    def reset(model, data):
        data.time = args.start
        set_pose(model, data, joints, kin[frame()])

    server = viser.ViserServer(port=args.port)
    info = server.gui.add_markdown("")
    labels = {name: server.scene.add_label(f"/labels/{name}", "", font_screen_scale=0.8) for name in joints}

    def render(scene):
        scene.update_from_mjdata(data)
        k = frame()
        for name, j in joints.items():
            value = kin[k, JOINT_INDEX[name]]
            labels[name].position = data.xanchor[j] + scene._scene_offset  # follows "track body" if enabled
            labels[name].text = f"{short(name)} {np.rad2deg(value):.0f}" if np.isfinite(value) else f"{short(name)} -"
        info.content = (
            f"**{meta['key']}** ({meta['label_source']}, {meta['kin_units']})  \n"
            f"t = {k / label_fs:.2f} s / {len(kin) / label_fs:.0f} s, stimulus = {stimulus[k]}"
        )

    reset(model, data)
    Viewer(model, data, step_fn=step, render_fn=render, reset_fn=reset, server=server).run()


if __name__ == "__main__":
    main()
