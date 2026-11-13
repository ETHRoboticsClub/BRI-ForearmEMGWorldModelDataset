import argparse
import sys

import mujoco
import numpy as np

from femg.hand import (
    HANDS,
    HandMap,
    available_hands,
    dump_mapping,
    hand_frame,
    load_model,
    neutral_pose,
    tips_center,
)
from femg.joints import CANONICAL_JOINTS

# Where the fingertip must go when a label closes (moves from 0 to 1).
# Thumb, from the instructed NinaPro movements: the glove's "thumb rotation" (thumb_cmc_flex) rises in fists, grips
# and opposition (the thumb comes in front of the palm), and "thumb abduction" (thumb_cmc_abd) is highest at rest and
# drops to ~0 in all of them (the thumb crosses the palm): closing it moves the thumb away from the little finger.
EXPECTED = {label: {"palmar"} for label in CANONICAL_JOINTS if label.endswith("_flex")} | {
    "thumb_mcp_flex": {"to pinky"},
    "thumb_ip_flex": {"to pinky"},
    "thumb_cmc_flex": {"palmar"},
    "thumb_cmc_abd": {"from pinky"},
    "index_middle_abd": {"radial"},
    "middle_ring_abd": {"ulnar"},
    "ring_pinky_abd": {"ulnar"},
    "palm_arch": {"palmar", "radial"},
    "wrist_dev": {"radial"},
}


def tip_direction(model, data, joint: int, step: float, frame, ulnar_body: int | None = None, base=None) -> str:
    """Where the end of the joint's chain moves for a small rotation `step` from the neutral pose.

    With ulnar_body: "to pinky" / "from pinky", whether the tip gets closer to that body. base: {joint id: angle}
    set before moving: thumb labels are tested with the rest of the thumb closed (it opposes once abducted)."""
    tip = model.jnt_bodyid[joint]
    neutral_pose(model, data)
    for j, angle in (base or {}).items():
        data.qpos[model.jnt_qposadr[j]] = angle
    mujoco.mj_forward(model, data)
    before = tips_center(model, data, tip).copy()
    adr = model.jnt_qposadr[joint]
    data.qpos[adr] = np.clip(data.qpos[adr] + step, *model.jnt_range[joint])
    mujoco.mj_forward(model, data)
    move = tips_center(model, data, tip) - before
    if ulnar_body is not None:
        target = data.xpos[ulnar_body]
        closer = np.linalg.norm(tips_center(model, data, tip) - target) < np.linalg.norm(before - target)
        return "to pinky" if closer else "from pinky"
    names = [("palmar", "dorsal"), ("radial", "ulnar"), ("distal", "proximal")]
    scores = [move.dot(axis) for axis in frame]
    i = int(np.argmax(np.abs(scores)))
    return names[i][0] if scores[i] > 0 else names[i][1]


def template(args):
    """Print a mapping file with every joint of the model, its limits and where it moves, labels empty."""
    source = {"builder": args.model} if ":" in args.model else {"model": args.model, "entry": args.entry}
    model = load_model(source)
    data = mujoco.MjData(model)
    frame = hand_frame(model, data, args.palmar_joint, args.radial_body, args.ulnar_body)
    landmarks = {"palmar_joint": args.palmar_joint, "radial_body": args.radial_body, "ulnar_body": args.ulnar_body}
    settings = {"title": args.title or args.model, **source, **landmarks}
    joints = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        lo, hi = model.jnt_range[j]
        joints[model.joint(j).name] = {
            "label": None,
            "at_0": round(float(np.clip(0, lo, hi)), 3),
            "at_1": round(float(hi), 3),
            "moves": tip_direction(model, data, j, 0.3, frame),
            "range": [round(float(lo), 2), round(float(hi), 2)],
        }
    sys.stdout.write(dump_mapping(settings, joints))


def check(args):
    names = args.names or list(available_hands())
    failed = False
    for name in names:
        hand = HandMap(name)
        model, data, ids = hand.load()
        s = hand.settings
        frame = hand_frame(model, data, s["palmar_joint"], s["radial_body"], s["ulnar_body"])
        print(f"\n{hand.title} ({name})")
        for r in hand.rules:
            if not r.label:
                note = f"copies {r.ratio:g} x {r.source}" if r.source else f"fixed at {r.fallback:g}"
                print(f"  {r.joint:24s} {'-':18s} {note}")
                continue
            step = 0.3 if r.at_1 >= r.at_0 else -0.3
            by_distance = EXPECTED.get(r.label) in ({"to pinky"}, {"from pinky"})
            ulnar = model.body(s["ulnar_body"]).id if by_distance else None
            thumb = {ids[t.joint]: t.at_1 for t in hand.rules if t.label and t.label.startswith("thumb") and t is not r}
            opposition = r.label == "thumb_cmc_abd"  # crossing the palm only shows once the thumb is lifted
            moves = tip_direction(model, data, ids[r.joint], step, frame, ulnar, thumb if opposition else None)
            ok = moves in EXPECTED.get(r.label, {moves})
            failed |= not ok
            extra = f", else {r.ratio:g} x {r.source}" if r.source else ""
            verdict = "ok" if ok else "WRONG, expected " + "/".join(EXPECTED[r.label])
            print(f"  {r.joint:24s} {r.label:18s} closes {moves:10s} {verdict}{extra}")
        unused = [label for label in CANONICAL_JOINTS if label not in {r.label for r in hand.rules}]
        print(f"  labels with no joint: {', '.join(unused) or 'none'}")
    sys.exit(1 if failed else 0)


def main():
    parser = argparse.ArgumentParser(prog="python -m tools.hands", description="Write and check hand mapping files.")
    sub = parser.add_subparsers(required=True)
    t = sub.add_parser("template", help=f"print a mapping file to fill, e.g. > {HANDS}/<name>.json")
    t.add_argument("model", help="mujoco-menagerie name (leap_hand), or a builder module:function returning an MjSpec")
    t.add_argument("--entry", default="right_hand")
    t.add_argument("--title")
    t.add_argument("--palmar-joint", required=True, help="an index MCP flexion joint: its + direction is 'palmar'")
    t.add_argument("--radial-body", required=True, help="a body on the thumb side of the palm (index knuckle)")
    t.add_argument("--ulnar-body", required=True, help="a body on the other side (pinky or ring knuckle)")
    t.set_defaults(run=template)
    c = sub.add_parser("check", help="validate mapping files and test every label's direction")
    c.add_argument("names", nargs="*")
    c.set_defaults(run=check)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
