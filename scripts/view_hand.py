import argparse
import json
from pathlib import Path

import numpy as np
import viser
from mjviser import Viewer

from femg.hand import HandMap, available_hands, label_text, set_pose
from femg.normalize import as_array, normalize_kin


def main():
    parser = argparse.ArgumentParser(description="Replay a processed session on a hand at http://localhost:PORT.")
    parser.add_argument("session", type=Path, help="e.g. data/processed/ninapro_db8/s01/A1")
    parser.add_argument("--hand", default="shadow_hand", choices=list(available_hands()))
    parser.add_argument("--start", type=float, default=0.0, help="start time (s)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rad", action="store_true", help="drive the hand with the angles, not the 0-1 joints")
    args = parser.parse_args()

    meta = json.loads((args.session / "meta.json").read_text())
    stats = meta["kin_stats"]
    kin = np.load(args.session / "kin.npy")
    unit = not args.rad
    centers = (as_array(stats["mean"]) - as_array(stats["low"])) / (as_array(stats["high"]) - as_array(stats["low"]))
    if unit:
        kin = normalize_kin(kin, stats)
    stimulus = np.load(args.session / "stimulus.npy")
    label_fs = meta["label_fs"]
    hand = HandMap(args.hand)
    model, data, ids = hand.load()

    def frame() -> int:
        return int(data.time * label_fs) % len(kin)

    def step(model, data):
        data.time += model.opt.timestep
        set_pose(model, data, hand, ids, kin[frame()], unit, centers)

    def reset(model, data):
        data.time = args.start
        set_pose(model, data, hand, ids, kin[frame()], unit, centers)

    server = viser.ViserServer(port=args.port)
    info = server.gui.add_markdown("")
    rules = [r for r in hand.rules if r.label]
    labels = {r.joint: server.scene.add_label(f"/labels/{r.joint}", "", font_screen_scale=0.8) for r in rules}

    def render(scene):
        scene.update_from_mjdata(data)
        k = frame()
        how = hand.angles(kin[k], unit, centers)
        for r in rules:
            labels[r.joint].position = data.xanchor[ids[r.joint]] + scene._scene_offset
            labels[r.joint].text = label_text(r, kin[k], unit, how[r.joint][1])
        info.content = (
            f"**{meta['key']}** on {hand.title} ({meta['label_source']}, {meta['kin_units']})  \n"
            f"t = {k / label_fs:.2f} s / {len(kin) / label_fs:.0f} s, stimulus = {stimulus[k]}"
        )

    reset(model, data)
    Viewer(model, data, step_fn=step, render_fn=render, reset_fn=reset, server=server).run()


if __name__ == "__main__":
    main()
