import argparse
import json
from pathlib import Path

import numpy as np
from compare_synthetic import CONFIG, SyntheticCheck
from myogen import JointAdapter
from myogen.tools.viewer import replay


def main():
    parser = argparse.ArgumentParser(
        description="A stretch of a real session replayed through myogen (MuJoCo) at the dataset's electrodes, in the "
        "web viewer: the hand, the muscles (coloured by drive), the electrodes and the simulated EMG."
    )
    parser.add_argument("session", help="interim session key, e.g. ninapro_db2/s01/E1")
    parser.add_argument("--interim", default="data/interim")
    parser.add_argument("--config", default=str(CONFIG), help="the synthetic check settings (subject, params)")
    parser.add_argument("--first", type=int, default=0, help="first motion (cue repetition or chunk) shown")
    parser.add_argument("--motions", type=int, default=4, help="motions shown, with the rest between them")
    parser.add_argument("--save", help="also write the recording here (.h5): myogen replay <file> opens it again")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    check = SyntheticCheck.load(Path(args.config))
    folder = Path(args.interim) / args.session
    meta = json.loads((folder / "meta.json").read_text())
    kin, stimulus = np.load(folder / "kin.npy"), np.load(folder / "stimulus.npy")
    windows = sorted(check.windows(stimulus, meta["label_fs"], meta))[args.first : args.first + args.motions]
    if not windows:
        raise SystemExit(f"{args.session}: no motion {args.first}")
    start, end = windows[0][0], windows[-1][1]
    angles = check.model_angles(kin, meta, stimulus)
    simulator, _, _ = check.simulator(meta)
    motion = JointAdapter()({j: q[start:end] for j, q in angles.items()}, meta["label_fs"])
    print(f"{args.session}: {(end - start) / meta['label_fs']:.1f} s from {start / meta['label_fs']:.1f} s")
    recording = simulator.run(motion, progress=True)
    if args.save:
        recording.save(args.save)
    replay(recording, port=args.port)


if __name__ == "__main__":
    main()
