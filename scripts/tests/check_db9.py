"""Does NinaPro DB9 line up with DB2, and what would using its calibrated angles cost?

For every DB2 subject whose files are in --raw (and DB9 zip, --download fetches it):
- alignment: the DB9 file picked for each exercise, its rate against DB2's (ratio), its offset, how many restimulus
  labels agree, and whether DB9's raw glove is DB2's glove sample for sample (then the two files are one recording);
- signs: DB9's angles against DB2's raw glove as the repo reads it (with NINAPRO_FLIPPED; r < 0 = a sign disagrees),
  and against the instructed movements of E1 (abduction 5 > adduction 8 on the spreads, flexion 13 > extension 14,
  radial 15 > ulnar 16), which the paper's sign convention predicts;
- loss: subjects or exercises without a DB9 file, joints DB9 does not give (sensor 11), labelled frames lost.

    python scripts/tests/check_db9.py --download              # every DB2 subject in data/raw/ninapro_db2
    python scripts/tests/check_db9.py --subjects s01 s02
Exit code 1 if a session does not line up.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from femg.datasets import DATASETS
from femg.datasets import ninapro_db9 as db9
from femg.datasets.ninapro import NINAPRO_FLIPPED, glove_to_canonical, load_mat
from femg.download import download_file, is_downloaded
from femg.hand.joints import CANONICAL_JOINTS, CYBERGLOVE_22, JOINT_INDEX

SIGN_CUES = {  # E1 movement pairs: (joint, cue where it is larger, cue where it is smaller)
    "index_middle_abd": (5, 8),
    "middle_ring_abd": (5, 8),
    "ring_pinky_abd": (5, 8),
    "wrist_flex": (13, 14),
    "wrist_dev": (15, 16),
}


def check_session(subject: str, exercise: str, raw: Path) -> dict:
    path = raw / f"S{int(subject[1:])}_{exercise}_A1.mat"
    db2 = load_mat(path, ["glove", "restimulus"])
    stimulus = db2["restimulus"].reshape(-1).astype(np.int16)
    row = {"key": f"{subject}/{exercise}"}
    try:
        candidates = db9.mat_files(subject, raw)
    except FileNotFoundError as error:
        return row | {"error": f"no DB9: {error}"}
    if not candidates:
        return row | {"error": "DB9 zip holds no .mat"}
    db9_path, alignment, data = db9.pick(stimulus, candidates, exercise)
    row |= {"file": db9_path.name, **alignment}
    ratio, offset = alignment["ratio"], alignment["offset"]

    # the overlap, in DB9 samples
    start2, start9 = max(offset, 0), max(-offset, 0)
    n = min(len(stimulus[::ratio]) - start2, len(data["angles"]) - start9)
    glove2 = db2["glove"][::ratio][start2 : start2 + n].astype(np.float64)
    angles = data["angles"][start9 : start9 + n].astype(np.float64)
    cue = stimulus[::ratio][start2 : start2 + n]

    if data["glove"] is not None:
        glove9 = data["glove"][start9 : start9 + n].astype(np.float64)
        row["glove_identical"] = float(np.isclose(glove2, glove9, atol=1e-6).all(axis=1).mean())

    row["names"] = "from file" if data["names"] else "assumed glove order"
    kin9 = db9.to_canonical(angles, data["names"])
    kin2 = glove_to_canonical(glove2, CYBERGLOVE_22, NINAPRO_FLIPPED)
    row["joints_db2"] = {CANONICAL_JOINTS[j] for j in range(len(CANONICAL_JOINTS)) if np.isfinite(kin2[:, j]).any()}
    row["joints_db9"] = {CANONICAL_JOINTS[j] for j in range(len(CANONICAL_JOINTS)) if np.isfinite(kin9[:, j]).any()}
    labelled2 = np.isfinite(kin2).any(axis=1)
    labelled9 = np.isfinite(kin9[:, sorted(JOINT_INDEX[j] for j in row["joints_db9"])]).all(axis=1)
    row["frames_lost"] = float((labelled2 & ~labelled9).sum() / max(labelled2.sum(), 1)) + (1 - alignment["overlap"])

    row["r_glove"] = {}
    for joint in sorted(row["joints_db9"] & row["joints_db2"]):
        a, b = kin9[:, JOINT_INDEX[joint]], kin2[:, JOINT_INDEX[joint]]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 100 and a[ok].std() > 0 and b[ok].std() > 0:
            row["r_glove"][joint] = float(np.corrcoef(a[ok], b[ok])[0, 1])

    row["cue_signs"] = {}
    if exercise == "E1":
        for joint, (high, low) in SIGN_CUES.items():
            if joint not in row["joints_db9"]:
                continue
            values = kin9[:, JOINT_INDEX[joint]]
            a, b = np.nanmedian(values[cue == high]), np.nanmedian(values[cue == low])
            if np.isfinite(a) and np.isfinite(b):
                row["cue_signs"][joint] = float(np.rad2deg(a - b))
    return row


def print_rows(rows: list[dict]):
    print(f"{'session':12}{'DB9 file':24}{'ratio':>6}{'offset':>8}{'labels':>8}{'overlap':>9}{'glove =':>9}"
          f"{'frames lost':>13}  angle names")  # fmt: skip
    for r in rows:
        if "error" in r:
            print(f"{r['key']:12}{r['error']}")
            continue
        glove = f"{r['glove_identical']:.1%}" if "glove_identical" in r else "-"
        print(
            f"{r['key']:12}{r['file'][:23]:24}{r['ratio']:>6}{r['offset']:>8}{r['match']:>8.1%}{r['overlap']:>9.1%}"
            f"{glove:>9}{r['frames_lost']:>13.1%}  {r['names']}"
        )


def print_signs(rows: list[dict]):
    ok = [r for r in rows if "error" not in r]
    joints = sorted({j for r in ok for j in r["r_glove"]}, key=CANONICAL_JOINTS.index)
    print("\nDB9 angle vs DB2 raw glove as the repo reads it (median r over sessions; < 0: the signs disagree)")
    for joint in joints:
        values = [r["r_glove"][joint] for r in ok if joint in r["r_glove"]]
        flag = "  <- sign" if np.median(values) < 0 else "  (weak)" if np.median(values) < 0.5 else ""
        flipped = " [flipped in DB2]" if joint in NINAPRO_FLIPPED else ""
        print(f"  {joint:18} r {np.median(values):+.2f} (min {min(values):+.2f}){flipped}{flag}")
    print("\nDB9 on the E1 cues (median degrees, first cue minus second; > 0 = canonical sign)")
    for joint, (high, low) in SIGN_CUES.items():
        values = [r["cue_signs"][joint] for r in ok if joint in r["cue_signs"]]
        if values:
            agree = np.mean(np.array(values) > 0)
            print(f"  {joint:18} cue {high} - cue {low}: {np.median(values):+6.1f} deg, > 0 in {agree:.0%} of sessions")
        else:
            print(f"  {joint:18} not in DB9")


def print_loss(rows: list[dict], subjects: list[str]):
    ok = [r for r in rows if "error" not in r]
    aligned = [r for r in ok if r["offset"] == 0 and r["match"] >= 0.999 and r["overlap"] >= 0.999]
    print("\nWHAT USING DB9 COSTS")
    print(f"  sessions lined up: {len(aligned)} / {len(rows)} ({len(subjects)} subjects, E1 and E2)")
    for r in rows:
        if "error" in r:
            print(f"  {r['key']}: {r['error']}")
        elif r not in aligned:
            print(f"  {r['key']}: offset {r['offset']}, labels {r['match']:.1%}, overlap {r['overlap']:.1%}")
    if ok:
        lost = sorted(set.union(*[r["joints_db2"] - r["joints_db9"] for r in ok]), key=CANONICAL_JOINTS.index)
        print(f"  joints lost against DB2's glove: {', '.join(lost) or 'none'}")
        print(f"  labelled frames lost: {np.mean([r['frames_lost'] for r in ok]):.2%} on average")
    print("  E3 (force patterns): no glove in DB2 either, nothing lost")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw/ninapro_db2"))
    parser.add_argument("--subjects", nargs="+", help="default: every DB2 subject with its files in --raw")
    parser.add_argument("--download", action="store_true", help="download the DB9 zips that are missing")
    args = parser.parse_args()

    dataset = DATASETS["ninapro_db2"]()
    subjects = args.subjects or [
        s
        for s in dataset.subjects()
        if all((args.raw / f"S{int(s[1:])}_{e}_A1.mat").exists() for e in dataset.GLOVE_EXERCISES)
    ]
    if not subjects:
        sys.exit(f"no DB2 subject in {args.raw}: python scripts/pull_data.py --dataset ninapro_db2")

    rows = []
    for subject in subjects:
        remote = db9.remote_file(subject)
        if args.download and not is_downloaded(remote, args.raw):
            print(f"downloading {remote.url}", flush=True)
            try:
                download_file(remote, args.raw)
            except Exception as error:
                print(f"  FAILED: {error!r}")
        for exercise in dataset.GLOVE_EXERCISES:
            rows.append(check_session(subject, exercise, args.raw))
    print_rows(rows)
    print_signs(rows)
    print_loss(rows, subjects)
    failed = [r for r in rows if "error" in r or r["offset"] != 0 or r["match"] < 0.999 or r["overlap"] < 0.999]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
