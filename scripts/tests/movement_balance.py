"""How balanced the movements of every dataset are: between its instructed movements (time and repetitions per cue)
and between the fingers (how much each finger moves, how often, and how often alone).

All on the training targets: the joints in 0 (open) - 1 (closed) of every session (Normalizer.kin), so a finger's
motion is counted in the units the loss sees. Speeds are computed inside segments only.

    python scripts/tests/movement_balance.py                       # every dataset in data/interim
    python scripts/tests/movement_balance.py --datasets great --detail
-> docs/figures/movement_balance_fingers.png, movement_balance_cues.png and tables on stdout, the last one the shared
gestures x datasets (data/processed/<dataset>/gestures.parquet, cue-only sessions included)
"""

import argparse
import json
import warnings
from collections import defaultdict
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from femg.datasets import DATASETS
from femg.hand.joints import JOINT_INDEX
from femg.normalize import Normalizer, has_kinematics

FINGERS = {
    "thumb": ["thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp_flex", "thumb_ip_flex"],
    **{f: [f"{f}_mcp_flex", f"{f}_pip_flex", f"{f}_dip_flex"] for f in ["index", "middle", "ring", "pinky"]},
}
OTHERS = {  # measured by the gloves only; shown next to the fingers, not in their shares
    "spread": ["index_middle_abd", "middle_ring_abd", "ring_pinky_abd"],
    "wrist": ["wrist_flex", "wrist_dev"],
}
GROUPS = {**FINGERS, **OTHERS}
ACTIVE = 0.3  # a group moves when its mean joint speed is above this (range / s: the whole range in ~3 s)
MIN_REP_S = 0.5  # a repetition: a run of one cue at least this long


@cache
def dataset_cues(dataset: str) -> dict:
    return DATASETS[dataset]().config.cues


def cue_name(dataset: str, label: int) -> str:
    """The cue's shared gesture (short, comparable across datasets), else its name in the dataset (config cues)."""
    if label == 0:
        return "rest"
    native, gesture = dataset_cues(dataset).get(label, (str(label), None))
    return gesture or native


def print_gestures(processed: Path, names: list[str]):
    """Shared gestures x datasets: minutes of each (from data/processed/<dataset>/gestures.parquet), cue-only sessions
    included."""
    tables = [pd.read_parquet(f) for name in names if (f := processed / name / "gestures.parquet").exists()]
    if not tables:
        print(f"\nSHARED GESTURES: no gestures.parquet in {processed} (build the processed stage)")
        return
    table = pd.concat(tables)
    seconds = table.pivot_table(index=["gesture_id", "gesture"], columns="dataset", values="seconds", aggfunc="sum")
    minutes = seconds / 60
    print("\nSHARED GESTURES (minutes; gesture.npy ids)")
    print(minutes.round(1).fillna("").to_string())
    shared = (minutes.drop(index=0, level="gesture_id").notna().sum(axis=1) > 1).sum()
    print(f"{shared} of {len(minutes) - 1} gestures (rest apart) appear in more than one dataset")


def group_speeds(kin: np.ndarray, segments: list, label_fs: int) -> np.ndarray:
    """(T, groups): mean |speed| of each group's measured joints, 0-1 range / s; NaN where nothing is labelled."""
    speed = np.full_like(kin, np.nan)
    for start, end in segments:
        if end - start > 2:
            speed[start:end] = np.abs(np.gradient(kin[start:end], axis=0)) * label_fs
    out = np.full((len(kin), len(GROUPS)), np.nan)
    for g, joints in enumerate(GROUPS.values()):
        columns = speed[:, [JOINT_INDEX[j] for j in joints]]
        measured = np.isfinite(columns).any(axis=0)
        if measured.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # frames without a label
                out[:, g] = (
                    np.nanmean(columns[:, measured], axis=1) if measured.sum() > 1 else columns[:, measured][:, 0]
                )
    return out


def load_session(folder: Path) -> dict | None:
    meta = json.loads((folder / "meta.json").read_text())
    if not has_kinematics(meta):
        return None
    kin = Normalizer.kin(np.load(folder / "kin.npy"), meta["kin_stats"])
    speeds = group_speeds(kin, meta["segments"], meta["label_fs"])
    stimulus = np.load(folder / "stimulus.npy")[: len(kin)].astype(int)
    return {"meta": meta, "speeds": speeds, "stimulus": stimulus}


def finger_stats(speeds: np.ndarray) -> dict[str, np.ndarray]:
    """Per group: mean speed, fraction of labelled frames moving, fraction of a finger's moving frames with no other
    finger moving. Frames count when every finger is labelled."""
    n_fingers = len(FINGERS)
    labelled = np.isfinite(speeds[:, :n_fingers]).all(axis=1)
    s = speeds[labelled]
    with np.errstate(invalid="ignore"):
        moving = s > ACTIVE
        mean = np.nanmean(s, axis=0) if len(s) else np.full(len(GROUPS), np.nan)
        active = np.where(np.isfinite(s).any(axis=0), moving.mean(axis=0), np.nan) if len(s) else mean
        fingers_moving = moving[:, :n_fingers].sum(axis=1)
        alone = np.full(len(GROUPS), np.nan)
        for g in range(n_fingers):
            if moving[:, g].any():
                alone[g] = (moving[:, g] & (fingers_moving == 1)).sum() / moving[:, g].sum()
    return {"mean": mean, "active": active, "alone": alone, "frames": labelled.sum()}


def shares(mean: np.ndarray) -> np.ndarray:
    """Each finger's part of the finger motion; spread and wrist relative to the mean finger (1 = as much)."""
    fingers = mean[: len(FINGERS)]
    out = np.full(len(GROUPS), np.nan)
    out[: len(FINGERS)] = fingers / np.nansum(fingers)
    out[len(FINGERS) :] = mean[len(FINGERS) :] / np.nanmean(fingers)
    return out


def analyse(folder: Path) -> dict:
    sessions = [s for s in (load_session(m.parent) for m in sorted(folder.glob("*/*/meta.json"))) if s]
    if not sessions:
        return {}
    name, label_fs = sessions[0]["meta"]["dataset"], sessions[0]["meta"]["label_fs"]
    pooled = finger_stats(np.concatenate([s["speeds"] for s in sessions]))
    per_session = np.array([shares(finger_stats(s["speeds"])["mean"]) for s in sessions])

    cues = defaultdict(
        lambda: {
            "time_s": 0.0,
            "reps": 0,
            "sessions": set(),
            "speed": np.zeros(len(GROUPS)),
            "n": np.zeros(len(GROUPS)),
        }
    )
    names = {}
    for s in sessions:
        stimulus, meta = s["stimulus"], s["meta"]
        if (stimulus < 0).all():
            continue
        change = np.flatnonzero(np.diff(np.r_[-99, stimulus, -99]) != 0)
        starts = [a for a, _ in meta["segments"] if 0 < a < len(stimulus)]  # a repetition ends with its trial
        edges = np.union1d(change, np.asarray(starts, int))
        for start, end in zip(edges[:-1], edges[1:], strict=True):
            label = int(stimulus[start])
            if label < 0:
                continue
            cue = cues[label]
            names[label] = cue_name(name, label)
            cue["time_s"] += (end - start) / label_fs
            cue["reps"] += (end - start) / label_fs >= MIN_REP_S
            cue["sessions"].add(meta["key"])
            block = s["speeds"][start:end]
            finite = np.isfinite(block)
            cue["speed"] += np.where(finite, block, 0).sum(axis=0)
            cue["n"] += finite.sum(axis=0)
    return {
        "name": name,
        "pooled": pooled,
        "share": shares(pooled["mean"]),
        "session_share": per_session,
        "n_sessions": len(sessions),
        "hours": sum(len(s["speeds"]) for s in sessions) / label_fs / 3600,
        "cues": dict(sorted(cues.items())),
        "cue_names": names,
        "all_sessions": {s["meta"]["key"] for s in sessions},
    }


def pct(x: float) -> str:
    return "    -" if not np.isfinite(x) else f"{x:5.0%}"


def print_fingers(results: list[dict]):
    groups = list(GROUPS)
    head = "".join(f"{g:>8}" for g in groups)
    print("\nFINGER MOTION SHARE (fingers sum to 100%; spread, wrist: x the mean finger), [min..max over sessions]")
    print(f"{'':14}{head}   max/min finger")
    for r in results:
        share, lo, hi = r["share"], np.nanmin(r["session_share"], axis=0), np.nanmax(r["session_share"], axis=0)
        cells = "".join(f"{pct(v):>8}" if i < len(FINGERS) else f"{v:8.2f}" if np.isfinite(v) else f"{'-':>8}"
                        for i, v in enumerate(share))  # fmt: skip
        fingers = share[: len(FINGERS)]
        print(f"{r['name']:14}{cells}   {np.nanmax(fingers) / np.nanmin(fingers):.1f}x")
        spread = "".join(f"{f'{a:.0%}..{b:.0%}':>8}" if np.isfinite(a) else f"{'':>8}"
                         for a, b in zip(lo[: len(FINGERS)], hi[: len(FINGERS)], strict=True))  # fmt: skip
        print(f"{'':14}{spread}")
    for key, title in [
        ("active", f"TIME MOVING (speed > {ACTIVE} range/s)"),
        ("alone", "MOVING ALONE (of its moving time)"),
    ]:
        print(f"\n{title}\n{'':14}{head}")
        for r in results:
            print(f"{r['name']:14}" + "".join(f"{pct(v):>8}" for v in r["pooled"][key]))


def print_cues(results: list[dict], detail: bool):
    print("\nINSTRUCTED MOVEMENTS (time and repetitions per cue, rest apart)")
    print(f"{'':14}{'cues':>5}{'rest':>7}{'time per cue (s) min/med/max':>32}{'reps min/med/max':>20}  missing in")
    for r in results:
        moving = {k: v for k, v in r["cues"].items() if k > 0}
        if not moving:
            print(f"{r['name']:14}  no cues (free movement)")
            continue
        total = sum(v["time_s"] for v in r["cues"].values())
        rest = r["cues"].get(0, {"time_s": 0})["time_s"] / total
        t = np.array([v["time_s"] for v in moving.values()])
        reps = np.array([v["reps"] for v in moving.values()])
        missing = [
            r["cue_names"][k]
            for k, v in moving.items()
            if len(v["sessions"]) < max(len(m["sessions"]) for m in moving.values())
        ]
        print(
            f"{r['name']:14}{len(moving):>5}{rest:>7.0%}{f'{t.min():.0f} / {np.median(t):.0f} / {t.max():.0f}':>32}"
            f"{f'{reps.min()} / {np.median(reps):.0f} / {reps.max()}':>20}  {', '.join(missing) or '-'}"
        )
        if detail:
            for label, v in r["cues"].items():
                share = cue_share(v)
                fingers = " ".join(f"{g[:2]} {pct(x)}" for g, x in zip(FINGERS, share, strict=False))
                print(f"    {label:3} {r['cue_names'][label]:22} {v['time_s']:7.0f} s {v['reps']:4} reps  {fingers}")


def cue_share(cue: dict) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cue["n"] > 0, cue["speed"] / cue["n"], np.nan)
    return shares(mean)


def plot_fingers(results: list[dict], path: Path):
    groups, names = list(GROUPS), [r["name"] for r in results]
    panels = [
        ("share", "share of finger motion\n(spread, wrist: × mean finger)"),
        ("active", f"time moving (> {ACTIVE} range/s)"),
        ("alone", "moving alone\n(of the finger's moving time)"),
    ]
    style = {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(15, 1.2 + 0.55 * len(names)), sharey=True)
        for ax, (key, title) in zip(axes, panels, strict=True):
            values = np.array([r["share"] if key == "share" else r["pooled"][key] for r in results])
            shown = values.copy()
            if key == "share":
                shown[:, len(FINGERS) :] = np.nan  # other scale: written, not coloured
            ax.imshow(np.ma.masked_invalid(shown), cmap="Blues", vmin=0, vmax=np.nanmax(shown) * 1.15, aspect="auto")
            for i in range(len(names)):
                for j in range(len(groups)):
                    v = values[i, j]
                    if not np.isfinite(v):
                        text = "–"
                    elif key == "share" and j >= len(FINGERS):
                        text = f"{v:.2f}×"
                    else:
                        text = f"{v:.0%}"
                    dark = np.isfinite(shown[i, j]) and shown[i, j] > np.nanmax(shown) * 0.6
                    ax.text(j, i, text, ha="center", va="center", color="white" if dark else "#2b2b2b", fontsize=8.5)
            ax.axvline(len(FINGERS) - 0.5, color="white", lw=3)
            ax.set_xticks(range(len(groups)), groups)
            ax.set_yticks(range(len(names)), names)
            ax.set_title(title, loc="left", fontsize=9.5)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.suptitle("Finger balance per dataset (training targets, 0-1 per joint)", x=0.01, ha="left",
                     fontweight="bold")  # fmt: skip
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)


def plot_cues(results: list[dict], path: Path):
    cued = [r for r in results if any(k > 0 for k in r["cues"])]
    if not cued:
        return
    heights = [len(r["cues"]) for r in cued]
    style = {"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        fig, axes = plt.subplots(len(cued), 1, figsize=(8.5, 1.2 + 0.2 * sum(heights) + 0.6 * len(cued)),
                                 gridspec_kw={"height_ratios": heights}, squeeze=False)  # fmt: skip
        for ax, r in zip(axes[:, 0], cued, strict=True):
            labels = list(r["cues"])
            values = np.array([cue_share(r["cues"][k]) for k in labels])[:, : len(FINGERS)]
            ax.imshow(np.ma.masked_invalid(values), cmap="Blues", vmin=0, vmax=0.6, aspect="auto")
            for i in range(len(labels)):
                for j in range(len(FINGERS)):
                    if np.isfinite(values[i, j]):
                        ax.text(j, i, f"{values[i, j]:.0%}", ha="center", va="center", fontsize=7,
                                color="white" if values[i, j] > 0.36 else "#2b2b2b")  # fmt: skip
            ticks = [f"{r['cue_names'][k]}  ({r['cues'][k]['time_s'] / 60:.0f} min, {r['cues'][k]['reps']} reps)"
                     for k in labels]  # fmt: skip
            ax.set_yticks(range(len(labels)), ticks)
            ax.set_xticks(range(len(FINGERS)), list(FINGERS))
            ax.xaxis.tick_top()
            ax.tick_params(length=0)
            ax.set_title(f"{r['name']}: share of finger motion per cue", loc="left", fontsize=9.5, fontweight="bold")
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/interim"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"), help="for the shared gestures")
    parser.add_argument("--datasets", nargs="+", help="default: every dataset folder in --data")
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    parser.add_argument("--detail", action="store_true", help="one line per cue: time, repetitions, finger shares")
    args = parser.parse_args()

    names = args.datasets or sorted(p.name for p in args.data.iterdir() if p.name in DATASETS)
    results = [r for r in (analyse(args.data / name) for name in names) if r]
    for r in results:
        print(f"{r['name']}: {r['n_sessions']} sessions, {r['hours']:.1f} h, {r['pooled']['frames']} labelled frames")
    print_fingers(results)
    print_cues(results, args.detail)
    print_gestures(args.processed, names)

    args.out.mkdir(parents=True, exist_ok=True)
    plot_fingers(results, args.out / "movement_balance_fingers.png")
    plot_cues(results, args.out / "movement_balance_cues.png")
    print(f"\nsaved movement_balance_fingers.png and movement_balance_cues.png in {args.out}")


if __name__ == "__main__":
    main()
