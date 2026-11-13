import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

from femg.joints import CANONICAL_JOINTS, JOINT_INDEX

FINGERS = [JOINT_INDEX[f"{f}_{j}_flex"] for f in ["index", "middle", "ring", "pinky"] for j in ["mcp", "pip"]]
SMOOTH = 10  # label frames (100 ms) for the RMS and the hand speed


def scenarios(kin: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean mask over label frames for every scenario."""
    flexion = np.nanmean(kin[:, FINGERS], axis=1)  # how closed the hand is
    speed = uniform_filter1d(np.abs(np.gradient(flexion)), SMOOTH)  # how fast it closes/opens
    p = lambda x, q: np.nanpercentile(x, q)  # noqa: E731
    return {
        "all": np.ones(len(kin), bool),
        "still": speed <= p(speed, 25),
        "moving": speed >= p(speed, 75),
        "open": flexion <= p(flexion, 10),
        "closed": flexion >= p(flexion, 90),
    }


def load_session(folder: Path) -> dict:
    """Per label frame: ring8 RMS (µV), the same RMS / calib_rms, the joints, and the scenario masks."""
    meta = json.loads((folder / "meta.json").read_text())
    ring = meta["montages"]["ring8"]
    kin = np.load(folder / "kin.npy")
    emg = np.load(folder / "emg.npy", mmap_mode="r")[:, ring]
    ratio = meta["fs"] // meta["label_fs"]
    power = (emg[: len(kin) * ratio] ** 2).reshape(len(kin), ratio, len(ring)).mean(axis=1)
    rms = np.sqrt(uniform_filter1d(power, SMOOTH, axis=0))  # (frames, 8) volts
    calib = np.asarray(meta["emg_stats"]["calib_rms"])[ring]
    return {
        "raw": rms.mean(axis=1) * 1e6,
        "norm": (rms / calib).mean(axis=1),
        "kin": kin,
        "masks": scenarios(kin),
    }


def load_dataset(folder: Path, max_frames: int = 200_000) -> dict:
    """All sessions of a dataset concatenated, subsampled to max_frames so every dataset weighs the same."""
    sessions = [load_session(meta.parent) for meta in sorted(folder.glob("*/*/meta.json"))]
    data = {k: np.concatenate([s[k] for s in sessions]) for k in ["raw", "norm", "kin"]}
    data["masks"] = {k: np.concatenate([s["masks"][k] for s in sessions]) for k in sessions[0]["masks"]}
    keep = np.random.default_rng(0).permutation(len(data["kin"]))[:max_frames]
    return {k: (v[keep] if k != "masks" else {m: x[keep] for m, x in v.items()}) for k, v in data.items()}


def pool(datasets: dict) -> dict:
    first = next(iter(datasets.values()))
    pooled = {k: np.concatenate([d[k] for d in datasets.values()]) for k in ["raw", "norm", "kin"]}
    pooled["masks"] = {m: np.concatenate([d["masks"][m] for d in datasets.values()]) for m in first["masks"]}
    return pooled


def grouped_boxes(ax, groups: list[str], values: dict[str, list[np.ndarray]], colors: dict[str, str]):
    """One box per (group, dataset): whiskers at the 1st and 99th percentiles, no outliers."""
    width = 0.8 / len(values)
    for i, (name, series) in enumerate(values.items()):
        positions = np.arange(len(groups)) + (i - (len(values) - 1) / 2) * width
        series = [x[np.isfinite(x)] if np.isfinite(x).any() else [np.nan] for x in series]
        box = ax.boxplot(
            series,
            positions=positions,
            widths=width * 0.85,
            whis=(1, 99),
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "black"},
        )
        for patch in box["boxes"]:
            patch.set(facecolor=colors[name], edgecolor=colors[name], alpha=0.8)
        ax.plot([], [], "s", color=colors[name], label=name)
    ax.set_xticks(range(len(groups)), groups)
    ax.grid(axis="y", alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--datasets", nargs="+", help="default: every folder in --data")
    parser.add_argument("--out", type=Path, default=Path("output/compare"))
    args = parser.parse_args()

    names = args.datasets or sorted(p.name for p in args.data.iterdir() if p.is_dir())
    datasets = {name: load_dataset(args.data / name) for name in names}
    datasets["all"] = pool(datasets)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100"]
    colors = {name: palette[i] for i, name in enumerate(names)} | {"all": "#8a8985"}
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. EMG amplitude per scenario
    groups = list(datasets["all"]["masks"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, key, label in [(axes[0], "raw", "ring8 RMS (µV, log)"), (axes[1], "norm", "ring8 RMS / calib_rms")]:
        values = {n: [d[key][d["masks"][g]] for g in groups] for n, d in datasets.items()}
        grouped_boxes(ax, groups, values, colors)
        ax.set(yscale="log" if key == "raw" else "linear", ylabel=label, xlabel="scenario")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("EMG amplitude per scenario (100 ms RMS, mean over the 8 ring channels)")
    fig.tight_layout()
    fig.savefig(args.out / "emg.png", dpi=130)

    # 2. Joint angles
    fig, ax = plt.subplots(figsize=(15, 4.5))
    values = {n: [d["kin"][:, j] for j in range(len(CANONICAL_JOINTS))] for n, d in datasets.items()}
    grouped_boxes(ax, CANONICAL_JOINTS, values, colors)
    ax.set_xticks(range(len(CANONICAL_JOINTS)), CANONICAL_JOINTS, rotation=60, ha="right")
    ax.set(ylabel="angle (rad)", ylim=(-1.5, 3.5))  # clipped: a glove glitch can reach ±10 rad
    ax.set_title("Joint angles (whiskers: 1st-99th percentile; empty = not measured; y clipped to [-1.5, 3.5])")
    ax.legend(frameon=False, ncol=len(datasets))
    fig.tight_layout()
    fig.savefig(args.out / "kin.png", dpi=130)
    print(f"saved {args.out}/emg.png and {args.out}/kin.png")


if __name__ == "__main__":
    main()
