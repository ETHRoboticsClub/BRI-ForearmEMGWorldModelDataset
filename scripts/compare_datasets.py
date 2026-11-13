import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from femg.normalize import bouts, hand_flexion, hand_speed, normalize_emg, normalize_kin, scenarios
from femg.signal_utils import envelope

PERCENTILES = np.arange(1, 100)
CHECKED = (PERCENTILES >= 10) & (PERCENTILES <= 90)
SCENARIOS = ["all", "rest", "hold", "moving"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100"]
POOL_COLOR = "#5f5e5a"

# metric: (y label, log scale, deviation kind, tolerance); kind None = plotted only
METRICS = {
    **{f"raw|{s}": ("ring8 RMS (µV)", True, None, None) for s in SCENARIOS},
    "emg|all": ("ring8 RMS / active", True, None, None),
    **{f"emg|{s}": ("ring8 RMS / active", True, "ratio", 0.25) for s in SCENARIOS[1:]},
    "flexion": ("hand flexion (0 open, 1 closed)", False, "difference", 0.10),
    "speed": ("hand speed while moving (flexion / s)", True, "ratio", 0.25),
    "bout": ("movement bout duration (s)", True, "ratio", 0.25),
}


def load_session(folder: Path) -> dict[str, np.ndarray]:
    meta = json.loads((folder / "meta.json").read_text())
    ring, label_fs = meta["montages"]["ring8"], meta["label_fs"]
    window = round(meta["processor"]["rms_window_s"] * label_fs)

    emg = np.load(folder / "emg.npy", mmap_mode="r")[:, ring]
    rms = envelope(emg, meta["fs"] // label_fs, window)
    raw_uv = rms.mean(axis=1) * 1e6
    normalized = normalize_emg(rms, meta["emg_stats"], ring).mean(axis=1)

    kin = normalize_kin(np.load(folder / "kin.npy"), meta["kin_stats"])
    flexion = hand_flexion(kin)
    speed = hand_speed(flexion, label_fs, window)
    masks = scenarios(flexion, speed)

    metrics = {}
    for s in SCENARIOS:
        metrics[f"raw|{s}"] = raw_uv[masks[s]]
        metrics[f"emg|{s}"] = normalized[masks[s]]
    metrics["flexion"] = flexion
    metrics["speed"] = speed[masks["moving"]]
    metrics["bout"] = bouts(masks["moving"], label_fs)
    return metrics


def load_dataset(folder: Path) -> dict[str, np.ndarray]:
    sessions = [load_session(meta.parent) for meta in sorted(folder.glob("*/*/meta.json"))]
    return {metric: np.concatenate([s[metric] for s in sessions]) for metric in METRICS}


def equal_weight_pool(values: dict[str, np.ndarray], rng) -> np.ndarray:
    n = min(len(v) for v in values.values())
    return np.concatenate([rng.choice(v, n, replace=False) for v in values.values()])


def percentile_curves(datasets: dict) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    curves = {}
    for metric in METRICS:
        values = {name: d[metric][np.isfinite(d[metric])] for name, d in datasets.items()}
        values["all"] = equal_weight_pool(values, rng)
        curves[metric] = {name: np.percentile(v, PERCENTILES) for name, v in values.items()}
    return curves


def deviation(curve: np.ndarray, pool: np.ndarray, kind: str) -> float:
    if kind == "ratio":
        return float(np.max(np.abs(curve[CHECKED] / pool[CHECKED] - 1)))
    return float(np.max(np.abs(curve[CHECKED] - pool[CHECKED])))


def plot_curves(ax, curves: dict, colors: dict, metric: str):
    label, log, _, _ = METRICS[metric]
    for name, curve in curves.items():
        pooled = name == "all"
        ax.plot(PERCENTILES, curve, label=name, color=colors[name], lw=1.5 if pooled else 2, ls="--" if pooled else "-")
    ax.axvspan(10, 90, color="0.5", alpha=0.06)
    ax.set(xlabel="percentile", ylabel=label, yscale="log" if log else "linear")
    ax.grid(alpha=0.3)


def plot_emg(curves, colors, path: Path):
    fig, axes = plt.subplots(2, len(SCENARIOS), figsize=(4 * len(SCENARIOS), 7), sharey="row")
    for col, scenario in enumerate(SCENARIOS):
        plot_curves(axes[0, col], curves[f"raw|{scenario}"], colors, f"raw|{scenario}")
        plot_curves(axes[1, col], curves[f"emg|{scenario}"], colors, f"emg|{scenario}")
        axes[0, col].set_title(scenario)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("EMG amplitude percentiles: raw (top), / active (bottom); shaded = checked range")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def plot_hand(curves, colors, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ["flexion", "speed", "bout"], strict=True):
        plot_curves(ax, curves[metric], colors, metric)
    axes[0].legend(frameon=False)
    fig.suptitle("Hand movement percentiles (0 = open, 1 = closed)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def print_acceptance(curves, names: list[str]):
    print(f"worst deviation from the pool over p10..p90\n{'':22}" + "".join(f"{n:>16}" for n in names))
    for metric, (_, _, kind, tolerance) in METRICS.items():
        if kind is None:
            continue
        cells = []
        for name in names:
            dev = deviation(curves[metric][name], curves[metric]["all"], kind)
            value = f"{dev:.0%}" if kind == "ratio" else f"{dev:.2f}"
            cells.append(f"{value} {'PASS' if dev <= tolerance else 'FAIL'}")
        print(f"{metric + f' (≤{tolerance:g})':22}" + "".join(f"{c:>16}" for c in cells))


def main():
    parser = argparse.ArgumentParser(description="Compare datasets percentile by percentile, per scenario.")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--datasets", nargs="+", help="default: every folder in --data")
    parser.add_argument("--out", type=Path, default=Path("output/compare"))
    args = parser.parse_args()

    names = args.datasets or sorted(p.name for p in args.data.iterdir() if p.is_dir())
    curves = percentile_curves({name: load_dataset(args.data / name) for name in names})
    colors = {name: COLORS[i] for i, name in enumerate(names)} | {"all": POOL_COLOR}

    args.out.mkdir(parents=True, exist_ok=True)
    plot_emg(curves, colors, args.out / "emg.png")
    plot_hand(curves, colors, args.out / "hand.png")
    print_acceptance(curves, names)
    print(f"\nsaved {args.out}/emg.png and {args.out}/hand.png")


if __name__ == "__main__":
    main()
