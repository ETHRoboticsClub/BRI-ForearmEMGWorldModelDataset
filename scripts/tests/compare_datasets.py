import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from femg.datasets import DATASETS
from femg.hand.joints import CANONICAL_JOINTS
from femg.normalize import (
    Normalizer,
    bouts,
    display_montage,
    hand_flexion,
    hand_speed,
    has_kinematics,
    scenarios,
)
from femg.signal_utils import envelope

PERCENTILES = np.arange(1, 100)
CHECKED = (PERCENTILES >= 10) & (PERCENTILES <= 90)
SCENARIOS = ["all", "rest", "hold", "moving"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100", "#8a5cd1", "#1aa3b8", "#b5651d", "#6b8e23", "#d43f3a"]
POOL_COLOR = "#5f5e5a"
KIN_EVERY = 10  # joint angles: one label frame in 10 (0.1 s), enough for their distribution

# metric: (y label, log scale, deviation kind, tolerance); kind None = plotted only
METRICS = {
    **{f"raw|{s}": ("display montage RMS (µV)", True, None, None) for s in SCENARIOS},
    "emg|all": ("display montage RMS / active", True, None, None),
    **{f"emg|{s}": ("display montage RMS / active", True, "ratio", 0.25) for s in SCENARIOS[1:]},
    "flexion": ("hand flexion (0 open, 1 closed)", False, "difference", 0.10),
    "speed": ("hand speed while moving (flexion / s)", True, "ratio", 0.25),
    "bout": ("movement bout duration (s)", True, "ratio", 0.25),
}


def load_session(folder: Path) -> dict[str, np.ndarray]:
    """EMG over the display montage (ring8, else the dataset's small montage). Sessions without kinematics (cue only)
    give EMG only, with rest = instructed rest (stimulus 0) instead of still + open hand."""
    meta = json.loads((folder / "meta.json").read_text())
    ring, label_fs = display_montage(meta)[1], meta["label_fs"]
    window = round(meta["processor"]["rms_window_s"] * label_fs)

    emg = np.load(folder / "emg.npy", mmap_mode="r")[:, ring]
    rms = envelope(emg, meta["fs"] // label_fs, window)
    raw_uv = rms.mean(axis=1) * 1e6
    normalized = Normalizer.emg(rms, meta["emg_stats"], ring).mean(axis=1)

    if not has_kinematics(meta):
        stimulus = np.load(folder / "stimulus.npy")[: len(raw_uv)]
        none = np.zeros(len(raw_uv), bool)
        masks = {"all": ~none, "rest": stimulus == 0, "hold": none, "moving": none}
        metrics = {
            f"{kind}|{s}": values[masks[s]]
            for s in SCENARIOS
            for kind, values in [("raw", raw_uv), ("emg", normalized)]
        }
        empty = np.array([])
        return metrics | {
            "flexion": empty,
            "speed": empty,
            "bout": empty,
            "joints": np.empty((0, len(CANONICAL_JOINTS))),
        }

    radians = np.load(folder / "kin.npy")
    kin = Normalizer.kin(radians, meta["kin_stats"])
    flexion = hand_flexion(kin)
    speed = hand_speed(flexion, label_fs)
    masks = scenarios(flexion, speed)

    metrics = {}
    for s in SCENARIOS:
        metrics[f"raw|{s}"] = raw_uv[masks[s]]
        metrics[f"emg|{s}"] = normalized[masks[s]]
    metrics["flexion"] = flexion
    metrics["speed"] = speed[masks["moving"]]
    metrics["bout"] = bouts(masks["moving"], label_fs)
    metrics["joints"] = radians[::KIN_EVERY]
    return metrics


def load_dataset(folder: Path) -> dict[str, np.ndarray]:
    metas = sorted(folder.glob("*/*/meta.json"))
    sessions = [load_session(meta.parent) for meta in metas]
    out = {metric: np.concatenate([s[metric] for s in sessions]) for metric in [*METRICS, "joints"]}
    out["kin_units"] = json.loads(metas[0].read_text()).get("kin_units") if metas else None
    return out


def equal_weight_pool(values: dict[str, np.ndarray], rng) -> np.ndarray:
    n = min(len(v) for v in values.values())
    return np.concatenate([rng.choice(v, n, replace=False) for v in values.values()])


def percentile_curves(datasets: dict) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    curves = {}
    for metric in METRICS:
        values = {name: d[metric][np.isfinite(d[metric])] for name, d in datasets.items()}
        values = {name: v for name, v in values.items() if len(v)}  # e.g. hand metrics of cue-only datasets
        if values:
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


def plot_kin(datasets: dict, colors: dict, path: Path):
    """Joint angles of every dataset, in the units of its labels: one box per joint and dataset (whiskers at the
    1st and 99th percentiles, no outliers; empty: not measured)."""
    names = list(datasets)
    width = 0.84 / len(names)
    style = {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#a8a7a2"}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(18, 5.2))
        for i, name in enumerate(names):
            joints = datasets[name]["joints"]
            positions = np.arange(len(CANONICAL_JOINTS)) + (i - (len(names) - 1) / 2) * width
            for j, at in enumerate(positions):
                x = joints[:, j][np.isfinite(joints[:, j])] if len(joints) else np.array([])
                if len(x) == 0:
                    continue
                box = ax.boxplot(
                    [x],
                    positions=[at],
                    widths=width * 0.85,
                    whis=(1, 99),
                    showfliers=False,
                    patch_artist=True,
                    medianprops={"color": "white", "lw": 1.2},
                    whiskerprops={"color": colors[name], "lw": 0.9},
                    capprops={"color": colors[name], "lw": 0.9},
                )
                box["boxes"][0].set(facecolor=colors[name], edgecolor=colors[name], alpha=0.85)
            units = datasets[name]["kin_units"] or "no kinematics"
            ax.plot([], [], "s", color=colors[name], ms=8, label=f"{name} ({units.replace('_', ' ')})")
        for j in range(1, len(CANONICAL_JOINTS)):
            ax.axvline(j - 0.5, color="#ecebe7", lw=0.6, zorder=0)
        ax.axhline(0, color="#d8d7d2", lw=0.8, zorder=0)
        ax.set_xticks(range(len(CANONICAL_JOINTS)), CANONICAL_JOINTS, rotation=55, ha="right")
        ax.set_xlim(-0.6, len(CANONICAL_JOINTS) - 0.4)
        ax.set(ylabel="angle (rad)", ylim=(-1.5, 3.5))  # clipped: a glove glitch can reach ±10 rad
        ax.legend(frameon=False, ncol=len(names), loc="upper left", fontsize=8.5)
        ax.set_title(
            "Joint angles per dataset (interim labels; boxes: quartiles, whiskers: 1st-99th percentile; empty: not "
            "measured; y clipped to [-1.5, 3.5])",
            loc="left",
            fontsize=10,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def print_acceptance(curves, names: list[str]):
    print(f"worst deviation from the pool over p10..p90\n{'':22}" + "".join(f"{n:>16}" for n in names))
    for metric, (_, _, kind, tolerance) in METRICS.items():
        if kind is None:
            continue
        cells = []
        for name in names:
            if name not in curves[metric]:
                cells.append("-")
                continue
            dev = deviation(curves[metric][name], curves[metric]["all"], kind)
            value = f"{dev:.0%}" if kind == "ratio" else f"{dev:.2f}"
            cells.append(f"{value} {'PASS' if dev <= tolerance else 'FAIL'}")
        print(f"{metric + f' (≤{tolerance:g})':22}" + "".join(f"{c:>16}" for c in cells))


def main():
    parser = argparse.ArgumentParser(description="Compare datasets percentile by percentile, per scenario.")
    parser.add_argument("--data", type=Path, default=Path("data/interim"))
    parser.add_argument("--datasets", nargs="+", help="default: every folder in --data")
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    names = args.datasets or sorted(p.name for p in args.data.iterdir() if p.name in DATASETS)
    datasets = {name: load_dataset(args.data / name) for name in names}
    curves = percentile_curves(datasets)
    fixed = {
        "emg_fk": "#2a78d6",
        "ninapro_db2": "#eb6834",
        "ninapro_db7": "#1baf7a",
        "ninapro_db8": "#8a5cd1",
        "great": "#eda100",
    }  # as in docs/figures/calib_baseline.png
    others = iter(c for c in COLORS if c not in fixed.values())
    colors = {name: fixed.get(name) or next(others) for name in names} | {"all": POOL_COLOR}

    args.out.mkdir(parents=True, exist_ok=True)
    plot_emg(curves, colors, args.out / "dataset_emg.png")
    plot_hand(curves, colors, args.out / "dataset_hand.png")
    plot_kin(datasets, colors, args.out / "dataset_kin.png")
    print_acceptance(curves, names)
    print(f"\nsaved dataset_emg.png, dataset_hand.png and dataset_kin.png in {args.out}")


if __name__ == "__main__":
    main()
