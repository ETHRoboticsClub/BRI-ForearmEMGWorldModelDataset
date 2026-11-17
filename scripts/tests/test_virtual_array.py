import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from scipy.stats import spearmanr

from femg.datasets import DATASETS
from femg.electrodes import FOREARM_CIRCUMFERENCE_CM, FOREARM_LENGTH_CM, ElectrodeLayout
from femg.normalize import Normalizer, as_array, hand_flexion, has_kinematics
from femg.signal_utils import envelope
from femg.virtual_array import VirtualArray

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100", "#8a5cd1", "#1aa3b8", "#b5651d", "#6b8e23"]
INK = "#28282a"
DISTANCE_BINS = np.arange(0, 16.5, 1.5)  # cm
LAG_S = 0.2  # the pose trails the EMG (scripts/tests/estimate_kin_lag.py)
FOLDS = 5


def log_envelope(x: np.ndarray, meta: dict) -> np.ndarray:
    """(T, C) EMG -> (T / per label frame, C) log RMS, the processor's window: a quality measure, not a model input."""
    per_label = meta["fs"] // meta["label_fs"]
    window = round(meta["processor"]["rms_window_s"] * meta["label_fs"])
    return np.log(envelope(x, per_label, window).astype(np.float64) + 1e-9)


def load(folder: Path, max_s: float, array: VirtualArray, interim: Path) -> dict:
    """One processed session: the virtual EMG the build saved (each virtual channel / its own active level), the
    joints in 0-1, and the real EMG / active level per channel from its interim session (interim: data/interim)."""
    meta = json.loads((folder / "meta.json").read_text())
    n = int(max_s * meta["fs"])
    real = interim / meta["key"] / "emg.npy"
    emg = Normalizer.emg(np.load(real, mmap_mode="r")[:n], meta["emg_stats"])
    layout = ElectrodeLayout.from_meta(meta)
    positions = layout.positions()
    active = as_array(meta["virtual_stats"]["active"])
    gains = np.divide(1, active, out=np.zeros_like(active), where=active > 0).reshape(array.shape)
    virtual = np.asarray(np.load(folder / "virtual.npy", mmap_mode="r")[:n], np.float32)
    out = {
        "meta": meta,
        "emg": emg,
        "layout": layout,
        "positions": positions,
        "inputs": array.inputs(layout),
        "groups": np.array([e["group"] for e in meta["electrodes"]]),
        "gains": gains,
        "covered": array.covered(array.inputs(layout)),
        "virtual": virtual.reshape(len(virtual), *array.shape),
        "log_env": log_envelope(emg, meta),
        "virtual_log_env": log_envelope(virtual, meta),
    }
    if has_kinematics(meta):
        labels = np.load(folder / "labels.npy")[: len(out["log_env"])]
        out["pose"] = labels[:, meta["measured_joints"]].astype(np.float64)
        out["flexion"] = hand_flexion(labels).astype(np.float64)
    return out


def load_all(data: Path, array: VirtualArray, per_dataset: int, max_s: float, interim: Path) -> dict[str, list[dict]]:
    sessions = {}
    for dataset in sorted(p for p in data.iterdir() if p.name in DATASETS):
        folders = [f.parent for f in sorted(dataset.glob("*/*/meta.json"))[:per_dataset]]
        built = [json.loads((f / "meta.json").read_text())["processor"].get("virtual_array") for f in folders]
        if any(b != array.settings() for b in built):
            print(f"{dataset.name}: built with another virtual array, rebuild it (build_dataset.py --overwrite)")
            continue
        if folders:
            sessions[dataset.name] = [load(f, max_s, array, interim) for f in folders]
    return sessions


# Checks of the virtual array


def array_checks(array: VirtualArray, sessions: dict[str, list[dict]]) -> list:
    checks = []
    rings = [array.inputs(DATASETS[name]().config.electrodes)[:, 1] for name in sessions]
    mean = np.mean([np.nanmean(z) for z in rings])
    ok = all(abs(h - mean) < 0.01 for h in array.heights)
    detail = f"heights {list(array.heights)}, mean ring z of the datasets {mean:.3f}"
    checks.append(("virtual ring at the datasets' mean ring height", ok, detail))
    one = {name: runs[0] for name, runs in sessions.items()}
    shapes = {s["virtual"].shape[1:] for s in one.values()}
    counts = sorted({s["meta"]["n_channels"] for s in one.values()})
    checks.append(("same virtual array for every layout", len(shapes) == 1, f"{counts} channels -> {shapes}"))

    for name, s in one.items():
        again = array(s["emg"], s["inputs"], s["gains"])
        error = np.abs(again - s["virtual"]).max()
        checks.append((f"{name}: built virtual EMG = matrix @ EMG", error < 1e-3, f"max difference {error:.1e}"))

    s = one.get("ninapro_db2") or next(iter(one.values()))
    emg, positions = s["emg"][: 4 * s["meta"]["fs"]], s["inputs"]
    reference = array(emg, positions)

    order = np.random.default_rng(0).permutation(len(positions))
    ok = np.allclose(array(emg[:, order], positions[order]), reference, atol=1e-5)
    checks.append(("channel order ignored", ok, "channels shuffled with their positions"))

    noise = np.random.default_rng(1).normal(size=(len(emg), 4)).astype(np.float32)
    unknown = np.tile(np.float32([[np.nan, np.nan, 1.0]]), (4, 1))
    ok = np.allclose(array(np.c_[emg, noise], np.r_[positions, unknown]), reference, atol=1e-5)
    checks.append(("channels with unknown positions ignored", ok, "4 noise channels added at unknown positions"))

    step = 360 / array.n_angles
    ok = np.allclose(array(emg, positions, rotation=step), np.roll(reference, 1, axis=-1), atol=1e-5)
    checks.append(("a band turned by one step rolls the array", ok, f"rotation {step:g} deg = one column"))

    flipped = reference[..., (-np.arange(array.n_angles)) % array.n_angles]
    ok = np.allclose(array(emg, positions, mirror=True), flipped, atol=1e-5)
    checks.append(("a band running the other way flips the array", ok, "mirror: virtual column i -> -i"))

    if "emg_fk" in one:
        s = one["emg_fk"]
        emg, positions = s["emg"][: 4 * s["meta"]["fs"]], s["inputs"]
        shared = np.random.default_rng(2).normal(size=(len(emg), 1)).astype(np.float32)
        ok = np.allclose(array(emg + shared, positions), array(emg, positions), atol=1e-4)
        checks.append(("monopolar common signal removed (EMG-FK)", ok, "the same noise added to every channel"))

    for name, s in one.items():
        rings = s["covered"].all(axis=1)
        full = ", ".join(f"{z:g}" for z in np.array(array.heights)[rings]) or "none"
        detail = f"{s['covered'].sum()} of {s['covered'].size} virtual electrodes, full rings at z: {full}"
        checks.append((f"{name}: covers a full ring", rings.any(), detail))
    return checks


# Quality of the virtual signals


def spatial_decay(runs: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Envelope correlation of the channel pairs of a band vs their distance, for pairs side by side around the
    forearm ("around") and one above the other ("along"). At every frame the band's mean log envelope is removed
    first: the overall effort that raises every channel together and hides how much neighbours share."""
    pairs = {"around": ([], []), "along": ([], [])}
    for s in runs:
        p, env = s["positions"], s["log_env"].copy()
        for g in np.unique(s["groups"]):
            env[:, s["groups"] == g] -= env[:, s["groups"] == g].mean(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(env.T)
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                same_band = s["groups"][i] == s["groups"][j] and np.isfinite(corr[i, j])
                if not same_band or not np.isfinite(p[[i, j], :2]).all():
                    continue
                around = abs((p[i, 0] - p[j, 0] + 180) % 360 - 180) / 360 * FOREARM_CIRCUMFERENCE_CM
                along = abs(p[i, 1] - p[j, 1]) * FOREARM_LENGTH_CM
                if along < 0.1:
                    pairs["around"][0].append(around), pairs["around"][1].append(corr[i, j])
                elif around < 0.1:
                    pairs["along"][0].append(along), pairs["along"][1].append(corr[i, j])
    return {k: (np.array(d), np.array(c)) for k, (d, c) in pairs.items()}


def leave_one_out(
    array: VirtualArray, runs: list[dict], max_s: float = 120, other_height: bool = False
) -> tuple[float, float]:
    """Each real channel on the forearm rebuilt from the others with the array's kernel, as a virtual electrode at
    its place would be: the median correlation of the rebuilt and the real channel, of the raw signal and of the
    envelope (log RMS). Harder than the real case: on a ring of 8 the nearest others are 3 cm away, while a virtual
    electrode is at most 1.5 cm from a real one. other_height: only from the electrodes at another height (two-ring
    layouts: one ring from the other of its band, 2.5 cm away: how well a ring stands in for the virtual one at
    another height, as NinaPro's and EMG-FK's single rings, 2.1 and 2.6 cm from it)."""
    raw, env = [], []
    for s in runs:
        x, p = s["emg"][: int(max_s * s["meta"]["fs"])], s["inputs"]
        for c in np.flatnonzero(np.isfinite(p[:, :2]).all(axis=1) & (p[:, 1] >= 0)):
            others = p.copy()
            others[c, :2] = np.nan
            if other_height:
                others[(np.abs(p[:, 1] - p[c, 1]) < 0.01) | (s["groups"] != s["groups"][c]), :2] = np.nan
            k = array.kernel(p[c, :2], others)
            if k.sum() < 1e-3:
                continue
            rebuilt = x @ ((k / k.sum()) @ array.reference(others))
            target = x[:, c]
            monopolar = np.isfinite(others[:, 0]) & (others[:, 2] == 0)
            if array.common_average and p[c, 2] == 0 and monopolar.any():
                target = target - x[:, monopolar].mean(axis=1)
            pair = np.c_[target, rebuilt]
            raw.append(np.corrcoef(pair.T)[0, 1])
            env.append(np.corrcoef(log_envelope(pair, s["meta"]).T)[0, 1])
    if not env:
        return np.nan, np.nan
    return float(np.median(raw)), float(np.median(env))


def sources(array: VirtualArray, s: dict) -> dict[str, float]:
    """Share of the virtual electrodes' weight (before the gains) coming from each group of real electrodes, rings
    split by height: {"rings z 0.2": 0.66, ...}."""
    w = array.weights(s["inputs"]).reshape(-1, len(s["inputs"]))
    w = (w / w.sum(axis=1, keepdims=True)).sum(axis=0) / len(w)
    z = s["positions"][:, 1].round(2)
    names = [f"{g} z {h:g}" if np.isfinite(h) else str(g) for g, h in zip(s["groups"], z, strict=True)]
    out = defaultdict(float)
    for name, share in zip(names, w, strict=True):
        out[name] += share
    return {k: v for k, v in out.items() if v > 0.005}


def spectrum(runs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Mean power spectrum of the covered virtual channels, as a share of their total power."""
    shares = []
    for s in runs:
        v = s["virtual"][:, s["covered"]]
        f, p = welch(v, fs=s["meta"]["fs"], nperseg=256, axis=0)
        shares.append((p / p.sum(axis=0)).mean(axis=1))
    return f, np.mean(shares, axis=0)


def decoding_r2(x: np.ndarray, y: np.ndarray, label_fs: float, ridge: float = 10.0) -> float:
    """Cross-validated R2 of a ridge decoder: features at t -> pose at t + LAG_S (5 contiguous folds)."""
    lag = round(LAG_S * label_fs)
    x, y = x[:-lag], y[lag:]
    ok = np.isfinite(y).all(axis=1) & np.isfinite(x).all(axis=1)
    x, y = x[ok], y[ok]
    fold = np.arange(len(x)) * FOLDS // len(x)
    error = total = 0.0
    for f in range(FOLDS):
        train, test = fold != f, fold == f
        mean, std = x[train].mean(axis=0), x[train].std(axis=0) + 1e-9
        a = np.c_[(x[train] - mean) / std, np.ones(train.sum())]
        b = np.c_[(x[test] - mean) / std, np.ones(test.sum())]
        weights = np.linalg.solve(a.T @ a + ridge * np.eye(a.shape[1]), a.T @ y[train])
        error += ((y[test] - b @ weights) ** 2).sum()
        total += ((y[test] - y[train].mean(axis=0)) ** 2).sum()
    return 1 - error / total


def pose_r2(runs: list[dict], heights: tuple[float, ...]) -> tuple[float, float, float]:
    """Pose R2 from the envelopes of all real channels, of the real ring of 8 nearest the virtual ring (as many
    channels, about the same place) and of the virtual channels: how much the resampling keeps."""
    real, ring, virtual = [], [], []
    for s in runs:
        fs, z = s["meta"]["label_fs"], s["positions"][:, 1].round(2)
        real.append(decoding_r2(s["log_env"][:, np.isfinite(s["positions"][:, 0])], s["pose"], fs))
        rings = [h for h in np.unique(z[np.isfinite(z)]) if (z == h).sum() == 8]
        nearest = min(rings, key=lambda h: min(abs(h - v) for v in heights))
        ring.append(decoding_r2(s["log_env"][:, z == nearest], s["pose"], fs))
        virtual.append(decoding_r2(s["virtual_log_env"][:, s["covered"].ravel()], s["pose"], fs))
    return float(np.mean(real)), float(np.mean(ring)), float(np.mean(virtual))


# Figures


def around(angle):
    """Degrees -> -180..180, palmar (0) in the middle, so a grid over the flexors is not cut in two."""
    return (np.asarray(angle) + 180) % 360 - 180


def draw_section(ax, layout: ElectrodeLayout):
    """The forearm cut across, seen from the hand: palmar down, radial right. One circle per ring height; the number
    next to each electrode is its channel. Uncertainty under the circle (spots: shaded arc of +- angle_sd)."""
    t = np.linspace(0, 2 * np.pi, 200)
    ax.fill(np.sin(t), -np.cos(t), color="#f3e6dc", zorder=0)
    for text, (x, y) in {"palmar": (0, -1.75), "radial": (1.75, 0), "dorsal": (0, 1.75), "ulnar": (-1.75, 0)}.items():
        ax.text(x, y, text, ha="center", va="center", fontsize=7, color="#6b6a66")
    rings = [g for g in layout.groups if 1 < len(g) <= 16 and np.isfinite(g.angles).all()]
    heights = sorted({round(z, 3) for g in rings for z in g.zs if z >= 0})
    notes, channel = [], 0
    for i, g in enumerate(layout.groups):
        angles, zs = np.array(g.angles), np.array(g.zs)
        known = np.isfinite(angles) & (zs >= 0)
        color = COLORS[i % len(COLORS)]
        if not np.isfinite(angles).any():
            notes.append(f"{g.name}: {len(g)} electrodes, position unknown")
        elif (zs < 0).all():
            notes.append(f"{g.name}: upper arm")
        elif len(g) > 16:  # a grid: its angular extent
            span = np.deg2rad(np.linspace(around(angles).min(), around(angles).max(), 50))
            ax.plot(1.08 * np.sin(span), -1.08 * np.cos(span), color=color, lw=6, solid_capstyle="butt")
            ax.text(0, -0.55, f"{g.name}\n{len(g)} electrodes", ha="center", fontsize=7, color=color)
        else:
            for c in np.flatnonzero(known):
                r = 1 + 0.22 * heights.index(round(zs[c], 3)) if g in rings else 1.45  # spots: outside
                x, y = np.sin(np.deg2rad(angles[c])), -np.cos(np.deg2rad(angles[c]))
                ax.plot(r * x, r * y, "o", color=color, ms=6, mec="white", mew=0.5)
                ax.text((r + 0.17) * x, (r + 0.17) * y, channel + c + 1, ha="center", va="center", fontsize=6)
                if 0 < g.angle_sd < 180:
                    arc = np.deg2rad(angles[c] + np.linspace(-g.angle_sd, g.angle_sd, 30))
                    ax.plot(r * np.sin(arc), -r * np.cos(arc), color=color, lw=4, alpha=0.25, solid_capstyle="butt")
        if g.angle_sd >= 180 and np.isfinite(angles).any():
            notes.append(f"{g.name}: any rotation" + (", either direction" if g.mirror else ""))
        channel += len(g)
    if len(heights) > 1:
        notes.append("rings: inner z " + ", outer z ".join(f"{h:g}" for h in heights))
    ax.text(0, -2.05, "\n".join(notes), ha="center", va="top", fontsize=7)
    ax.set(xlim=(-2.1, 2.1), ylim=(-2.1, 2.1), aspect="equal")
    ax.axis("off")


def draw_unrolled(ax, layout: ElectrodeLayout, array: VirtualArray | None = None):
    """The skin cut along the dorsal midline and laid flat: angle across, elbow (z 0) at the top, wrist (1) below.
    Bars: how far each group may sit from there (angle_sd, z_sd); an arrow across the arm: any rotation. With
    `array`: the virtual electrodes as squares, dark where real electrodes cover them, hollow where none do."""
    ax.axhspan(0, 1, color="#f3e6dc", zorder=0)
    ax.axhspan(-0.6, 0, color="#f7f1ec", zorder=0)
    ax.text(178, 0.97, "wrist", ha="right", va="bottom", fontsize=7, color="#6b6a66")
    ax.text(-178, -0.01, "elbow", ha="left", va="bottom", fontsize=7, color="#6b6a66")
    ax.text(178, -0.55, "upper arm", ha="right", va="top", fontsize=7, color="#6b6a66")
    if array is not None:
        virtual = array.positions().reshape(-1, 2)
        fill = array.covered(array.inputs(layout)).ravel()
        x = around(virtual[:, 0])
        x = np.r_[x, 180.0 * (x == -180).any() * np.ones((x == -180).sum())]  # the dorsal column on both edges
        z = np.r_[virtual[:, 1], virtual[x[: len(virtual)] == -180, 1]]
        fill = np.r_[fill, fill[x[: len(virtual)] == -180]]
        ax.scatter(x[fill], z[fill], marker="s", s=110, color="#a8a7a2", alpha=0.35, zorder=1, lw=0)
        ax.scatter(x[~fill], z[~fill], marker="s", s=110, facecolor="none", edgecolor="#a8a7a2", zorder=1, lw=0.8)
    channel = 0
    for i, g in enumerate(layout.groups):
        angles, zs = around(np.array(g.angles)), np.array(g.zs)
        color = COLORS[i % len(COLORS)]
        if np.isfinite(angles).any():
            many = len(g) > 16
            ax.plot(angles, zs, "s" if many else "o", color=color, ms=2.5 if many else 6, mec="white", mew=0.4)
            if not many:
                for c in range(len(g)):
                    ax.text(angles[c] + 9, zs[c] + 0.035, channel + c + 1, fontsize=6, ha="center")
            middle = (np.nanmean(angles), np.nanmean(zs))
            if g.angle_sd >= 180:
                below = np.nanmax(zs) + 0.09
                ax.annotate("", (178, below), (-178, below), arrowprops=dict(arrowstyle="<->", color=color, lw=0.8))
                ax.text(0, below + 0.01, "any rotation", ha="center", va="top", fontsize=6, color=color)
            elif g.angle_sd or g.z_sd:
                ax.errorbar(*middle, xerr=g.angle_sd, yerr=g.z_sd, color=color, alpha=0.5, capsize=2, lw=1)
        else:
            ax.text(0, 0.8, f"{len(g)} electrodes\nposition unknown", ha="center", va="center", fontsize=8, color=color)
        channel += len(g)
    ax.set(xlim=(-185, 185), ylim=(1.05, -0.6), xticks=[-180, -90, 0, 90, 180])
    ax.set_xticklabels(["dorsal", "ulnar", "palmar", "radial", "dorsal"], fontsize=7)


def pretty(name: str) -> str:
    return name.replace("ninapro_", "NinaPro ").replace("_", "-").upper()


def layouts(sessions: dict[str, list[dict]]) -> list[tuple[str, ElectrodeLayout]]:
    """Datasets with the same electrode layout share one entry: [("NINAPRO DB2, NINAPRO DB7", layout)]."""
    by_layout = defaultdict(list)
    for name, runs in sessions.items():
        by_layout[runs[0]["positions"].round(2).tobytes()].append((name, runs[0]["layout"]))
    return [(", ".join(pretty(n) for n, _ in entries), entries[0][1]) for entries in by_layout.values()]


def plot_layouts(sessions: dict[str, list[dict]], array: VirtualArray, path: Path):
    entries = layouts(sessions)
    fig, axes = plt.subplots(2, len(entries), figsize=(3.4 * len(entries), 8.2), height_ratios=[1, 1.3], squeeze=False)
    for column, (names, layout) in enumerate(entries):
        draw_section(axes[0, column], layout)
        draw_unrolled(axes[1, column], layout, array)
        axes[0, column].set_title(f"{names}\n{len(layout)} channels, {layout.kind}", fontsize=9)
    axes[1, 0].set_ylabel("z: 0 elbow, 1 wrist, < 0 upper arm")
    fig.tight_layout(rect=(0.03, 0.03, 1, 1))
    fig.text(0.01, 0.78, "cross-section, seen from the hand", rotation=90, va="center", fontsize=9, weight="bold")
    fig.text(
        0.01, 0.3, "skin unrolled (cut at the dorsal midline)", rotation=90, va="center", fontsize=9, weight="bold"
    )
    fig.text(
        0.5,
        0.01,
        "grey squares: virtual electrodes (filled: covered by real ones, hollow: no real electrode near, output 0)",
        ha="center",
        fontsize=8,
        color="#6b6a66",
    )
    fig.savefig(path, dpi=130)


def plot_report(sessions, array, decay, rebuilt, r2, path: Path):
    color = {name: COLORS[i % len(COLORS)] for i, name in enumerate(sessions)}
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    (ax_decay, ax_cover, ax_rebuilt), (ax_gain, ax_spectrum, ax_r2) = axes

    d = np.linspace(0, 12, 100)
    for direction, marker, width in (("around", "o-", array.width_around_cm), ("along", "s--", array.width_along_cm)):
        for name, pairs in decay.items():
            dd, cc = pairs[direction]
            which = np.digitize(dd, DISTANCE_BINS)
            bins = [b for b in np.unique(which) if (which == b).sum() > 2]
            if len(bins) > 0:
                points = np.array([(dd[which == b].mean(), np.median(cc[which == b])) for b in bins])
                label = pretty(name) if direction == "around" else None
                ax_decay.plot(*points.T, marker, ms=4, lw=1, color=color[name], label=label)
        ax_decay.plot(d, np.exp(-0.5 * (d / width) ** 2), color=INK, lw=2, ls=marker[1:], alpha=0.5)
    ax_decay.plot([], [], "k-", lw=2, alpha=0.5, label=f"kernel around (sd {array.width_around_cm} cm)")
    ax_decay.plot([], [], "k--", lw=2, alpha=0.5, label=f"kernel along (sd {array.width_along_cm} cm)")
    ax_decay.axhline(0, color="#a8a7a2", lw=0.5)
    ax_decay.set(xlabel="distance between two electrodes of a band (cm)", ylabel="envelope correlation (median)")
    ax_decay.set_title("a  How far does a channel's activity reach?\n(circles: around the arm, squares: along it)")
    ax_decay.legend(fontsize=7, ncol=2)

    names = list(sessions)
    for row, name in enumerate(names):
        left = 0.0
        for k, (source, share) in enumerate(sources(array, sessions[name][0]).items()):
            ax_cover.barh(row, share, left=left, color=COLORS[k % len(COLORS)], edgecolor="white")
            if share > 0.06:
                ax_cover.text(left + share / 2, row, f"{source}\n{share:.0%}", ha="center", va="center", fontsize=7)
            left += share
    ax_cover.set_yticks(range(len(names)), [pretty(n) for n in names], fontsize=8)
    ax_cover.set(xlim=(0, 1), xlabel="share of the virtual ring's weight")
    ax_cover.invert_yaxis()
    heights = ", ".join(f"{z:g}" for z in array.heights)
    ax_cover.set_title(f"b  Which real electrodes feed the virtual ring (z {heights})\n(mean share of the weight)")

    names = list(rebuilt)
    x = np.arange(len(names))
    ax_rebuilt.bar(x - 0.27, [rebuilt[n][0] for n in names], 0.27, color="#a8a7a2", label="raw signal")
    ax_rebuilt.bar(x, [rebuilt[n][1] for n in names], 0.27, color=COLORS[0], label="envelope (log RMS)")
    across = [rebuilt[n][3] for n in names]
    ax_rebuilt.bar(x + 0.27, np.nan_to_num(across), 0.27, color=COLORS[1], label="envelope, from the other ring only")
    ax_rebuilt.set_xticks(x, [pretty(n) for n in names], rotation=20, fontsize=8)
    ax_rebuilt.set(ylim=(0, 1), ylabel="correlation with the real channel (median)")
    ax_rebuilt.set_title("c  A real channel rebuilt from its neighbours\n(what a virtual electrode at its place gives)")
    ax_rebuilt.legend(fontsize=7)

    names = list(sessions)
    level = [
        np.median(1 / sessions[n][0]["gains"][sessions[n][0]["covered"]])
        for n in names
        if sessions[n][0]["covered"].any()
    ]
    shown = [n for n in names if sessions[n][0]["covered"].any()]
    kinds = [sessions[n][0]["layout"].kind for n in shown]
    ax_gain.bar(range(len(shown)), level, color=[COLORS[0] if k == "bipolar" else COLORS[1] for k in kinds])
    ax_gain.axhline(1, color=INK, lw=0.8, ls="--")
    labels = [f"{pretty(n).replace('NINAPRO ', '')}\n{k}" for n, k in zip(shown, kinds, strict=True)]
    ax_gain.set_xticks(range(len(shown)), labels, fontsize=7)
    ax_gain.set(ylabel="active level of a virtual channel / of a real one")
    ax_gain.set_title(
        "d  Why the gains: a mean of independent signals is weaker\n(before the gains; after them all are 1)"
    )

    for name, runs in sessions.items():
        if runs[0]["covered"].any():
            f, share = spectrum(runs)
            ax_spectrum.semilogy(f, share, color=color[name], label=pretty(name))
    ax_spectrum.set(xlabel="frequency (Hz)", ylabel="share of the power", xlim=(0, 100))
    ax_spectrum.set_title("e  Do the virtual signals of all datasets look alike?\n(spectrum of the virtual channels)")
    ax_spectrum.legend(fontsize=7)

    names = list(r2)
    x = np.arange(len(names))
    ax_r2.bar(x - 0.27, [r2[n][0] for n in names], 0.27, color="#d8d7d2", label="all real channels")
    ax_r2.bar(x, [r2[n][1] for n in names], 0.27, color="#a8a7a2", label="nearest real ring of 8")
    ax_r2.bar(x + 0.27, [r2[n][2] for n in names], 0.27, color=COLORS[0], label="virtual ring")
    for i, n in enumerate(names):
        ax_r2.text(i + 0.27, max(r2[n][2], 0) + 0.01, f"{r2[n][2] - r2[n][1]:+.2f}", ha="center", fontsize=7)
    ax_r2.set_xticks(x, [pretty(n) for n in names], rotation=20, fontsize=8)
    ax_r2.set(ylabel=f"pose R² (ridge on log RMS, 5-fold, pose at t + {LAG_S} s)")
    ax_r2.set_title("f  Is the hand information still there?\n(same decoder on the real and the virtual channels)")
    ax_r2.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    parser = argparse.ArgumentParser(description="Checks of the virtual electrode array and of its signals.")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    parser.add_argument("--per-dataset", type=int, default=2, help="sessions per dataset")
    parser.add_argument("--max-s", type=float, default=300, help="seconds read per session")
    args = parser.parse_args()

    array = VirtualArray.load()
    sessions = load_all(args.data, array, args.per_dataset, args.max_s, args.interim)
    checks = array_checks(array, sessions)

    decay = {name: spatial_decay(runs) for name, runs in sessions.items()}
    for name, pairs in decay.items():
        d, c = pairs["around"]
        if len(d) > 10:
            rho = spearmanr(d, c).statistic
            checks.append((f"{name}: correlation falls with distance", rho < -0.2, f"Spearman {rho:.2f} (around)"))

    known = {name: runs for name, runs in sessions.items() if runs[0]["covered"].any()}
    rebuilt = {
        name: leave_one_out(array, runs) + leave_one_out(array, runs, other_height=True) for name, runs in known.items()
    }
    for name, (raw, env, _, across) in rebuilt.items():
        detail = f"envelope r {env:.2f}, raw signal r {raw:.2f}"
        checks.append((f"{name}: channels rebuilt from their neighbours", env >= 0.5, detail))
        if np.isfinite(across):
            detail = f"envelope r {across:.2f}: a ring 2.5 cm from the one it stands in for"
            checks.append((f"{name}: a ring rebuilt from the other ring", across >= 0.5, detail))

    r2 = {name: pose_r2(runs, array.heights) for name, runs in known.items() if "pose" in runs[0]}
    for name, (real, ring, virtual) in r2.items():
        detail = f"R2 all real {real:.2f}, nearest real ring {ring:.2f}, virtual {virtual:.2f}"
        checks.append(
            (f"{name}: virtual ring keeps the pose information of a real one", virtual >= ring - 0.05, detail)
        )

    args.out.mkdir(parents=True, exist_ok=True)
    plot_layouts(sessions, array, args.out / "virtual_array_layouts.png")
    plot_report(sessions, array, decay, rebuilt, r2, args.out / "virtual_array_report.png")

    width = max(len(c[0]) for c in checks)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:{width}}  {detail}")
    print(f"\nwrote virtual_array_report.png and virtual_array_layouts.png in {args.out}")


if __name__ == "__main__":
    main()
