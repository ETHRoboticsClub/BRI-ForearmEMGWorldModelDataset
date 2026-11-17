import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from femg.calib_baseline import CalibBaseline
from femg.datasets import DATASETS
from femg.electrodes import ElectrodeLayout
from femg.normalize import Normalizer
from femg.virtual_array import VirtualArray

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100", "#8a5cd1"]
INK = "#28282a"


def session_folders(data: Path) -> list[Path]:
    return [m.parent for name in DATASETS for m in sorted((data / name).glob("*/*/meta.json"))]


def signatures(calib: CalibBaseline, folders: list[Path]) -> tuple[dict, dict]:
    out, info = {}, {}
    for folder in folders:
        meta = json.loads((folder / "meta.json").read_text())
        signature = calib.session_signature(folder)
        if signature:
            out[meta["key"]] = signature
            info[meta["key"]] = {"dataset": meta["dataset"], "subject": meta["subject"], "folder": folder}
    return out, info


def aligned(calib, sigs, sessions) -> dict[str, dict[str, np.ndarray]]:
    rows = sessions.set_index("key")
    n = len(next(iter(next(iter(sigs.values())).values())))
    return {
        k: {
            a: calib.transform(x, round(rows.loc[k, "rotation_deg"] * n / 360), bool(rows.loc[k, "mirror"]))
            for a, x in s.items()
        }
        for k, s in sigs.items()
    }


def templates_without(calib, aligned_sigs, key) -> dict[str, np.ndarray]:
    others = {k: s for k, s in aligned_sigs.items() if k != key}
    out = {}
    for s in others.values():
        for a, x in s.items():
            out.setdefault(a, []).append(x)
    return {a: np.mean(xs, axis=0) for a, xs in out.items()}


# Checks


def check_transform(calib: CalibBaseline, folder: Path, interim: Path, max_s: float = 300) -> tuple[bool, str]:
    """The signature of a band rebuilt by VirtualArray(rotation, mirror) from the real EMG (its interim session) =
    transform of the signature: the placements the calibration searches are the ones the virtual array can apply."""
    meta = json.loads((folder / "meta.json").read_text())
    array = VirtualArray.load()
    real = np.load(interim / meta["key"] / "emg.npy", mmap_mode="r")[: int(max_s * meta["fs"])]
    emg = Normalizer.emg(real, meta["emg_stats"])
    labels, stimulus = np.load(folder / "labels.npy"), np.load(folder / "stimulus.npy")
    inputs = array.inputs(ElectrodeLayout.from_meta(meta))
    base = calib.signature(array(emg, inputs).reshape(len(emg), -1), labels, meta, stimulus)
    worst = 0.0
    for steps, mirror in [(1, False), (3, True), (6, True)]:
        turned = array(emg, inputs, rotation=steps * 360 / array.n_angles, mirror=mirror).reshape(len(emg), -1)
        moved = calib.signature(turned, labels, meta, stimulus)
        worst = max(worst, max(np.abs(moved[a] - calib.transform(base[a], steps, mirror)).max() for a in base))
    return worst < 1e-3, f"{meta['key']}: max difference {worst:.1e} over 3 placements"


def check_recovery(calib, aligned_sigs, rng) -> tuple[float, int]:
    """Each aligned session moved to a random placement, matched to the others' templates: share recovered."""
    n = len(next(iter(next(iter(aligned_sigs.values())).values())))
    hits = 0
    for key, s in aligned_sigs.items():
        p = (int(rng.integers(n)), bool(rng.integers(2)) and calib.search_mirror)
        moved = {a: calib.transform(x, *p) for a, x in s.items()}
        m = calib.match(moved, templates_without(calib, aligned_sigs, key))
        hits += calib.compose(p, (m.steps, m.mirror), n) == (0, False)
    return hits / len(aligned_sigs), len(aligned_sigs)


def leave_one_out(calib, sigs, aligned_sigs, sessions) -> dict[str, tuple[bool, float, float]]:
    """Each session matched to the templates of the others: same placement as in the group fit? score, margin."""
    rows = sessions.set_index("key")
    n = len(next(iter(next(iter(sigs.values())).values())))
    out = {}
    for key in sigs:
        m = calib.match(sigs[key], templates_without(calib, aligned_sigs, key))
        fitted = (round(rows.loc[key, "rotation_deg"] * n / 360), bool(rows.loc[key, "mirror"]))
        out[key] = ((m.steps, m.mirror) == fitted, m.score, m.margin)
    return out


# Figure


DATASET_COLORS = {
    "emg_fk": "#2a78d6",
    "ninapro_db2": "#eb6834",
    "ninapro_db7": "#1baf7a",
    "ninapro_db8": "#8a5cd1",
    "great": "#eda100",
}
SIDES = {0: "palmar", 90: "radial", 180: "dorsal", 270: "ulnar", 360: "palmar"}


def closed(x: np.ndarray) -> np.ndarray:
    """The ring's values with the first repeated at the end: 0..360 degrees."""
    return np.r_[x, x[:1]]


def plot(calib, baseline, aligned_sigs, info, path: Path):
    """One panel per action: its template (mean of the aligned sessions, ± 1 sd), the mean of each dataset, the peak
    and the anatomical anchor."""
    actions = list(baseline.templates)
    n = len(baseline.templates[actions[0]])
    x = np.arange(n + 1) * 360 / n
    columns = 3
    rows = -(-len(actions) // columns)
    style = {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#a8a7a2",
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "xtick.color": "#5f5e5a",
        "ytick.color": "#5f5e5a",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.3 * rows), sharex=True, sharey=True)
        axes = axes.ravel()
        limit = max(np.abs(baseline.templates[a]).max() + baseline.spread[a].max() for a in actions) * 1.1
        for ax, action in zip(axes, actions, strict=False):
            template, spread = closed(baseline.templates[action]), closed(baseline.spread[action])
            ax.axhline(0, color="#d8d7d2", lw=0.8, zorder=0)
            ax.fill_between(x, template - spread, template + spread, color="#28282a", alpha=0.08, lw=0)
            for dataset, color in DATASET_COLORS.items():
                rows_of = [s[action] for k, s in aligned_sigs.items() if action in s and info[k]["dataset"] == dataset]
                if rows_of:
                    ax.plot(x, closed(np.mean(rows_of, axis=0)), color=color, lw=1, alpha=0.85)
            ax.plot(x, template, color=INK, lw=2.2)
            peak = calib.peak_deg(baseline.templates[action])
            for at in {peak, peak + 360 if peak < 15 else peak - 360 if peak > 345 else peak}:  # 0° is also 360°
                ax.plot(at, limit * 0.93, "v", color=INK, ms=6, clip_on=False)
            if "anchor_deg" in calib.actions[action]:
                anchor = calib.actions[action]["anchor_deg"] % 360
                for at in {anchor, 360} if anchor == 0 else {anchor}:
                    ax.axvline(at, color=INK, lw=0.9, ls=(0, (2, 2)))
            name = action.replace("_", " ").capitalize()
            ax.set_title(name, loc="left")
            ax.text(
                1,
                1.02,
                f"{baseline.counts[action]} sessions · peak {peak:.0f}°",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#5f5e5a",
            )
            ax.grid(axis="y", color="#ecebe7", lw=0.6)
            ax.set_xlim(0, 360)
            ax.set_ylim(-limit, limit)
            ax.set_xticks(list(SIDES), [f"{side}\n{deg}°" for deg, side in SIDES.items()])
            ax.tick_params(labelbottom=True)
        for ax in axes[len(actions) :]:
            ax.axis("off")
        for ax in axes[::columns]:
            ax.set_ylabel("log RMS vs rest\n(minus the ring mean)")

        handles = [plt.Line2D([], [], color=INK, lw=2.2, label="template (all sessions)")]
        handles.append(plt.Rectangle((0, 0), 1, 1, color=INK, alpha=0.08, label="± 1 sd across sessions"))
        handles += [plt.Line2D([], [], color=c, lw=1, label=d) for d, c in DATASET_COLORS.items()]
        handles.append(plt.Line2D([], [], color=INK, marker="v", ls="", label="peak"))
        handles.append(plt.Line2D([], [], color=INK, lw=0.8, ls=(0, (2, 2)), label="anatomical anchor"))
        spare = axes[len(actions)] if len(actions) < len(axes) else None
        if spare is not None:
            spare.legend(handles=handles, loc="center", frameon=False, fontsize=8.5)
        else:
            fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5)
        fig.suptitle(
            "Calibration baseline: where each action shows up around the forearm (virtual ring, aligned sessions)",
            x=0.01,
            ha="left",
            fontsize=11,
            fontweight="bold",
        )
        fig.text(
            0.01,
            0.005,
            "Angles: 0° palmar, 180° dorsal (anchored on wrist flexion); radial / ulnar consistent across "
            "sessions, not fixed anatomically.",
            fontsize=7.5,
            color="#5f5e5a",
        )
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        fig.savefig(path, dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Checks of the calibration baseline (femg.calib_baseline).")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    calib = CalibBaseline.load()
    folders = session_folders(args.data)
    sigs, info = signatures(calib, folders)
    baseline = calib.fit(sigs, {k: {"dataset": v["dataset"], "subject": v["subject"]} for k, v in info.items()})
    aligned_sigs = aligned(calib, sigs, baseline.sessions)

    checks = []
    with_wrist = next(info[k]["folder"] for k in sigs if "wrist_flexion" in sigs[k])
    ok, detail = check_transform(calib, with_wrist, args.interim)
    checks.append(("transform = VirtualArray(rotation, mirror) on the real EMG", ok, detail))

    share, count = check_recovery(calib, aligned_sigs, np.random.default_rng(0))
    checks.append(("a session moved to a random placement is recovered", share >= 0.8, f"{share:.0%} of {count}"))

    loo = leave_one_out(calib, sigs, aligned_sigs, baseline.sessions)
    agree = np.mean([v[0] for v in loo.values()])
    margins = np.median([v[2] for v in loo.values()])
    detail = f"{agree:.0%} agree, median score {np.median([v[1] for v in loo.values()]):.2f}, margin {margins:.2f}"
    checks.append(("each session placed the same against the others' templates", agree >= 0.8, detail))

    for subject, rows in baseline.sessions.groupby("subject"):
        if len(rows) > 1:
            placements = sorted({(r, m) for r, m in zip(rows["rotation_deg"], rows["mirror"], strict=True)})
            detail = ", ".join(f"{r:g}°{' mirrored' if m else ''}" for r, m in placements)
            # not a pass / fail: a band put on again between sessions (GREAT: every day) may sit elsewhere
            checks.append((f"{subject}: placements of its sessions", None, detail))

    for a, spec in calib.actions.items():
        if "anchor_deg" in spec and a in baseline.templates:
            peak = calib.peak_deg(baseline.templates[a])
            error = abs((peak - spec["anchor_deg"] + 180) % 360 - 180)
            checks.append((f"{a} peaks near {spec['anchor_deg']}°", error <= 45, f"peak {peak:.0f}°"))

    args.out.mkdir(parents=True, exist_ok=True)
    plot(calib, baseline, aligned_sigs, info, args.out / "calib_baseline.png")
    width = max(len(c[0]) for c in checks)
    for name, ok, detail in checks:
        status = "INFO" if ok is None else "PASS" if ok else "FAIL"
        print(f"{status}  {name:{width}}  {detail}")
    print(
        f"\n{len(sigs)} of {len(folders)} sessions with rest and another action",
        f"wrote {args.out / 'calib_baseline.png'}",
    )


if __name__ == "__main__":
    main()
