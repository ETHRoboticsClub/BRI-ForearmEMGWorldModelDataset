import argparse
import json
import logging
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from femg.calib_baseline import CalibBaseline
from femg.electrodes import ElectrodeLayout
from femg.hand.joints import JOINT_INDEX
from femg.hand.mapping import HandMap
from femg.normalize import Normalizer, as_array
from femg.processor import Processor
from femg.signal_utils import causal_filter, resample, runs
from myogen import Electrode, Electrodes, JointAdapter, SimParams, Simulator, Subject
from myogen.inputs.motion import REST_POSTURE
from myogen.tools.calibration import statistics

CONFIG = Path(__file__).parents[2] / "src/femg/config/synthetic.yaml"
DATASET_COLORS = {"emg_fk": "#2a78d6", "ninapro_db2": "#eb6834", "ninapro_db7": "#1baf7a", "ninapro_db8": "#8a5cd1"}
REAL, SIM = "#5f5e5a", "#d1495b"
N_ANGLES = 8  # electrodes per ring
STATS = {  # myogen statistic: label
    "autocorr_50ms": "autocorr 50 ms",
    "autocorr_200ms": "autocorr 200 ms",
    "neighbour_correlation": "neighbour r",
    "rest_move_ratio": "rest / move",
}


@dataclass(frozen=True)
class SyntheticCheck:
    """Real free-movement kinematics replayed through myogen, at the dataset's own ring electrodes; the simulated EMG
    through the interim filters; then its RMS against the real one."""

    free_movement: dict[str, list[str]]
    conditions: list[str]
    hand: str
    subject: dict
    params: dict
    random_subject: bool
    ied_mm: float
    monopolar_reference: str
    rest_anchor: bool
    reps_per_cue: int
    pad_s: float
    chunk_s: float
    chunks: int
    skip_s: float

    @classmethod
    def load(cls, path: Path = CONFIG) -> "SyntheticCheck":
        return cls(**yaml.safe_load(path.read_text()))

    def settings(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "free_movement"}

    def sessions(self, interim: Path, per_dataset: int, datasets: list[str] | None = None) -> list[Path]:
        out = []
        for dataset, names in self.free_movement.items():
            if datasets and dataset not in datasets:
                continue
            folders = [
                m.parent
                for m in sorted((interim / dataset).glob("*/*/meta.json"))
                if m.parent.name in names and json.loads(m.read_text())["condition"] in self.conditions
            ]
            subjects = sorted({f.parent.name for f in folders})[:per_dataset]
            out += [f for f in folders if f.parent.name in subjects]
        return out

    # The motion

    def windows(self, stimulus: np.ndarray, label_fs: float) -> list[tuple[int, int]]:
        """[start, end) label frames of each simulated motion: the first repetitions of every cue with rest around
        them, or evenly spaced windows without cues."""
        n, pad = len(stimulus), round(self.pad_s * label_fs)
        if (stimulus > 0).any():
            out = []
            for cue in np.unique(stimulus[stimulus > 0]):
                for start, end in runs(stimulus == cue)[: self.reps_per_cue]:
                    out.append((max(0, start - pad), min(n, end + pad)))
            return out
        length = round(self.chunk_s * label_fs)
        starts = np.linspace(0, n - length, self.chunks).astype(int)
        return [(int(s), int(s) + length) for s in starts]

    def model_angles(self, kin: np.ndarray, meta: dict, stimulus: np.ndarray) -> dict[str, np.ndarray]:
        """Interim joints -> the model's (rad): each joint in 0-1 by its anchors, then onto the model's range
        (the hand mapping, as the explorer poses the hand). Works alike on calibrated and raw gloves.

        rest_anchor (sessions with a rest cue): the session's median rest posture is the model's relaxed hand, the
        0-1 span keeps its size. Without it a session of mostly extensions (NinaPro exercise B) puts rest at ~0.7,
        a half-closed hand the model holds actively."""
        unit = np.clip(Normalizer.kin(kin, meta["kin_stats"]), 0, 1)
        stats = meta["kin_stats"]
        low, high, mean = (as_array(stats[k]) for k in ("low", "high", "mean"))
        centers = (mean - low) / (high - low)
        rest = stimulus[: len(unit)] == 0
        anchored = self.rest_anchor and rest.any() and (stimulus > 0).any()
        if anchored:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # unmeasured joints
                centers = np.nanmedian(unit[rest], axis=0)
        done = {}
        for rule in HandMap(self.hand).rules:
            x = unit[:, JOINT_INDEX[rule.label]] if rule.label else None
            if x is not None and np.isfinite(x).any():
                x = fill_nan(x)
                c = centers[JOINT_INDEX[rule.label]]
                if rule.neutral_mean and 0 < c < 1:
                    done[rule.joint] = np.where(x < c, rule.at_0 * (c - x) / c, rule.at_1 * (x - c) / (1 - c))
                elif anchored and np.isfinite(c):
                    done[rule.joint] = REST_POSTURE[rule.joint] + (x - c) * (rule.at_1 - rule.at_0)
                else:
                    done[rule.joint] = rule.at_0 + x * (rule.at_1 - rule.at_0)
            elif rule.source:
                done[rule.joint] = rule.ratio * done[rule.source]
            else:
                done[rule.joint] = np.full(len(kin), rule.fallback)
        return done

    def electrodes(self, meta: dict) -> tuple[Electrodes, np.ndarray]:
        """myogen electrodes at the ring channels of the session, and those channels, ring by ring."""
        positions = ElectrodeLayout.from_meta(meta).positions(("ring",))
        channels = np.flatnonzero(np.isfinite(positions[:, 0]))
        channels = channels[np.lexsort((positions[channels, 0], positions[channels, 1]))]
        if len(channels) % N_ANGLES:
            raise ValueError(f"{meta['key']}: {len(channels)} ring channels, not rings of {N_ANGLES}")
        derivation = "bipolar" if positions[channels, 2].all() else "monopolar"
        length_cm = 25.8  # myogen's forearm, elbow to wrist
        electrodes = [
            Electrode(
                angle_deg=(a - 90 + 180) % 360 - 180,  # femg 0 palmar, 90 radial; myogen 0 radial, -90 palmar
                position_cm=float(z) * length_cm,
                ied_mm=self.ied_mm,
                name=f"ch{c}",
            )
            for c, (a, z) in zip(channels, positions[channels, :2], strict=True)
        ]
        device = {"device": "raw", "fs": 2000.0, "mains_hz": meta["mains_hz"]}
        return Electrodes(electrodes, derivation=derivation, reference=self.monopolar_reference, **device), channels

    def simulate(self, folder: Path, cache: Path, workers: int | None) -> dict:
        """The motions of a session through myogen -> the EMG as interim (V, fs, filtered), cached."""
        meta = json.loads((folder / "meta.json").read_text())
        file = cache / f"{meta['key']}.npz"
        if file.exists():
            saved = np.load(file, allow_pickle=False)
            if json.loads(str(saved["settings"])) == self.settings():
                return {k: saved[k] for k in ("emg", "windows", "channels")}
        kin, stimulus = np.load(folder / "kin.npy"), np.load(folder / "stimulus.npy")
        windows = self.windows(stimulus, meta["label_fs"])
        angles = self.model_angles(kin, meta, stimulus)
        electrodes, channels = self.electrodes(meta)
        seed = zlib.crc32(meta["key"].encode())
        subject = Subject.random(seed) if self.random_subject else Subject()
        subject = Subject.from_dict({**subject.to_dict(), **self.subject})
        motions = [JointAdapter()({j: q[a:b] for j, q in angles.items()}, meta["label_fs"]) for a, b in windows]
        params = SimParams().replace(**self.params)
        recordings = Simulator(electrodes, subject, params).run(motions, workers=workers, progress=True)
        per_label = meta["fs"] // meta["label_fs"]
        emg = []
        for (a, b), rec in zip(windows, recordings, strict=True):
            x = resample(rec.emg.astype(np.float64) * 1e-3, rec.fs, meta["fs"])
            x = causal_filter(x, meta["fs"], meta["processor"]["highpass"], meta["mains_hz"])
            n = (b - a) * per_label
            emg.append(np.pad(x[:n], ((0, max(0, n - len(x))), (0, 0)), mode="edge"))
        out = {"emg": np.concatenate(emg).astype(np.float32), "windows": np.array(windows), "channels": channels}
        file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(file, **out, settings=json.dumps(self.settings()))
        return out

    # The comparison

    def kept(self, windows: np.ndarray, label_fs: float) -> list[np.ndarray]:
        """Label frames of each window, its first skip_s left out."""
        skip = round(self.skip_s * label_fs)
        return [np.arange(a + skip, b) for a, b in windows]

    def compare(self, folder: Path, sim: dict, normalizer: Normalizer) -> dict:
        meta = json.loads((folder / "meta.json").read_text())
        per_label, label_fs = meta["fs"] // meta["label_fs"], meta["label_fs"]
        channels, windows = sim["channels"], sim["windows"]
        real_emg = np.load(folder / "emg.npy", mmap_mode="r")
        real = np.concatenate([real_emg[a * per_label : b * per_label, channels] for a, b in windows]).astype(
            np.float64
        )
        if meta["electrode_type"] == "monopolar":
            real = real - real.mean(axis=1, keepdims=True)  # as the virtual array does
        offsets = np.cumsum([0] + [b - a for a, b in windows])
        kept = self.kept(windows, label_fs)
        frames = np.concatenate([o + k - a for o, k, (a, _) in zip(offsets[:-1], kept, windows, strict=True)])
        samples = (frames[:, None] * per_label + np.arange(per_label)).ravel()

        env = {"real": normalizer.envelope(real)[frames], "sim": normalizer.envelope(sim["emg"])[frames]}
        log_env = {k: np.log(v + 1e-9) for k, v in env.items()}
        stimulus = np.concatenate([np.load(folder / "stimulus.npy")[a:b] for a, b in windows])[frames]
        labels = Normalizer.kin(
            np.concatenate([np.load(folder / "kin.npy")[a:b] for a, b in windows]), meta["kin_stats"]
        )
        sigs = {k: signature(v, stimulus, labels[frames], label_fs) for k, v in log_env.items()}
        actions = sorted(set(sigs["real"]) & set(sigs["sim"]))
        step, mirror, r_sig = best_placement(sigs["sim"], sigs["real"], actions)
        sim_env = place(log_env["sim"], step, mirror)
        rest = None
        if (stimulus == 0).any() and (stimulus > 0).any():
            rest = np.repeat(stimulus == 0, per_label)
        stats = {
            k: statistics(x[samples], meta["fs"], "raw", rest)
            for k, x in (("real", real), ("sim", place(sim["emg"], step, mirror).astype(np.float64)))
        }
        return {
            "key": meta["key"],
            "dataset": meta["dataset"],
            "r_global": pearson(log_env["real"].mean(1), log_env["sim"].mean(1)),
            "r_channel": float(
                np.median([pearson(log_env["real"][:, c], sim_env[:, c]) for c in range(len(channels))])
            ),
            "r_signature": r_sig,
            "placement": (step, mirror),
            "amplitude_uv": {k: float(np.median(np.percentile(v, 95, axis=0))) * 1e6 for k, v in env.items()},
            "range": {
                k: float(np.median(np.log10(np.percentile(v, 95, 0) / np.percentile(v, 10, 0)))) for k, v in env.items()
            },
            "stats": {k: {s: stats[k][s] for s in STATS} for k in stats},
            "signatures": sigs,
            "actions": actions,
            "trace": {"real": env["real"], "sim": place(env["sim"], step, mirror), "lengths": [len(k) for k in kept]},
        }


def fill_nan(x: np.ndarray) -> np.ndarray:
    good = np.isfinite(x)
    return np.interp(np.arange(len(x)), np.flatnonzero(good), x[good])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def place(x: np.ndarray, step: int, mirror: bool) -> np.ndarray:
    """(..., R * 8) channels ring by ring -> each ring turned as CalibBaseline.transform."""
    shape = x.shape
    rings = x.reshape(*shape[:-1], -1, N_ANGLES)
    return CalibBaseline.transform(rings, step, mirror).reshape(shape)


def signature(log_env: np.ndarray, stimulus: np.ndarray, labels: np.ndarray, label_fs: float) -> dict:
    """Action -> (C,): mean log RMS minus rest, minus the mean over channels. Actions: the cues, else the held hand
    states of the calibration baseline."""
    if (stimulus > 0).any():
        frames = {"rest": stimulus == 0} | {f"cue {c}": stimulus == c for c in np.unique(stimulus[stimulus > 0])}
    else:
        frames = CalibBaseline.load().frames(labels, label_fs)
    if "rest" not in frames:
        return {}
    rest = log_env[frames["rest"]].mean(0)
    out = {}
    for name, mask in frames.items():
        if name != "rest" and mask.any():
            pattern = log_env[mask].mean(0) - rest
            out[name] = pattern - pattern.mean()
    return out


def best_placement(moved: dict, fixed: dict, actions: list[str]) -> tuple[int, bool, float]:
    """The placement of `moved` whose signatures correlate best with `fixed` over the actions."""
    if not actions:
        return 0, False, float("nan")
    b = np.concatenate([fixed[a] for a in actions])
    scores = {
        (k, m): pearson(np.concatenate([place(moved[a], k, m) for a in actions]), b)
        for m in (False, True)
        for k in range(N_ANGLES)
    }
    (k, m), r = max(scores.items(), key=lambda item: item[1])
    return k, m, r


def reference(results: list[dict]) -> dict[str, list[float]]:
    """Real against real: each session's signature against the sessions of the other subjects of its dataset (best
    placement)."""
    out = {}
    for res in results:
        subject = res["key"].split("/")[1]
        others = [o for o in results if o["dataset"] == res["dataset"] and o["key"].split("/")[1] != subject]
        scores = []
        for other in others:
            actions = sorted(set(res["signatures"]["real"]) & set(other["signatures"]["real"]))
            if actions:
                scores.append(best_placement(other["signatures"]["real"], res["signatures"]["real"], actions)[2])
        out[res["key"]] = scores
    return out


def median_of(runs_: list[dict], value) -> float:
    return float(np.nanmedian([value(r) for r in runs_]))


def report(results: list[dict], ref: dict) -> None:
    head = f"{'session':28} {'r global':>8} {'r chan':>7} {'r sig':>6} {'ref sig':>7} {'µV real':>8} {'µV sim':>7}"
    print(head + "".join(f" {v:>16}" for v in STATS.values()))
    for res in results:
        row = f"{res['key']:28} {res['r_global']:8.2f} {res['r_channel']:7.2f} {res['r_signature']:6.2f}"
        row += f" {np.mean(ref[res['key']]) if ref[res['key']] else np.nan:7.2f}"
        row += f" {res['amplitude_uv']['real']:8.1f} {res['amplitude_uv']['sim']:7.1f}"
        for s in STATS:
            a, b = res["stats"]["real"][s], res["stats"]["sim"][s]
            row += f" {'-':>16}" if a is None else f" {a:7.2f} / {b:6.2f}"
        print(row)
    print("\nmedian per dataset")
    for dataset in DATASET_COLORS:
        runs_ = [r for r in results if r["dataset"] == dataset]
        if not runs_:
            continue
        refs = [np.mean(ref[r["key"]]) for r in runs_ if ref[r["key"]]]
        values = [
            median_of(runs_, lambda r: r["r_global"]),
            median_of(runs_, lambda r: r["r_channel"]),
            median_of(runs_, lambda r: r["r_signature"]),
            np.median(refs) if refs else np.nan,
        ]
        amplitudes = [median_of(runs_, lambda r, k=k: r["amplitude_uv"][k]) for k in ("real", "sim")]
        row = f"{dataset:28} {values[0]:8.2f} {values[1]:7.2f} {values[2]:6.2f} {values[3]:7.2f}"
        row += f" {amplitudes[0]:8.1f} {amplitudes[1]:7.1f}"
        for s in STATS:
            pairs = [(r["stats"]["real"][s], r["stats"]["sim"][s]) for r in runs_ if r["stats"]["real"][s] is not None]
            row += f" {'-':>16}" if not pairs else f" {np.median([x for x, _ in pairs]):7.2f} / "
            row += "" if not pairs else f"{np.median([y for _, y in pairs]):6.2f}"
        print(row)


def plot(results: list[dict], ref: dict, path: Path) -> None:
    datasets = [d for d in DATASET_COLORS if any(r["dataset"] == d for r in results)]
    fig, axes = plt.subplots(
        len(datasets),
        3,
        figsize=(13, 2.4 * len(datasets)),
        gridspec_kw={"width_ratios": [2.6, 1.6, 1.2]},
        squeeze=False,
    )
    plt.rcParams.update({"font.size": 8})
    for row, dataset in zip(axes, datasets, strict=True):
        runs_ = [r for r in results if r["dataset"] == dataset]
        example = max(runs_, key=lambda r: r["r_global"])
        color = DATASET_COLORS[dataset]

        ax = row[0]  # mean RMS over the ring, each / its 95th percentile
        trace = example["trace"]
        n = min(len(trace["real"]), int(60 * 100))
        t = np.arange(n) / 100
        for name, c in (("real", REAL), ("sim", SIM)):
            y = trace[name][:n].mean(1)
            ax.plot(t, y / np.percentile(y, 95), color=c, lw=0.8, label="recorded" if name == "real" else "myogen")
        for edge in np.cumsum(trace["lengths"])[:-1]:
            if edge < n:
                ax.axvline(edge / 100, color="#d8d7d2", lw=0.5)
        ax.set_ylabel("ring RMS / p95")
        ax.set_title(f"{dataset}  ·  {example['key'].split('/', 1)[1]}", loc="left", color=color, fontsize=9)
        ax.set_xlim(0, n / 100)
        if row is axes[-1]:
            ax.set_xlabel("time in the simulated motions (s)")
        if row is axes[0]:
            ax.legend(frameon=False, loc="upper right", ncol=2)

        ax = row[1]  # signatures: actions x channels, recorded | myogen
        actions = example["actions"]
        real = np.array([example["signatures"]["real"][a] for a in actions])
        step, mirror = example["placement"]
        sim = place(np.array([example["signatures"]["sim"][a] for a in actions]), step, mirror)
        gap = np.full((len(actions), 1), np.nan)
        top = np.nanpercentile(np.abs(np.hstack([real, sim])), 98)
        ax.imshow(np.hstack([real, gap, sim]), cmap="RdBu_r", vmin=-top, vmax=top, aspect="auto", interpolation="none")
        ax.set_xticks([real.shape[1] / 2 - 0.5, real.shape[1] * 1.5 + 0.5], ["recorded", "myogen"])
        ax.set_yticks(range(len(actions)), actions if len(actions) <= 10 else [""] * len(actions))
        ax.tick_params(length=0)
        ax.set_title(f"action patterns  r = {example['r_signature']:.2f}", loc="left", fontsize=9)

        ax = row[2]  # agreement per session
        metrics = {
            "global": [r["r_global"] for r in runs_],
            "channel": [r["r_channel"] for r in runs_],
            "pattern": [r["r_signature"] for r in runs_],
            "pattern\nreal-real": [np.mean(ref[r["key"]]) for r in runs_ if ref[r["key"]]],
        }
        for i, values in enumerate(metrics.values()):
            jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0]
            ax.scatter(i + np.asarray(jitter), values, s=12, color=color if i < 3 else REAL, alpha=0.8, lw=0)
            if values:
                ax.hlines(np.median(values), i - 0.25, i + 0.25, color="black", lw=1)
        ax.set_xticks(range(len(metrics)), list(metrics))
        ax.set_ylim(-0.3, 1)
        ax.axhline(0, color="#d8d7d2", lw=0.5)
        ax.set_ylabel("r")
        for a in row:
            a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print(f"-> {path}")


def main():
    parser = argparse.ArgumentParser(description="Recorded vs myogen RMS on the free-movement sessions.")
    parser.add_argument("--interim", default="data/interim")
    parser.add_argument("--cache", default="data/cache/synthetic")
    parser.add_argument("--out", default="docs/figures")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--config", default=str(CONFIG), help="the synthetic check settings")
    parser.add_argument("--dataset", action="append", help="repeatable; default: every free-movement dataset")
    parser.add_argument("--subjects", type=int, default=3, help="subjects per dataset")
    parser.add_argument("--workers", type=int, help="processes per session; default: the CPU count")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    check = SyntheticCheck.load(Path(args.config))
    normalizer = Processor(**yaml.safe_load(Path(args.params).read_text())["processor"]).normalizer
    results = []
    for folder in check.sessions(Path(args.interim), args.subjects, args.dataset):
        sim = check.simulate(folder, Path(args.cache), args.workers)
        results.append(check.compare(folder, sim, normalizer))
        print(f"  {results[-1]['key']}: r global {results[-1]['r_global']:.2f}", flush=True)
    ref = reference(results)
    report(results, ref)
    plot(results, ref, Path(args.out) / "synthetic_rms.png")


if __name__ == "__main__":
    main()
