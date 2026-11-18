import argparse
import json
import logging
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from myogen import Electrode, Electrodes, JointAdapter, SimParams, Simulator, Subject
from myogen.inputs.motion import REST_POSTURE
from myogen.tools.calibration import statistics

from femg.calib_baseline import CalibBaseline
from femg.electrodes import ElectrodeLayout
from femg.hand.joints import JOINT_INDEX
from femg.hand.mapping import HandMap
from femg.normalize import Normalizer, as_array, hand_flexion
from femg.processor import Processor
from femg.signal_utils import causal_filter, resample, runs

CONFIG = Path(__file__).parents[2] / "src/femg/config/synthetic.yaml"
DATASET_COLORS = {"emg_fk": "#2a78d6", "ninapro_db2": "#eb6834", "ninapro_db7": "#1baf7a", "ninapro_db8": "#8a5cd1"}
REAL, SIM = "#5f5e5a", "#d1495b"
N_ANGLES = 8  # electrodes per ring
STATS = {  # myogen statistic: label
    "autocorr_50ms": "autocorr 50 ms",
    "autocorr_200ms": "autocorr 200 ms",
    "neighbour_correlation": "neighbour r",
    "rest_move_ratio": "rest / move",
    "median_freq_hz": "median Hz",
    "kurtosis": "kurtosis",
}
DISTANCES = (1, 2, 3, 4)  # electrodes apart around a ring
DECODER_LAGS = (0, 5, 10, 20)  # label frames the EMG features lead the joints by
DECODER_ALPHA = 10.0  # ridge penalty on z-scored features
MAX_LAG_S = 0.4  # EMG -> movement lags searched


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
    spots: bool
    placement: str
    sim_anchor: bool

    @classmethod
    def load(cls, path: Path = CONFIG) -> "SyntheticCheck":
        return cls(**yaml.safe_load(path.read_text()))

    def settings(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("free_movement", "placement", "sim_anchor")}

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

    def electrodes(self, meta: dict) -> tuple[Electrodes, np.ndarray, int]:
        """myogen electrodes at the session's ring channels (ring by ring, in increasing angle) and, with spots, at
        its electrodes placed over a muscle on the forearm (DB2 / DB7: FDS, EDC); those channels; how many are ring."""
        layout = ElectrodeLayout.from_meta(meta)
        rings = layout.positions(("ring",))
        channels = np.flatnonzero(np.isfinite(rings[:, 0]))
        channels = channels[np.lexsort((rings[channels, 0], rings[channels, 1]))]
        if len(channels) % N_ANGLES:
            raise ValueError(f"{meta['key']}: {len(channels)} ring channels, not rings of {N_ANGLES}")
        n_ring = len(channels)
        positions = rings
        if self.spots:
            spots = layout.positions(("spot",))
            on_forearm = np.isfinite(spots[:, 0]) & (spots[:, 1] > 0) & (spots[:, 1] < 1)
            channels = np.concatenate([channels, np.flatnonzero(on_forearm)])
            positions = np.where(np.isfinite(rings), rings, spots)
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
        electrodes = Electrodes(electrodes, derivation=derivation, reference=self.monopolar_reference, **device)
        return electrodes, channels, n_ring

    def simulate(self, folder: Path, cache: Path, workers: int | None) -> dict:
        """The motions of a session through myogen -> the EMG as interim (V, fs, filtered), cached."""
        meta = json.loads((folder / "meta.json").read_text())
        file = cache / f"{meta['key']}.npz"
        if file.exists():
            saved = np.load(file, allow_pickle=False)
            if json.loads(str(saved["settings"])) == self.settings():
                return {k: saved[k] for k in ("emg", "windows", "channels", "n_ring")}
        kin, stimulus = np.load(folder / "kin.npy"), np.load(folder / "stimulus.npy")
        windows = self.windows(stimulus, meta["label_fs"])
        angles = self.model_angles(kin, meta, stimulus)
        electrodes, channels, n_ring = self.electrodes(meta)
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
        out = {
            "emg": np.concatenate(emg).astype(np.float32),
            "windows": np.array(windows),
            "channels": channels,
            "n_ring": np.array(n_ring),
        }
        file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(file, **out, settings=json.dumps(self.settings()))
        return out

    # The comparison

    def kept(self, windows: np.ndarray, label_fs: float) -> list[np.ndarray]:
        """Label frames of each window, its first skip_s left out."""
        skip = round(self.skip_s * label_fs)
        return [np.arange(a + skip, b) for a, b in windows]

    def frames(self, windows: np.ndarray, label_fs: float) -> tuple[np.ndarray, list[np.ndarray]]:
        """Frames kept for the metrics in the concatenated windows, and the kept frames of each window."""
        offsets = np.cumsum([0] + [b - a for a, b in windows])
        kept = self.kept(windows, label_fs)
        return np.concatenate([o + k - a for o, k, (a, _) in zip(offsets[:-1], kept, windows, strict=True)]), kept

    def sim_signature(self, folder: Path, sim: dict, normalizer: Normalizer) -> dict:
        """Action -> (n_ring,) myogen's ring pattern (signature) for this session's motions."""
        meta = json.loads((folder / "meta.json").read_text())
        frames, _ = self.frames(sim["windows"], meta["label_fs"])
        log_env = np.log(normalizer.envelope(sim["emg"][:, : int(sim["n_ring"])].astype(np.float64))[frames] + 1e-9)
        stimulus = np.concatenate([np.load(folder / "stimulus.npy")[a:b] for a, b in sim["windows"]])[frames]
        labels = Normalizer.kin(
            np.concatenate([np.load(folder / "kin.npy")[a:b] for a, b in sim["windows"]]), meta["kin_stats"]
        )[frames]
        return signature(log_env, stimulus, labels, meta["label_fs"])

    def compare(self, folder: Path, sim: dict, normalizer: Normalizer, placement: tuple | None = None) -> dict:
        """Real against myogen. placement (steps, mirror, flip, sim_steps): the real band's placement, known from
        elsewhere (the calibration baseline): the real ring channels are turned into the calibration frame, and
        myogen's by sim_steps (sim_anchor). None: myogen's ring is turned to the placement that matches the real
        patterns best (optimistic)."""
        meta = json.loads((folder / "meta.json").read_text())
        per_label, label_fs = meta["fs"] // meta["label_fs"], meta["label_fs"]
        channels, windows, n_ring = sim["channels"], sim["windows"], int(sim["n_ring"])
        real_emg = np.load(folder / "emg.npy", mmap_mode="r")
        real = np.concatenate([real_emg[a * per_label : b * per_label, channels] for a, b in windows])
        real = real.astype(np.float64)
        if meta["electrode_type"] == "monopolar":
            real = real - real.mean(axis=1, keepdims=True)  # as the virtual array does
        frames, kept = self.frames(windows, label_fs)
        samples = (frames[:, None] * per_label + np.arange(per_label)).ravel()

        emg = {"real": real[:, :n_ring], "sim": sim["emg"][:, :n_ring].astype(np.float64)}
        env = {k: normalizer.envelope(x)[frames] for k, x in emg.items()}
        stimulus = np.concatenate([np.load(folder / "stimulus.npy")[a:b] for a, b in windows])[frames]
        labels = Normalizer.kin(
            np.concatenate([np.load(folder / "kin.npy")[a:b] for a, b in windows]), meta["kin_stats"]
        )[frames]
        targets, lengths = joint_targets(labels), [len(k) for k in kept]
        log_env = {k: np.log(v + 1e-9) for k, v in env.items()}
        sigs = {k: signature(v, stimulus, labels, label_fs) for k, v in log_env.items()}
        actions = sorted(set(sigs["real"]) & set(sigs["sim"]))
        if placement is None:  # move myogen's band onto the real one
            step, mirror, _ = best_placement(sigs["sim"], sigs["real"], actions)
            turns = {"sim": lambda x: place(x, step, mirror)}
        else:  # both bands into the calibration frame
            step, mirror, flip, sim_step = placement
            turns = {
                "real": lambda x: place(place(x, step, mirror), 0, flip),
                "sim": lambda x: place(x, sim_step, False),
            }
        for k, turn in turns.items():
            emg[k], env[k], log_env[k] = turn(emg[k]), turn(env[k]), turn(log_env[k])
            sigs[k] = {a: turn(v) for a, v in sigs[k].items()}
        r_sig = (
            pearson(*(np.concatenate([sigs[k][a] for a in actions]) for k in ("real", "sim"))) if actions else np.nan
        )

        moving = stimulus > 0 if (stimulus > 0).any() else hand_flexion(labels) > 0.3
        rest = np.repeat(stimulus == 0, per_label) if (stimulus == 0).any() and (stimulus > 0).any() else None
        stats = {k: statistics(x[samples], meta["fs"], "raw", rest) for k, x in emg.items()}
        spot_env = {
            k: np.log(normalizer.envelope(x)[frames] + 1e-9)
            for k, x in (("real", real[:, n_ring:]), ("sim", sim["emg"][:, n_ring:]))
        }
        return {
            "key": meta["key"],
            "dataset": meta["dataset"],
            "known": placement is not None,
            "r_global": pearson(log_env["real"].mean(1), log_env["sim"].mean(1)),
            "r_per_channel": [pearson(log_env["real"][:, c], log_env["sim"][:, c]) for c in range(n_ring)],
            "r_channel": float(
                np.median([pearson(log_env["real"][:, c], log_env["sim"][:, c]) for c in range(n_ring)])
            ),
            "r_signature": r_sig,
            "placement": (step, mirror),
            "profile": {k: ring_profile(v, moving) for k, v in env.items()},
            "spots": {
                "names": [
                    f"{(a % 360):.0f}°" for a in ElectrodeLayout.from_meta(meta).positions()[channels[n_ring:], 0]
                ],
                "r": [pearson(spot_env["real"][:, c], spot_env["sim"][:, c]) for c in range(len(channels) - n_ring)],
                "trace": {k: np.exp(v) for k, v in spot_env.items()},
            },
            "amplitude_uv": {k: float(np.median(np.percentile(v, 95, axis=0))) * 1e6 for k, v in env.items()},
            "stats": {k: {s: stats[k][s] for s in STATS} for k in stats},
            "distance_r": {k: distance_correlation(v) for k, v in log_env.items()},
            "lag_ms": {k: movement_lag_ms(v, targets, lengths, label_fs) for k, v in log_env.items()},
            "transfer": transfer(log_env["real"], log_env["sim"], targets, lengths),
            "decoder": (log_env["real"], targets, lengths),
            "signatures": sigs,
            "actions": actions,
            "trace": {"real": env["real"], "sim": env["sim"], "lengths": lengths},
        }


def ring_profile(env: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """(rings, 8) mean RMS of each channel while moving, over the mean of its ring: the shape around the forearm."""
    rings = env[moving].mean(0).reshape(-1, N_ANGLES)
    return rings / rings.mean(1, keepdims=True)


def distance_correlation(log_env: np.ndarray) -> list[float]:
    """Mean correlation of the envelopes of two electrodes of a ring, DISTANCES apart."""
    rings = log_env.reshape(len(log_env), -1, N_ANGLES)
    z = (rings - rings.mean(0)) / (rings.std(0) + 1e-12)
    return [float(np.mean(z * np.roll(z, -k, axis=2))) for k in DISTANCES]


def joint_targets(labels: np.ndarray) -> np.ndarray:
    """(T, J) the joints labelled in at least 90% of the frames, in 0-1, gaps filled."""
    measured = np.isfinite(labels).mean(0) >= 0.9
    return np.clip(np.stack([fill_nan(labels[:, j]) for j in np.flatnonzero(measured)], axis=1), 0, 1)


def segments(x: np.ndarray, lengths: list[int]) -> list[np.ndarray]:
    """The concatenated simulated motions, apart again."""
    return np.split(x, np.cumsum(lengths)[:-1])


def movement_lag_ms(log_env: np.ndarray, targets: np.ndarray, lengths: list[int], label_fs: float) -> float:
    """How long the mean EMG envelope leads the hand's speed: the peak of their cross-correlation within each motion
    (negative: it lags)."""
    env = segments(zscore(log_env.mean(1)), lengths)
    speed = np.concatenate([np.abs(np.gradient(t, axis=0)).sum(1) for t in segments(targets, lengths)])
    speed = segments(zscore(speed), lengths)
    n = round(MAX_LAG_S * label_fs)
    lags = np.arange(-n, n + 1)
    scores = [
        np.mean(
            np.concatenate(
                [
                    a[max(0, -k) : len(a) - max(0, k)] * b[max(0, k) : len(b) - max(0, -k)]
                    for a, b in zip(env, speed, strict=True)
                    if len(a) > 2 * n
                ]
            )
        )
        for k in lags
    ]
    return 1000 * float(lags[int(np.argmax(scores))]) / label_fs


def features(log_env: np.ndarray) -> np.ndarray:
    """(T - max lag, C * lags) the envelopes now and DECODER_LAGS frames before, within one motion."""
    m = max(DECODER_LAGS)
    return np.hstack([log_env[m - k : len(log_env) - k] for k in DECODER_LAGS])


def decoder_r2(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> float:
    """Ridge on features z-scored by their own set's statistics (as each session is normalized on its own)."""
    train_x, test_x = zscore(train_x), zscore(test_x)
    mean = train_y.mean(0)
    w = np.linalg.solve(train_x.T @ train_x + DECODER_ALPHA * np.eye(train_x.shape[1]), train_x.T @ (train_y - mean))
    error = ((test_x @ w + mean - test_y) ** 2).sum(0)
    return float(np.mean(1 - error / ((test_y - test_y.mean(0)) ** 2).sum(0)))


def decoder_data(log_env: np.ndarray, targets: np.ndarray, lengths: list[int], part: str) -> tuple:
    """(features, joints) of every other motion: part "train" the first repetition of each cue, "test" the second."""
    m = max(DECODER_LAGS)
    keep = [i for i, n in enumerate(lengths) if n > m]
    chosen = keep[0::2] if part == "train" else keep[1::2]
    env, y = segments(log_env, lengths), segments(targets, lengths)
    return np.concatenate([features(env[i]) for i in chosen]), np.concatenate([y[i][m:] for i in chosen])


def transfer(real: np.ndarray, sim: np.ndarray, targets: np.ndarray, lengths: list[int]) -> dict[str, float]:
    """Joint decoders trained on one repetition of each movement, tested on the other: real -> real, myogen -> real,
    myogen -> myogen. Against the decoder trained on another subject (reference): myogen -> real near it means the
    simulated EMG teaches as much as another person's recording."""
    data = {
        (k, part): decoder_data(x, targets, lengths, part)
        for k, x in (("real", real), ("sim", sim))
        for part in ("train", "test")
    }
    return {
        f"{a}_{b}": decoder_r2(*data[a, "train"], *data[b, "test"])
        for a, b in (("real", "real"), ("sim", "real"), ("sim", "sim"))
    }


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)) / (x.std(0) + 1e-12)


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


def reference(results: list[dict]) -> dict[str, dict[str, list[float]]]:
    """Real against real, each session against the sessions of the other subjects of its dataset: pattern r of the
    signatures, and R² of a joint decoder trained on the other subject. Both in the anatomical frame when their
    placements are known, else the other turned to its best placement (as myogen is)."""
    out = {}
    for res in results:
        subject = res["key"].split("/")[1]
        others = [o for o in results if o["dataset"] == res["dataset"] and o["key"].split("/")[1] != subject]
        pattern, decoder = [], []
        test = decoder_data(*res["decoder"], "test")
        for other in others:
            actions = sorted(set(res["signatures"]["real"]) & set(other["signatures"]["real"]))
            if not actions:
                continue
            if res["known"] and other["known"]:
                step, mirror = 0, False
                r = pearson(*(np.concatenate([o["signatures"]["real"][a] for a in actions]) for o in (other, res)))
            else:
                step, mirror, r = best_placement(other["signatures"]["real"], res["signatures"]["real"], actions)
            pattern.append(r)
            env, targets, lengths = other["decoder"]
            if targets.shape[1] == test[1].shape[1]:
                decoder.append(decoder_r2(*decoder_data(place(env, step, mirror), targets, lengths, "train"), *test))
        out[res["key"]] = {"pattern": pattern, "decoder": decoder}
    return out


def median_of(runs_: list[dict], value) -> float:
    return float(np.nanmedian([value(r) for r in runs_]))


def report(results: list[dict], ref: dict) -> None:
    head = f"{'session':28} {'r global':>8} {'r chan':>7} {'r sig':>6} {'ref sig':>7} {'µV real':>8} {'µV sim':>7}"
    print(head + "".join(f" {v:>16}" for v in STATS.values()))
    for res in results:
        row = f"{res['key']:28} {res['r_global']:8.2f} {res['r_channel']:7.2f} {res['r_signature']:6.2f}"
        pattern = ref[res["key"]]["pattern"]
        row += f" {np.mean(pattern) if pattern else np.nan:7.2f}"
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
        refs = [np.mean(ref[r["key"]]["pattern"]) for r in runs_ if ref[r["key"]]["pattern"]]
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
    print("\nmedian per dataset (real / myogen): r between ring electrodes k apart, EMG lead, decoder R² transfer")
    head = "".join(f" {f'r k={k}':>13}" for k in DISTANCES) + f" {'lead ms':>11} {'real>real':>9} {'other>real':>10}"
    print(f"{'':28}" + head + f" {'sim>real':>9} {'sim>sim':>9}")
    for dataset in DATASET_COLORS:
        runs_ = [r for r in results if r["dataset"] == dataset]
        if not runs_:
            continue
        row = f"{dataset:28}"
        for i in range(len(DISTANCES)):
            real, sim = (median_of(runs_, lambda r, k=k, i=i: r["distance_r"][k][i]) for k in ("real", "sim"))
            row += f" {real:5.2f} / {sim:5.2f}"
        real, sim = (median_of(runs_, lambda r, k=k: r["lag_ms"][k]) for k in ("real", "sim"))
        row += f" {real:4.0f} / {sim:4.0f}"
        others = [np.mean(ref[r["key"]]["decoder"]) for r in runs_ if ref[r["key"]]["decoder"]]
        row += f" {median_of(runs_, lambda r: r['transfer']['real_real']):9.2f}"
        row += f" {np.median(others) if others else np.nan:10.2f}"
        for k in ("sim_real", "sim_sim"):
            row += f" {median_of(runs_, lambda r, k=k: r['transfer'][k]):9.2f}"
        print(row)


def report_channels(results: list[dict]) -> None:
    """Per channel at the same anatomical place: the ring (known placement) and the electrodes over a muscle."""
    print("\nper channel, recorded vs myogen RMS over time (r), ring channels by anatomical angle")
    print(f"{'':28}" + "".join(f" {a:>6}" for a in range(0, 360, 360 // N_ANGLES)) + "   spots")
    for dataset in DATASET_COLORS:
        runs_ = [r for r in results if r["dataset"] == dataset]
        if not runs_:
            continue
        per_channel = np.nanmedian([r["r_per_channel"][:N_ANGLES] for r in runs_], axis=0)
        row = f"{dataset + (' (known)' if runs_[0]['known'] else ' (search)'):28}"
        row += "".join(f" {v:6.2f}" for v in per_channel)
        names = runs_[0]["spots"]["names"]
        if names:
            spots = np.nanmedian([r["spots"]["r"] for r in runs_], axis=0)
            row += "   " + "  ".join(f"{n} {v:.2f}" for n, v in zip(names, spots, strict=True))
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
        sim = np.array([example["signatures"]["sim"][a] for a in actions])
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
            "pattern\nreal-real": [np.mean(ref[r["key"]]["pattern"]) for r in runs_ if ref[r["key"]]["pattern"]],
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


def known_placements(calibration: Path) -> dict[str, tuple[int, bool]]:
    """Session key -> (steps, mirror) of its band in the calibration baseline's frame (wrist flexors palmar)."""
    file = calibration / "sessions.parquet"
    if not file.exists():
        return {}
    rows = pd.read_parquet(file)
    return {r.key: (round(r.rotation_deg / (360 / N_ANGLES)) % N_ANGLES, bool(r.mirror)) for r in rows.itertuples()}


def plot_channels(results: list[dict], path: Path) -> None:
    """Per dataset, its best session: the RMS around each ring while moving (each channel over its ring's mean), and
    every ring channel's RMS over time, recorded against myogen, at the same anatomical place."""
    datasets = [d for d in DATASET_COLORS if any(r["dataset"] == d for r in results)]
    fig = plt.figure(figsize=(14, 3.0 * len(datasets)))
    grid = fig.add_gridspec(len(datasets), 2, width_ratios=[1, 3], hspace=0.45, wspace=0.15)
    plt.rcParams.update({"font.size": 8})
    for row, dataset in enumerate(datasets):
        example = max((r for r in results if r["dataset"] == dataset), key=lambda r: r["r_channel"])
        color = DATASET_COLORS[dataset]
        ax = fig.add_subplot(grid[row, 0])
        angles = np.arange(N_ANGLES + 1) * 360 / N_ANGLES
        for i, (real, sim) in enumerate(zip(example["profile"]["real"], example["profile"]["sim"], strict=True)):
            style = {"lw": 1.2, "alpha": 1.0 if i == 0 else 0.5}
            ax.plot(angles, np.r_[real, real[0]], color=REAL, marker="o", ms=3, **style)
            ax.plot(angles, np.r_[sim, sim[0]], color=SIM, marker="o", ms=3, **style)
        ax.set_xticks([0, 90, 180, 270, 360], ["palmar", "radial", "dorsal", "ulnar", "palmar"])
        ax.set_ylabel("RMS while moving / ring mean")
        where = "known placement" if example["known"] else "best placement"
        ax.set_title(f"{dataset}  ·  {example['key'].split('/', 1)[1]}  ({where})", loc="left", color=color, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

        inner = grid[row, 1].subgridspec(2, N_ANGLES // 2, hspace=0.5, wspace=0.12)
        n = min(len(example["trace"]["real"]), 40 * 100)
        t = np.arange(n) / 100
        for c in range(N_ANGLES):
            ax = fig.add_subplot(inner[c // (N_ANGLES // 2), c % (N_ANGLES // 2)])
            for name, line in (("real", REAL), ("sim", SIM)):
                y = example["trace"][name][:n, c]
                ax.plot(t, y / np.percentile(y, 95), color=line, lw=0.6)
            ax.set_title(f"{c * 360 // N_ANGLES}°   r = {example['r_per_channel'][c]:.2f}", fontsize=7.5, loc="left")
            ax.set_yticks([])
            ax.tick_params(labelsize=6.5)
            ax.spines[["top", "right", "left"]].set_visible(False)
    handles = [plt.Line2D([], [], color=c, lw=1.2, label=n) for n, c in (("recorded", REAL), ("myogen", SIM))]
    fig.legend(handles=handles, loc="upper right", frameon=False, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"-> {path}")


def sim_anchor(check, sims: dict, normalizer) -> int:
    """Steps that turn myogen's ring into the calibration frame by the baseline's own rule: each anchored action
    (wrist flexion: a DB2 / DB7 cue) peaks at its angle. One turn for every session (myogen's anatomy is one);
    0 without sessions that have the cue."""
    anchored = {a: s for a, s in CalibBaseline.load().actions.items() if "anchor_deg" in s and "cues" in s}
    patterns = {a: [] for a in anchored}
    for folder, sim in sims.items():
        meta = json.loads((folder / "meta.json").read_text())
        sig = check.sim_signature(folder, sim, normalizer)
        for a, spec in anchored.items():
            cue = spec["cues"].get(f"{meta['dataset']}/{meta['session']}")
            if cue is not None and f"cue {cue}" in sig:
                patterns[a].append(sig[f"cue {cue}"].reshape(-1, N_ANGLES).mean(0))
    mean = {a: np.mean(p, axis=0) for a, p in patterns.items() if p}
    if not mean:
        return 0

    def error(k):
        return sum(
            abs((CalibBaseline.peak_deg(place(p, k, False)) - anchored[a]["anchor_deg"] + 180) % 360 - 180)
            for a, p in mean.items()
        )

    step = min(range(N_ANGLES), key=error)
    print(f"myogen ring anchored ({', '.join(mean)}): turned {step * 360 // N_ANGLES} deg")
    return step


def compare_all(check, sims: dict, normalizer, placements: dict) -> list[dict]:
    """Every session; with known placements, the band direction of the calibration frame (radial one way or the
    other: one choice for every session) taken as the one that matches myogen's patterns best, and reported."""
    if not placements:
        return [check.compare(folder, sim, normalizer) for folder, sim in sims.items()]
    sim_step = sim_anchor(check, sims, normalizer) if check.sim_anchor else 0
    keys = {folder: json.loads((folder / "meta.json").read_text())["key"] for folder in sims}
    known = {folder: sim for folder, sim in sims.items() if keys[folder] in placements}
    if missing := [keys[f] for f in sims if f not in known]:
        print(f"no calibration placement, left out: {missing}")
    by_flip = {
        flip: [check.compare(f, sim, normalizer, (*placements[keys[f]], flip, sim_step)) for f, sim in known.items()]
        for flip in (False, True)
    }
    scores = {flip: np.nanmedian([r["r_signature"] for r in res]) for flip, res in by_flip.items()}
    flip = max(scores, key=scores.get)
    print(
        f"calibration frame direction: pattern r {scores[False]:.2f} as is, {scores[True]:.2f} reversed -> "
        f"{'reversed' if flip else 'as is'}"
    )
    return by_flip[flip]


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
    parser.add_argument("--calibration", default="data/processed/calibration", help="the bands' known placements")
    parser.add_argument("--placement", choices=["calibration", "search"], help="default: the config's")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    check = SyntheticCheck.load(Path(args.config))
    normalizer = Processor(**yaml.safe_load(Path(args.params).read_text())["processor"]).normalizer
    sims = {}
    for folder in check.sessions(Path(args.interim), args.subjects, args.dataset):
        sims[folder] = check.simulate(folder, Path(args.cache), args.workers)
    mode = args.placement or check.placement
    placements = known_placements(Path(args.calibration)) if mode == "calibration" else {}
    if mode == "calibration" and not placements:
        print(f"no calibration baseline in {args.calibration}: best placements instead")
    results = compare_all(check, sims, normalizer, placements)
    ref = reference(results)
    report(results, ref)
    report_channels(results)
    plot(results, ref, Path(args.out) / "synthetic_rms.png")
    plot_channels(results, Path(args.out) / "synthetic_channels.png")


if __name__ == "__main__":
    main()
