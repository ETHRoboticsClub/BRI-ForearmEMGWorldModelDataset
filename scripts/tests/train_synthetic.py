import argparse
import json
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from compare_synthetic import (
    CONFIG,
    DATASET_COLORS,
    DECODER_LAGS,
    SyntheticCheck,
    decoder_r2,
    features,
    known_placements,
    place,
    segments,
    zscore,
)
from myogen import Subject

from femg.processor import Processor

CONDITIONS = {  # what the decoder of a held-out subject trains on
    "real": "the other subjects' recordings",
    "real + myogen": "the same, and myogen replaying their movements by random people",
    "myogen": "myogen replaying the other subjects' movements only",
    "myogen, own movements": "myogen replaying the held-out subject's movements (upper bound: movements known)",
}


def session_features(env: np.ndarray, targets: np.ndarray, lengths: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(features, joints) of a session's motions, its log envelopes z-scored on their own (as at deployment)."""
    m = max(DECODER_LAGS)
    env_parts, y_parts = segments(zscore(env), lengths), segments(targets, lengths)
    keep = [i for i, n in enumerate(lengths) if n > m]
    return np.concatenate([features(env_parts[i]) for i in keep]), np.concatenate([y_parts[i][m:] for i in keep])


class Experiment:
    """Cross-subject joint decoding per dataset: each subject held out in turn, decoders trained on CONDITIONS,
    tested on the held-out subject's recordings. Real rings in the calibration frame, myogen's electrodes placed in
    it (ring_offset_deg, ring_reversed)."""

    def __init__(self, check: SyntheticCheck, normalizer, placements: dict, cache: Path, workers: int | None) -> None:
        self.check, self.normalizer, self.placements = check, normalizer, placements
        self.cache, self.workers = cache, workers

    def people(self, key: str) -> list[Subject]:
        """The random people who replay a session's movements (the same for every run)."""
        aug = self.check.augment
        base = zlib.crc32(key.encode())
        return [
            Subject.from_dict({**Subject.random(base + k).to_dict(), **aug["overrides"]})
            for k in range(aug["subjects"])
        ]

    def session(self, folder: Path) -> dict | None:
        """A session's features and joints: real (calibration frame) and myogen per random person."""
        meta = json.loads((folder / "meta.json").read_text())
        if meta["key"] not in self.placements:
            return None
        step, mirror = self.placements[meta["key"]]
        real_turn = lambda x: place(place(x, step, mirror), 0, self.check.ring_reversed)  # noqa: E731
        out = {"subject": meta["subject"], "dataset": meta["dataset"], "sim": []}
        for person in self.people(meta["key"]):
            sim = self.check.simulate(folder, self.cache, self.workers, person)
            data = self.check.decoding_data(folder, sim, self.normalizer)
            if "real" not in out:
                out["real"] = session_features(real_turn(data["real"]), data["targets"], data["lengths"])
            out["sim"].append(session_features(data["sim"], data["targets"], data["lengths"]))
        return out

    def run(self, sessions: list[dict]) -> dict[str, dict[str, list[float]]]:
        """Dataset -> condition -> R² per held-out subject."""
        out = {}
        for dataset in sorted({s["dataset"] for s in sessions}):
            mine = [s for s in sessions if s["dataset"] == dataset]
            for subject in sorted({s["subject"] for s in mine}):
                test, others = (
                    [s for s in mine if s["subject"] == subject],
                    [s for s in mine if s["subject"] != subject],
                )
                if not others:
                    continue
                sets = {
                    "real": [s["real"] for s in others],
                    "real + myogen": [s["real"] for s in others] + [x for s in others for x in s["sim"]],
                    "myogen": [x for s in others for x in s["sim"]],
                    "myogen, own movements": [x for s in test for x in s["sim"]],
                }
                test_x, test_y = (np.concatenate(v) for v in zip(*[s["real"] for s in test], strict=True))
                for condition, parts in sets.items():
                    x, y = (np.concatenate(v) for v in zip(*parts, strict=True))
                    r2 = decoder_r2(x, y, test_x, test_y) if y.shape[1] == test_y.shape[1] else np.nan
                    out.setdefault(dataset, {}).setdefault(condition, []).append(r2)
        return out


def report(scores: dict) -> None:
    print("\ncross-subject joint decoding, R² on the held-out subject's recordings (median [min, max] over subjects)")
    print(f"{'':16}" + "".join(f"{c:>26}" for c in CONDITIONS))
    for dataset, by in scores.items():
        cells = [f"{np.nanmedian(v):6.2f} [{np.nanmin(v):5.2f},{np.nanmax(v):5.2f}]" for v in by.values()]
        print(f"{dataset:16}" + "".join(f"{c:>26}" for c in cells) + f"   ({len(by['real'])} subjects)")
    for condition, what in CONDITIONS.items():
        print(f"  {condition}: {what}")


def plot(scores: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.8 / len(CONDITIONS)
    shades = np.linspace(1.0, 0.35, len(CONDITIONS))
    for i, (dataset, by) in enumerate(scores.items()):
        for j, (condition, shade) in enumerate(zip(CONDITIONS, shades, strict=True)):
            v = np.asarray(by[condition], float)
            x = i + (j - (len(CONDITIONS) - 1) / 2) * width
            ax.bar(
                x,
                np.nanmedian(v),
                width * 0.9,
                color=DATASET_COLORS[dataset],
                alpha=shade,
                label=condition if i == 0 else None,
            )
            ax.plot(np.full(len(v), x), v, "k.", ms=3)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(range(len(scores)), list(scores))
    ax.set_ylabel("R² on the held-out subject")
    ax.set_title("Joint decoder across subjects: does myogen data help? (bars: median, dots: subjects)", loc="left")
    ax.legend(fontsize=8, ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"-> {path}")


def main():
    parser = argparse.ArgumentParser(description="Does myogen data help a cross-subject joint decoder?")
    parser.add_argument("--interim", default="data/interim")
    parser.add_argument("--cache", default="data/cache/synthetic")
    parser.add_argument("--out", default="docs/figures")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--dataset", action="append", help="repeatable; default: every free-movement dataset")
    parser.add_argument("--subjects", type=int, help="subjects per dataset (default: tune + validate)")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--calibration", default="data/processed/calibration")
    args = parser.parse_args()

    check = SyntheticCheck.load(Path(args.config))
    normalizer = Processor(**yaml.safe_load(Path(args.params).read_text())["processor"]).normalizer
    placements = known_placements(Path(args.calibration))
    experiment = Experiment(check, normalizer, placements, Path(args.cache), args.workers)
    folders = check.sessions(Path(args.interim), args.subjects, args.dataset, "all", set(placements))
    sessions = [experiment.session(f) for f in folders]
    scores = experiment.run([s for s in sessions if s is not None])
    report(scores)
    plot(scores, Path(args.out) / "synthetic_training.png")


if __name__ == "__main__":
    main()
