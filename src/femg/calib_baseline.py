import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from femg.hand.joints import JOINT_INDEX
from femg.normalize import hand_flexion
from femg.signal_utils import envelope

CONFIG = Path(__file__).parent / "config" / "calib_baseline.yaml"
FOLDER = "calibration"  # data/processed/calibration
FINGERS = ("index", "middle", "ring", "pinky")


@dataclass
class Match:
    """Where a band sits relative to the baseline frame: VirtualArray(rotation=rotation_deg, mirror=mirror) on its
    electrodes (or `CalibBaseline.transform` on its virtual channels) brings it there."""

    steps: int  # rotation in virtual electrodes
    mirror: bool
    score: float  # correlation of the aligned signatures with the templates
    margin: float  # score minus the second best placement: how sure the match is
    n_angles: int = 8

    @property
    def rotation_deg(self) -> float:
        return self.steps * 360 / self.n_angles


@dataclass
class Baseline:
    """The calibration baseline: per action, the spatial signature on the virtual ring (template, n_angles values)
    and its spread across sessions; per session, where its band sat (sessions)."""

    templates: dict[str, np.ndarray]
    spread: dict[str, np.ndarray]
    counts: dict[str, int]
    sessions: pd.DataFrame
    settings: dict = field(default_factory=dict)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        baseline = {
            "actions": list(self.templates),
            "templates": {a: v.round(4).tolist() for a, v in self.templates.items()},
            "spread": {a: v.round(4).tolist() for a, v in self.spread.items()},
            "counts": self.counts,
            "settings": self.settings,
        }
        (folder / "baseline.json").write_text(json.dumps(baseline, indent=1))
        self.sessions.to_parquet(folder / "sessions.parquet", index=False)

    @classmethod
    def load(cls, folder: Path) -> "Baseline":
        baseline = json.loads((folder / "baseline.json").read_text())
        return cls(
            templates={a: np.array(v) for a, v in baseline["templates"].items()},
            spread={a: np.array(v) for a, v in baseline["spread"].items()},
            counts=baseline["counts"],
            sessions=pd.read_parquet(folder / "sessions.parquet"),
            settings=baseline["settings"],
        )


@dataclass(frozen=True)
class CalibBaseline:
    """Builds the calibration baseline from the processed sessions, and matches a band to it.

    A session's signature: for each action (held hand states found in its labels, config actions), the mean log RMS
    of each virtual channel minus its mean at rest, minus the ring mean: n_angles values, the action's spatial pattern
    around the forearm, independent of gain. No dataset documents its band's rotation or direction, so the
    signatures are aligned first: every session is matched to the mean over 16 placements (n_angles rotations x 2
    directions), the mean recomputed, until it settles; then the whole frame is turned so that each anchored action
    (anchor_deg: wrist flexors palmar) peaks where the anatomy puts it. With one anchor the frame's rotation is fixed;
    its direction (which way is radial) only consistent across sessions, not anatomical.

    A new user: record the actions, compute the signature, `match` it to the templates: the Match gives the rotation
    and direction to give VirtualArray for that band."""

    actions: dict[str, dict]
    still_per_s: float
    min_s: float
    search_mirror: bool
    iterations: int

    @classmethod
    def load(cls, path: Path = CONFIG) -> "CalibBaseline":
        return cls(**yaml.safe_load(Path(path).read_text()))

    # Signatures

    @staticmethod
    def joint(labels: np.ndarray, name: str) -> np.ndarray:
        """(T,): a canonical joint, a finger (index, middle, ring, pinky: mean MCP / PIP flexion) or finger_flexion."""
        if name == "finger_flexion":
            return hand_flexion(labels)
        if name in FINGERS:
            columns = [JOINT_INDEX[f"{name}_mcp_flex"], JOINT_INDEX[f"{name}_pip_flex"]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # unlabelled frames
                return np.nanmean(labels[:, columns], axis=1)
        return labels[:, JOINT_INDEX[name]]

    def frames(self, labels: np.ndarray, label_fs: float, cues: tuple[str, np.ndarray] | None = None) -> dict:
        """Action -> (T,) bool: the held frames of each action that has at least min_s of them. cues: the session's
        "<dataset>/<session>" and stimulus (T,), for the actions defined by cues."""
        out = {}
        for name, spec in self.actions.items():
            if "cues" in spec:
                if cues is None or cues[0] not in spec["cues"]:
                    continue
                mask = np.asarray(cues[1][: len(labels)]) == spec["cues"][cues[0]]
            else:
                mask = np.ones(len(labels), bool)
                for joint, (low, high) in spec["joints"].items():
                    x = self.joint(labels, joint)
                    with np.errstate(invalid="ignore"):
                        mask &= np.isfinite(x) & (np.abs(np.gradient(x)) * label_fs <= self.still_per_s)
                        mask &= (x >= (-np.inf if low is None else low)) & (x <= (np.inf if high is None else high))
            if mask.sum() >= self.min_s * label_fs:
                out[name] = mask
        return out

    def signature(
        self, virtual: np.ndarray, labels: np.ndarray, meta: dict, stimulus: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Action -> (n_angles,): its spatial pattern on the virtual ring (empty without rest)."""
        per_label = meta["fs"] // meta["label_fs"]
        window = max(1, round(meta["processor"]["rms_window_s"] * meta["label_fs"]))
        log_env = np.log(envelope(virtual, per_label, window).astype(np.float64) + 1e-6)
        n = min(len(log_env), len(labels))
        cues = None if stimulus is None else (f"{meta['dataset']}/{meta['session']}", stimulus[:n])
        frames = self.frames(labels[:n], meta["label_fs"], cues)
        if "rest" not in frames:
            return {}
        rest = log_env[:n][frames["rest"]].mean(axis=0)
        out = {}
        for name, mask in frames.items():
            if name != "rest":
                pattern = log_env[:n][mask].mean(axis=0) - rest
                out[name] = pattern - pattern.mean()
        return out

    def session_signature(self, folder: Path) -> dict[str, np.ndarray]:
        meta = json.loads((folder / "meta.json").read_text())
        virtual = np.load(folder / "virtual.npy", mmap_mode="r")
        return self.signature(virtual, np.load(folder / "labels.npy"), meta, np.load(folder / "stimulus.npy"))

    # Placements

    @staticmethod
    def transform(x: np.ndarray, steps: int, mirror: bool) -> np.ndarray:
        """(..., n_angles): the virtual channels of a band reflected (mirror) and then turned by `steps` electrodes,
        as VirtualArray(rotation=steps * 360 / n_angles, mirror=mirror) gives them."""
        n = x.shape[-1]
        if mirror:
            x = x[..., (-np.arange(n)) % n]
        return np.roll(x, steps, axis=-1)

    def placements(self, n: int) -> list[tuple[int, bool]]:
        return [(k, m) for m in ((False, True) if self.search_mirror else (False,)) for k in range(n)]

    @classmethod
    def compose(cls, first: tuple[int, bool], then: tuple[int, bool], n: int) -> tuple[int, bool]:
        """The single placement equal to `first`, then `then`."""
        probe = np.arange(n, dtype=float)
        target = cls.transform(cls.transform(probe, *first), *then)
        for m in (False, True):
            for k in range(n):
                if np.array_equal(cls.transform(probe, k, m), target):
                    return k, m
        raise AssertionError("placements are closed under composition")

    def match(self, signature: dict[str, np.ndarray], templates: dict[str, np.ndarray]) -> Match | None:
        """The placement whose transformed signature correlates best with the templates (shared actions only)."""
        shared = [a for a in signature if a in templates]
        if not shared:
            return None
        n = len(templates[shared[0]])
        x = np.stack([signature[a] for a in shared])
        y = np.concatenate([templates[a] for a in shared])
        scores = {p: float(np.corrcoef(self.transform(x, *p).ravel(), y)[0, 1]) for p in self.placements(n)}
        ranked = sorted(scores, key=scores.get, reverse=True)
        best, second = ranked[0], ranked[1]
        return Match(best[0], best[1], scores[best], scores[best] - scores[second], n)

    # The baseline

    def align(self, signatures: dict[str, dict[str, np.ndarray]]) -> tuple[dict[str, np.ndarray], dict[str, Match]]:
        """Group alignment: templates (action -> mean aligned signature) and each session's Match."""
        keys = [k for k in signatures if signatures[k]]
        start = max(keys, key=lambda k: len(signatures[k]))
        templates = dict(signatures[start])
        matches = {}
        for _ in range(self.iterations):
            new = {k: self.match(signatures[k], templates) for k in keys}
            new = {k: m for k, m in new.items() if m is not None}
            settled = matches and all(
                (new[k].steps, new[k].mirror) == (matches[k].steps, matches[k].mirror) for k in new
            )
            matches = new
            templates = self.mean(signatures, matches)
            if settled:
                break
        return templates, matches

    def mean(self, signatures, matches) -> dict[str, np.ndarray]:
        aligned = {}
        for k, m in matches.items():
            for a, x in signatures[k].items():
                aligned.setdefault(a, []).append(self.transform(x, m.steps, m.mirror))
        return {a: np.mean(xs, axis=0) for a, xs in aligned.items()}

    @staticmethod
    def peak_deg(pattern: np.ndarray) -> float:
        """Circular mean angle of the positive part of a pattern on the ring."""
        angles = np.deg2rad(np.arange(len(pattern)) * 360 / len(pattern))
        w = np.clip(pattern, 0, None)
        return float(np.rad2deg(np.arctan2((w * np.sin(angles)).sum(), (w * np.cos(angles)).sum())) % 360)

    def anchor(self, templates: dict[str, np.ndarray]) -> tuple[int, bool]:
        """The placement of the whole frame that puts each anchored action's peak nearest its anatomical angle."""
        anchored = {a: s["anchor_deg"] for a, s in self.actions.items() if "anchor_deg" in s and a in templates}
        if not anchored:
            return 0, False
        n = len(next(iter(templates.values())))

        def error(p):
            return sum(
                abs((self.peak_deg(self.transform(templates[a], *p)) - deg + 180) % 360 - 180)
                for a, deg in anchored.items()
            )

        return min(self.placements(n) if self.search_mirror else [(k, False) for k in range(n)], key=error)

    def fit(self, signatures: dict[str, dict[str, np.ndarray]], info: dict[str, dict] | None = None) -> Baseline:
        """signatures: session key -> action -> pattern. info: session key -> extra columns (dataset, subject)."""
        templates, matches = self.align(signatures)
        frame = self.anchor(templates)
        n = len(next(iter(templates.values())))
        rows = []
        for key, m in matches.items():
            steps, mirror = self.compose((m.steps, m.mirror), frame, n)
            rows.append(
                {
                    "key": key,
                    **(info or {}).get(key, {}),
                    "rotation_deg": steps * 360 / n,
                    "mirror": mirror,
                    "score": m.score,
                    "margin": m.margin,
                    "actions": ",".join(signatures[key]),
                }
            )
        matches = {
            r["key"]: Match(round(r["rotation_deg"] * n / 360), r["mirror"], r["score"], r["margin"], n) for r in rows
        }
        templates = self.mean(signatures, matches)
        aligned = {}
        for k, m in matches.items():
            for a, x in signatures[k].items():
                aligned.setdefault(a, []).append(self.transform(x, m.steps, m.mirror))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spread = {a: np.std(xs, axis=0) for a, xs in aligned.items()}
        return Baseline(
            templates=templates,
            spread=spread,
            counts={a: len(xs) for a, xs in aligned.items()},
            sessions=pd.DataFrame(rows),
            settings={"actions": self.actions, "still_per_s": self.still_per_s, "min_s": self.min_s},
        )

    def build(self, folders: list[Path]) -> Baseline:
        """The baseline from processed session folders (data/processed/<dataset>/<subject>/<session>)."""
        signatures, info = {}, {}
        for folder in folders:
            meta = json.loads((folder / "meta.json").read_text())
            signatures[meta["key"]] = self.session_signature(folder)
            info[meta["key"]] = {"dataset": meta["dataset"], "subject": meta["subject"], "session": meta["session"]}
        signatures = {k: s for k, s in signatures.items() if s}
        if not signatures:
            raise ValueError("no session has rest and another action held long enough")
        return self.fit(signatures, info)
