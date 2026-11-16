import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from femg.hand.joints import CANONICAL_JOINTS, JOINT_INDEX

MODELS = Path(__file__).parent / "models"
FINGERS = {"thumb": "T", "index": "I", "middle": "M", "ring": "R", "pinky": "P"}
LANDMARKS = ["palmar_joint", "radial_body", "ulnar_body"]


def short_name(name: str) -> str:
    parts = name.split("_")
    if parts[0] not in FINGERS:
        return " ".join(parts)
    parts = name.removesuffix("_flex").split("_")
    fingers = "".join(FINGERS[p] for p in parts if p in FINGERS)
    return " ".join([fingers] + [p for p in parts if p not in FINGERS])


def available_hands() -> dict[str, str]:
    """name -> title of every mapping file in src/femg/hand/models."""
    return {path.stem: read_mapping(path)[0].get("title", path.stem) for path in sorted(MODELS.glob("*.json"))}


def read_mapping(path: Path) -> tuple[dict, list[dict]]:
    """(settings: title, model, entry and the landmarks; one dict per joint, in file order)."""
    spec = json.loads(path.read_text())
    settings = {k: v for k, v in spec.items() if k not in ("landmarks", "joints")} | spec["landmarks"]
    return settings, [{"joint": name, **joint} for name, joint in spec["joints"].items()]


def dump_mapping(settings: dict, joints: dict[str, dict]) -> str:
    """JSON with one line per joint, so the file reads like a table."""
    head = {k: settings[k] for k in ("title", "model", "entry", "builder") if k in settings}
    head["landmarks"] = {k: settings[k] for k in LANDMARKS}
    width = max(len(name) for name in joints) + 3
    rows = [
        f"    {json.dumps(name) + ':':{width}} {json.dumps(spec, separators=(', ', ': '))}"
        for name, spec in joints.items()
    ]
    return json.dumps(head, indent=2)[:-2] + ',\n  "joints": {\n' + ",\n".join(rows) + "\n  }\n}\n"


@dataclass(frozen=True)
class Rule:
    joint: str
    label: str | None
    at_0: float
    at_1: float
    neutral_mean: bool
    source: str | None
    ratio: float
    fallback: float

    def angle(self, value: float, unit: bool, center: float, done: dict[str, float]) -> tuple[float, str]:
        """The joint's angle (rad) and where it came from: "label", "derived" or "fixed"."""
        if self.label and np.isfinite(value):
            if not unit:
                return (1.0 if self.at_1 >= self.at_0 else -1.0) * float(value), "label"
            u = float(np.clip(value, 0, 1))
            if self.neutral_mean and 0 < center < 1:
                return (
                    self.at_0 * (center - u) / center if u < center else self.at_1 * (u - center) / (1 - center)
                ), "label"
            return self.at_0 + u * (self.at_1 - self.at_0), "label"
        if self.source:
            return self.ratio * done[self.source], "derived"
        return self.fallback, "fixed"


def parse_rule(row: dict) -> Rule:
    at_0 = float(row.get("at_0", 0.0))
    at_1 = float(row.get("at_1", at_0))
    missing = row.get("if_missing")
    source, ratio, fallback = None, 1.0, at_0
    if isinstance(missing, dict):
        source, ratio = missing["copy"], float(missing.get("times", 1.0))
    elif missing is not None:
        fallback = float(missing)
    return Rule(row["joint"], row.get("label"), at_0, at_1, row.get("neutral") == "mean", source, ratio, fallback)


class HandMap:
    """A hand model and its mapping file (models/<name>.json): every joint once, each label at most once."""

    def __init__(self, name: str = "shadow_hand"):
        self.name = name
        self.settings, rows = read_mapping(MODELS / f"{name}.json")
        self.title = self.settings.get("title", name)
        self.rules = [parse_rule(row) for row in rows]
        labels = [r.label for r in self.rules if r.label]
        if duplicated := {label for label in labels if labels.count(label) > 1}:
            raise ValueError(f"{name}: labels used for more than one joint: {sorted(duplicated)}")
        if unknown := set(labels) - set(CANONICAL_JOINTS):
            raise ValueError(f"{name}: unknown labels {sorted(unknown)}")
        seen = set()
        for r in self.rules:
            if r.source and r.source not in seen:
                raise ValueError(f"{name}: {r.joint} copies {r.source}, which must be listed before it")
            seen.add(r.joint)

    def load(self):
        model = load_model(self.settings)
        names = {model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE}
        if missing := names - {r.joint for r in self.rules}:
            raise ValueError(f"{self.name}: joints missing in the file: {sorted(missing)}")
        if extra := {r.joint for r in self.rules} - names:
            raise ValueError(f"{self.name}: joints not in the model: {sorted(extra)}")
        ids = {r.joint: model.joint(r.joint).id for r in self.rules}
        return model, mujoco.MjData(model), ids

    def angles(
        self, kin_row: np.ndarray, unit: bool, centers: np.ndarray | None = None
    ) -> dict[str, tuple[float, str]]:
        done, out = {}, {}
        for r in self.rules:
            value = kin_row[JOINT_INDEX[r.label]] if r.label else np.nan
            center = centers[JOINT_INDEX[r.label]] if r.label and centers is not None else 0.5
            angle, how = r.angle(value, unit, center, done)
            done[r.joint] = angle
            out[r.joint] = (angle, how)
        return out


def load_model(settings: dict) -> mujoco.MjModel:
    """A mujoco-menagerie robot ("model", "entry"), or "builder": "module:function" returning an MjSpec/MjModel."""
    if builder := settings.get("builder"):
        import importlib

        module, _, function = builder.partition(":")
        built = getattr(importlib.import_module(module), function)()
        return built.compile() if isinstance(built, mujoco.MjSpec) else built
    import mujoco_menagerie

    return mujoco_menagerie.load(settings["model"], settings.get("entry"))


def set_pose(model, data, hand: HandMap, ids, kin_row, unit: bool = False, centers=None):
    """kin_row: canonical joints in rad, or normalized 0-1 when unit (robust to each dataset's zero and gain)."""
    for joint, (angle, _) in hand.angles(kin_row, unit, centers).items():
        j = ids[joint]
        data.qpos[model.jnt_qposadr[j]] = np.clip(angle, *model.jnt_range[j]) if model.jnt_limited[j] else angle
    mujoco.mj_forward(model, data)


def label_text(rule: Rule, kin_row, unit: bool, how: str) -> str:
    if not rule.label:
        return ""
    value = kin_row[JOINT_INDEX[rule.label]]
    name = short_name(rule.label)
    if how != "label":
        return f"{name} ≈" if how == "derived" else ""
    return f"{name} {value:.2f}" if unit else f"{name} {np.rad2deg(value):.0f}°"
