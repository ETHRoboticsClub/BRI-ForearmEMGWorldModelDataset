import json
import threading
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from femg.joints import CANONICAL_JOINTS, JOINT_INDEX

HANDS = Path(__file__).parent / "hands"
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
    """name -> title of every mapping file in src/femg/hands."""
    return {path.stem: read_mapping(path)[0].get("title", path.stem) for path in sorted(HANDS.glob("*.json"))}


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
    """A hand model and its mapping file (src/femg/hands/<name>.json): every joint once, each label at most once."""

    def __init__(self, name: str = "shadow_hand"):
        self.name = name
        self.settings, rows = read_mapping(HANDS / f"{name}.json")
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


def neutral_pose(model, data):
    for j in range(model.njnt):
        lo, hi = model.jnt_range[j]
        data.qpos[model.jnt_qposadr[j]] = np.clip(0.0, lo, hi) if model.jnt_limited[j] else 0.0
    mujoco.mj_forward(model, data)


def hand_frame(model, data, palmar_joint: str, radial_body: str, ulnar_body: str):
    """Unit vectors (palmar, radial, distal) of the hand at its neutral pose, from three landmarks of the file."""
    j = model.joint(palmar_joint).id
    tip = model.jnt_bodyid[j]
    neutral_pose(model, data)
    before = tips_center(model, data, tip).copy()
    data.qpos[model.jnt_qposadr[j]] += 0.3
    mujoco.mj_forward(model, data)
    palmar = unit_vector(tips_center(model, data, tip) - before)
    neutral_pose(model, data)
    radial = data.xpos[model.body(radial_body).id] - data.xpos[model.body(ulnar_body).id]
    radial = unit_vector(radial - radial.dot(palmar) * palmar)
    distal = np.cross(palmar, radial)
    if (tips_center(model, data, tip) - data.xpos[model.body(radial_body).id]).dot(distal) < 0:
        distal = -distal
    return palmar, radial, distal


def segment_center(model, data, body: int) -> np.ndarray:
    """Centre of the body's geometry (bounding boxes): its centre of mass can sit on the joint (MyoHand)."""
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
    if not geoms:
        return data.xipos[body].copy()
    centers = [data.geom_xpos[g] + data.geom_xmat[g].reshape(3, 3) @ model.geom_aabb[g][:3] for g in geoms]
    return np.mean(centers, axis=0)


def tips_center(model, data, body: int) -> np.ndarray:
    """Mean of the fingertips below a body: one fingertip for a finger joint, all of them for the wrist."""
    parents = set(model.body_parentid[1:])
    tips = [b for b in range(1, model.nbody) if b not in parents and is_below(model, b, body)]
    return np.mean([segment_center(model, data, b) for b in tips or [body]], axis=0)


def is_below(model, body: int, ancestor: int) -> bool:
    while body:
        if body == ancestor:
            return True
        body = model.body_parentid[body]
    return False


def visible_bounds(model, data) -> tuple[np.ndarray, float]:
    """Centre and half diagonal of the box around everything visible (geom groups 0-2) in the current pose."""
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(model.ngeom):
        if model.geom_group[g] > 2:
            continue
        frame = data.geom_xmat[g].reshape(3, 3)
        middle = data.geom_xpos[g] + frame @ model.geom_aabb[g][:3]
        half = np.abs(frame) @ model.geom_aabb[g][3:]
        low, high = np.minimum(low, middle - half), np.maximum(high, middle + half)
    return (low + high) / 2, float(np.linalg.norm(high - low) / 2)


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix about a unit axis (Rodrigues)."""
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def unit_vector(v):
    return v / (np.linalg.norm(v) or 1.0)


def label_text(rule: Rule, kin_row, unit: bool, how: str) -> str:
    if not rule.label:
        return ""
    value = kin_row[JOINT_INDEX[rule.label]]
    name = short_name(rule.label)
    if how != "label":
        return f"{name} ≈" if how == "derived" else ""
    return f"{name} {value:.2f}" if unit else f"{name} {np.rad2deg(value):.0f}°"


class HandScene:
    """A hand in a viser server, posed one frame at a time. switch() changes the hand model."""

    def __init__(self, port: int, host: str = "127.0.0.1", dark: bool = False, name: str = "shadow_hand"):
        import viser

        self.server = viser.ViserServer(host=host, port=port, verbose=False)
        self.server.on_client_connect(lambda client: threading.Timer(0.5, self.prepare_client, (client,)).start())
        self.set_theme(dark)
        self.labels_visible = True
        self.switch(name)

    def switch(self, name: str):
        from mjviser import ViserMujocoScene

        self.server.scene.reset()
        self.hand = HandMap(name)
        self.model, self.data, self.ids = self.hand.load()
        self.scene = ViserMujocoScene(self.server, self.model, num_envs=1)
        self.scene.show_tendons = False  # musculoskeletal hands (MyoHand): show the bones, not the muscles
        add_frame = self.server.scene.add_frame
        groups = [add_frame(name, show_axes=False) for name in ("/bodies", "/hull", "/decor", "/labels")]
        self.frames = [self.scene.fixed_bodies_frame, *groups]
        self.turn = np.eye(3)
        self.point_camera()
        self.pivots = self.group_by_pivot()
        self.labels = [
            self.server.scene.add_label(f"/labels/{i}", "", font_screen_scale=0.8, visible=self.labels_visible)
            for i in range(len(self.pivots))
        ]
        self.hide_panel()

    def point_camera(self):
        """Palm seen from the thumb side, fingers up, the whole model in view and centred: the same for every hand.
        The centre of the model is also the pivot of the trackball, so it stays centred while it turns."""
        s = self.hand.settings
        palmar, radial, distal = hand_frame(self.model, self.data, s["palmar_joint"], s["radial_body"], s["ulnar_body"])
        neutral_pose(self.model, self.data)
        center, radius = visible_bounds(self.model, self.data)
        position = center + 1.7 * radius * unit_vector(1.4 * palmar + 0.6 * radial - 0.3 * distal)
        self.camera = (position, center, distal)
        self.pivot = center
        self.server.initial_camera.position, self.server.initial_camera.look_at = tuple(position), tuple(center)
        self.server.initial_camera.up = tuple(distal)
        for client in self.server.get_clients().values():
            self.aim(client)

    def aim(self, client, attempts: int = 10):
        """Point a viewer at the centre of the hand, retrying while it has not reported its camera yet."""
        position, center, up = self.camera
        try:
            client.camera.up_direction = up
            client.camera.position, client.camera.look_at = position, center
        except AssertionError:
            if attempts > 1:
                threading.Timer(0.3, self.aim, (client, attempts - 1)).start()

    def orbit(self, right: float, up: float):
        """Trackball: turn the hand itself about the viewer's vertical and horizontal axes (rad). The camera stays
        still, so viser's orbit limits (stuck at the poles) never come into play."""
        clients = list(self.server.get_clients().values())
        try:
            camera = clients[0].camera
            forward = unit_vector(np.asarray(camera.look_at) - np.asarray(camera.position))
            top = np.asarray(camera.up_direction, float)
        except (IndexError, AssertionError):
            return
        side = unit_vector(np.cross(forward, top))
        top = np.cross(side, forward)
        self.turn = rotation(top, right) @ rotation(side, up) @ self.turn
        self.apply_turn()

    def apply_turn(self):
        """Rotate the whole hand (all of mjviser's scene groups and the labels) by self.turn about its center."""
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.turn.flatten())
        shift = self.pivot - self.turn @ self.pivot
        for frame in self.frames:
            frame.wxyz, frame.position = quat, shift

    def zoom(self, factor: float):
        for client in self.server.get_clients().values():
            try:
                target = np.asarray(client.camera.look_at)
                client.camera.position = target + factor * (np.asarray(client.camera.position) - target)
                client.camera.look_at = target
            except AssertionError:
                pass

    def reset_view(self):
        self.turn = np.eye(3)
        self.apply_turn()
        for client in self.server.get_clients().values():
            self.aim(client)

    def prepare_client(self, client):
        self.aim(client)
        client.gui.main_panel.minimize()

    def group_by_pivot(self) -> list[list[Rule]]:
        """Joints that turn about the same point (e.g. MCP spread + flexion) share one label."""
        neutral_pose(self.model, self.data)
        groups: list[tuple[np.ndarray, list[Rule]]] = []
        for r in self.hand.rules:
            if not r.label:
                continue
            anchor = self.data.xanchor[self.ids[r.joint]].copy()
            for position, rules in groups:
                if np.linalg.norm(position - anchor) < 0.004:
                    rules.append(r)
                    break
            else:
                groups.append((anchor, [r]))
        return [rules for _, rules in groups]

    def show(self, kin_row: np.ndarray, unit: bool = False, centers: np.ndarray | None = None):
        set_pose(self.model, self.data, self.hand, self.ids, kin_row, unit, centers)
        how = self.hand.angles(kin_row, unit, centers)
        self.scene.update_from_mjdata(self.data)
        self.scene.fixed_bodies_frame.position = self.pivot - self.turn @ self.pivot  # mjviser resets it on update
        for label, rules in zip(self.labels, self.pivots, strict=True):
            label.position = self.data.xanchor[self.ids[rules[0].joint]]
            label.text = "  ".join(t for r in rules if (t := label_text(r, kin_row, unit, how[r.joint][1])))

    def set_theme(self, dark: bool):
        self.server.gui.configure_theme(dark_mode=dark, show_logo=False, show_share_button=False)
        self.hide_panel()

    def hide_panel(self):
        """The viser control panel is not used: keep it minimized (configure_theme opens it again)."""
        self.server.gui.main_panel.minimize()
        for client in self.server.get_clients().values():
            client.gui.main_panel.minimize()

    def show_labels(self, visible: bool):
        self.labels_visible = visible
        for label in self.labels:
            label.visible = visible
