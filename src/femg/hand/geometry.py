import mujoco
import numpy as np


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
