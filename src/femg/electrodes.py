from dataclasses import dataclass

import numpy as np

FOREARM_LENGTH_CM = 26.0  # adult, elbow to wrist crease
FOREARM_CIRCUMFERENCE_CM = 24.0  # proximal third, where the bands sit
KINDS = ("bipolar", "monopolar")


@dataclass(frozen=True)
class ElectrodeGroup:
    """Electrodes fixed to each other (one band, one grid, one spot), in channel order.

    angle: degrees around the forearm, 0 palmar midline, 90 radial (thumb side), 180 dorsal, 270 ulnar, for either
    arm. z: fraction of the forearm from the elbow crease (0) to the wrist crease (1), < 0 on the upper arm. NaN:
    unknown. Their relative positions are known; where the group sits only up to a shift: angle_sd (deg, >= 180 any
    rotation), z_sd, and mirror (the channels may run the other way round). layout: how it was described (ring,
    grid, spot, unknown, points)."""

    name: str
    kind: str  # bipolar or monopolar
    angles: tuple[float, ...]
    zs: tuple[float, ...]
    angle_sd: float = 0.0
    z_sd: float = 0.0
    mirror: bool = False
    layout: str = "points"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"electrode group {self.name}: kind must be one of {KINDS}, got {self.kind!r}")
        if len(self.angles) != len(self.zs):
            raise ValueError(f"electrode group {self.name}: {len(self.angles)} angles but {len(self.zs)} z")

    def __len__(self) -> int:
        return len(self.angles)

    @classmethod
    def ring(
        cls, name: str, kind: str, n: int, z: float | list[float], first_angle: float | list[float] = 0.0, **placement
    ):
        """n electrodes evenly spaced around the forearm, in increasing angle. Several z: one ring at each, in turn;
        first_angle per ring (rows staggered) or one for all."""
        heights = np.atleast_1d(z)
        starts = np.broadcast_to(np.atleast_1d(first_angle), heights.shape)
        points = [(start + i * 360 / n, at) for at, start in zip(heights, starts, strict=True) for i in range(n)]
        return cls.from_points(name, kind, points, layout="ring", **placement)

    @classmethod
    def grid(cls, name: str, kind: str, rows: int, columns: int, pitch_mm: float, center: list[float], **placement):
        """Row-major: rows go towards the wrist, columns towards increasing angle. center: (angle, z)."""
        degrees = pitch_mm / 10 / FOREARM_CIRCUMFERENCE_CM * 360
        fraction = pitch_mm / 10 / FOREARM_LENGTH_CM
        angle0 = center[0] - degrees * (columns - 1) / 2
        z0 = center[1] - fraction * (rows - 1) / 2
        points = [(angle0 + c * degrees, z0 + r * fraction) for r in range(rows) for c in range(columns)]
        return cls.from_points(name, kind, points, layout="grid", **placement)

    @classmethod
    def spot(cls, name: str, kind: str, angle: float, z: float, **placement):
        """One electrode over a muscle."""
        return cls.from_points(name, kind, [(angle, z)], layout="spot", **placement)

    @classmethod
    def unknown(cls, name: str, kind: str, n: int):
        """n electrodes whose positions are not documented."""
        return cls.from_points(name, kind, [(np.nan, np.nan)] * n, layout="unknown")

    @classmethod
    def from_points(cls, name: str, kind: str, points, **placement):
        angles = tuple(float(a) % 360 if np.isfinite(a) else np.nan for a, _ in points)
        return cls(name, kind, angles, tuple(float(z) for _, z in points), **placement)

    @classmethod
    def from_config(cls, spec: dict) -> "ElectrodeGroup":
        """One entry of `electrodes` in a dataset config: name, kind, layout (ring, grid, spot or unknown) with its
        parameters, and the placement uncertainty."""
        spec = dict(spec)
        layouts = {"ring": cls.ring, "grid": cls.grid, "spot": cls.spot, "unknown": cls.unknown}
        layout = spec.pop("layout", None)
        if layout not in layouts:
            raise ValueError(f"electrode group {spec.get('name')}: layout must be one of {list(layouts)}")
        return layouts[layout](**spec)


class ElectrodeLayout:
    """Every electrode of a dataset in channel order, and named channel subsets (montages, e.g. ring8)."""

    def __init__(self, groups: list[ElectrodeGroup], montages: dict[str, list[int]] | None = None):
        self.groups = groups
        self.montages = montages or {}
        if len({g.name for g in groups}) != len(groups):
            raise ValueError(f"electrode group names repeat: {[g.name for g in groups]}")
        for name, channels in self.montages.items():
            if not all(0 <= c < len(self) for c in channels):
                raise ValueError(f"montage {name} has channels outside 0..{len(self) - 1}")

    def __len__(self) -> int:
        return sum(len(g) for g in self.groups)

    @property
    def kind(self) -> str:
        """bipolar, monopolar or mixed."""
        kinds = {g.kind for g in self.groups}
        return kinds.pop() if len(kinds) == 1 else "mixed"

    @classmethod
    def from_config(cls, groups: list[dict], montages: dict[str, str] | None = None) -> "ElectrodeLayout":
        """montages: name -> channel range, "first-last" (0-based, inclusive), e.g. ring8: 0-7."""
        ranges = {name: channel_range(text) for name, text in (montages or {}).items()}
        return cls([ElectrodeGroup.from_config(g) for g in groups], ranges)

    @classmethod
    def from_meta(cls, meta: dict) -> "ElectrodeLayout":
        """Back from a session's meta.json."""
        groups = []
        for name in dict.fromkeys(e["group"] for e in meta["electrodes"]):
            rows = [e for e in meta["electrodes"] if e["group"] == name]
            points = [
                (np.nan if e["angle"] is None else e["angle"], np.nan if e["z"] is None else e["z"]) for e in rows
            ]
            placement = meta.get("electrode_groups", {}).get(name, {})
            groups.append(ElectrodeGroup.from_points(name, rows[0]["kind"], points, **placement))
        return cls(groups, meta.get("montages"))

    def to_meta(self) -> dict:
        """The meta.json fields: one entry per channel, how well each group's placement is known, the montages."""
        electrodes = [
            {
                "angle": None if np.isnan(a) else round(a, 2),
                "z": None if np.isnan(z) else round(z, 4),
                "kind": g.kind,
                "group": g.name,
            }
            for g in self.groups
            for a, z in zip(g.angles, g.zs, strict=True)
        ]
        return {
            "electrode_type": self.kind,
            "electrodes": electrodes,
            "electrode_groups": {
                g.name: {"angle_sd": g.angle_sd, "z_sd": g.z_sd, "mirror": g.mirror, "layout": g.layout}
                for g in self.groups
            },
            "montages": self.montages,
        }

    def positions(self, layouts: tuple[str, ...] | None = None) -> np.ndarray:
        """(C, 3) float32: angle (deg), z, bipolar (0/1); NaN where unknown, and for the groups whose layout is not in
        `layouts` (all groups when None)."""
        rows = [
            (a, z, g.kind == "bipolar") if layouts is None or g.layout in layouts else (np.nan, np.nan, np.nan)
            for g in self.groups
            for a, z in zip(g.angles, g.zs, strict=True)
        ]
        return np.array(rows, np.float32).reshape(-1, 3)

    def place_randomly(self, rng: np.random.Generator) -> np.ndarray:
        """positions() with every group moved by a random shift within its uncertainty: the training augmentation
        for bands whose rotation (or direction) is not documented."""
        out = self.positions()
        start = 0
        for g in self.groups:
            rows = slice(start, start + len(g))
            start += len(g)
            if g.mirror and rng.random() < 0.5:
                out[rows, 0] = -out[rows, 0]
            out[rows, 0] += rng.uniform(0, 360) if g.angle_sd >= 180 else rng.normal(0, g.angle_sd)
            out[rows, 1] += rng.normal(0, g.z_sd)
        out[:, 0] %= 360
        return out


def channel_range(text: str | int) -> list[int]:
    """ "0-7" -> [0, 1, ..., 7]; "3" -> [3]."""
    first, _, last = str(text).partition("-")
    return list(range(int(first), int(last or first) + 1))


def distance_cm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance over the skin between positions (..., 2+) (angle deg, z), the forearm unrolled as a cylinder."""
    around = (a[..., 0] - b[..., 0] + 180) % 360 - 180
    return np.hypot(around / 360 * FOREARM_CIRCUMFERENCE_CM, (a[..., 1] - b[..., 1]) * FOREARM_LENGTH_CM)
