from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from femg.electrodes import FOREARM_CIRCUMFERENCE_CM, FOREARM_LENGTH_CM, ElectrodeLayout, distance_cm

CONFIG = Path(__file__).parent / "config" / "virtual_array.yaml"


@dataclass(frozen=True)
class VirtualArray:
    """A fixed grid of virtual electrodes around the forearm (rows: heights, columns: angles) that the raw EMG of any
    electrode layout is resampled onto, sample by sample: (T, C) -> (T, H, A) at the same rate.

    It is one linear map, `matrix(positions, gains)` (H * A, C), so it runs as a single matrix product anywhere:
    1. monopolar channels lose their common average (common_average): the signal every electrode shares, which
       bipolar electrodes cancel by design;
    2. each virtual electrode is the weighted mean of the real channels near it, weighted by a Gaussian of their
       distance over the skin (sd width_around_cm around the forearm, width_along_cm along it);
    3. optional gains (H, A) scale each virtual channel to its own active level (1 / the Normalizer's active level of
       the virtual channel): a mean of independent signals (bipolar) is weaker than a mean of similar ones (monopolar).

    Input: EMG divided by each real channel's active level (meta emg_stats.active), and `inputs(layout)`: the
    positions of the electrode groups whose layout is in `layouts` (rings), NaN for the others (spots of uncertain
    position, unknown ones), which get no weight. A virtual electrode with no real one within reach_cm is not covered:
    its row is 0. rotation (deg) turns the real band: k * 360 / n_angles rolls the output by k columns. mirror
    reflects it first (angle -> -angle: a band whose channels run the other way round), so virtual column i becomes
    column -i; both together: `CalibBaseline.transform`."""

    n_angles: int
    heights: tuple[float, ...]
    width_around_cm: float
    width_along_cm: float
    reach_cm: float
    layouts: tuple[str, ...]
    common_average: bool

    @classmethod
    def load(cls, path: Path = CONFIG) -> "VirtualArray":
        spec = yaml.safe_load(Path(path).read_text())
        return cls(**{**spec, "heights": tuple(spec["heights"]), "layouts": tuple(spec["layouts"])})

    def settings(self) -> dict:
        return {**asdict(self), "heights": list(self.heights), "layouts": list(self.layouts)}

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.heights), self.n_angles

    @property
    def angles(self) -> np.ndarray:
        return np.arange(self.n_angles) * 360 / self.n_angles

    def positions(self) -> np.ndarray:
        """(H, A, 2): angle (deg), z of every virtual electrode."""
        angle, z = np.meshgrid(self.angles, self.heights)
        return np.stack([angle, z], axis=-1)

    @staticmethod
    def place(positions: np.ndarray, rotation: float = 0.0, mirror: bool = False) -> np.ndarray:
        """(C, 2): the real (angle, z), the band reflected (mirror) and then turned by rotation (deg)."""
        real = np.asarray(positions, np.float64)[:, :2].copy()
        if mirror:
            real[:, 0] = -real[:, 0]
        real[:, 0] += rotation
        return real

    def kernel(self, at: np.ndarray, positions: np.ndarray, rotation: float = 0.0, mirror: bool = False) -> np.ndarray:
        """(..., C): weight of each real channel (positions (C, 2+): angle, z) at the points `at` (..., 2)."""
        real = self.place(positions, rotation, mirror)
        at = np.asarray(at, np.float64)[..., None, :]
        around = ((real[:, 0] - at[..., 0] + 180) % 360 - 180) / 360 * FOREARM_CIRCUMFERENCE_CM
        along = (real[:, 1] - at[..., 1]) * FOREARM_LENGTH_CM
        squared = (around / self.width_around_cm) ** 2 + (along / self.width_along_cm) ** 2
        return np.nan_to_num(np.exp(-0.5 * squared))

    def weights(self, positions: np.ndarray, rotation: float = 0.0, mirror: bool = False) -> np.ndarray:
        """(H, A, C): the kernel at every virtual electrode, before normalising."""
        return self.kernel(self.positions(), positions, rotation, mirror)

    def inputs(self, layout: ElectrodeLayout) -> np.ndarray:
        """(C, 3): the positions the virtual electrodes are built from (NaN: groups of other layouts, unknown)."""
        return layout.positions(self.layouts)

    def covered(self, positions: np.ndarray, rotation: float = 0.0, mirror: bool = False) -> np.ndarray:
        """(H, A) True where a real electrode is within reach_cm over the skin."""
        real = self.place(positions, rotation, mirror)
        distance = distance_cm(self.positions()[..., None, :], real)
        return np.where(np.isfinite(distance), distance, np.inf).min(axis=-1) <= self.reach_cm

    def reference(self, positions: np.ndarray) -> np.ndarray:
        """(C, C): the identity, minus the common average over the monopolar channels of known position."""
        positions = np.asarray(positions)
        out = np.eye(len(positions))
        monopolar = np.isfinite(positions[:, :2]).all(axis=1) & (positions[:, 2] == 0)
        if self.common_average and monopolar.sum() > 1:
            out[np.ix_(monopolar, monopolar)] -= 1 / monopolar.sum()
        return out

    def matrix(
        self, positions: np.ndarray, gains: np.ndarray | None = None, rotation: float = 0.0, mirror: bool = False
    ) -> np.ndarray:
        """(H * A, C): virtual EMG = matrix @ real EMG, one sample (channels / active level) at a time."""
        w = self.weights(positions, rotation, mirror).reshape(-1, len(positions))
        total = w.sum(axis=1, keepdims=True)
        covered = self.covered(positions, rotation, mirror).reshape(-1, 1)
        w = np.where(covered & (total > 0), w / np.maximum(total, 1e-300), 0.0)
        if gains is not None:
            w = w * np.asarray(gains, np.float64).reshape(-1, 1)
        return w @ self.reference(positions)

    def __call__(self, emg, positions, gains=None, rotation: float = 0.0, mirror: bool = False) -> np.ndarray:
        """(T, C) EMG / active level -> (T, H, A) virtual EMG."""
        out = np.asarray(emg, np.float32) @ self.matrix(positions, gains, rotation, mirror).T.astype(np.float32)
        return out.reshape(len(out), *self.shape)
