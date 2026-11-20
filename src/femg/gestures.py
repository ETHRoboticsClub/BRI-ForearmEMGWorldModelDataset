"""The shared gesture vocabulary (config/gestures.yaml) and the mapping of a dataset's cues onto it."""

from functools import cache
from pathlib import Path

import numpy as np
import yaml

VOCABULARY_FILE = Path(__file__).parent / "config" / "gestures.yaml"
NO_GESTURE = -1  # no cue, or a cue that maps to no shared gesture


@cache
def _load() -> dict:
    spec = yaml.safe_load(VOCABULARY_FILE.read_text())
    names = tuple(spec["gestures"])
    if names[0] != "rest":
        raise ValueError(f"{VOCABULARY_FILE}: the first gesture must be rest")
    families = tuple(family for family, _ in spec["gestures"].values())
    unknown = sorted(set(families) - set(spec["families"]))
    if unknown:
        raise ValueError(f"{VOCABULARY_FILE}: families not declared under families: {unknown}")
    return {"names": names, "families": families}


def vocabulary() -> tuple[str, ...]:
    """The shared gesture names; a gesture's id is its position (0 = rest)."""
    return _load()["names"]


def families() -> tuple[str, ...]:
    """The family (coarse class) of every gesture, in vocabulary order."""
    return _load()["families"]


def check_cues(cues: dict[int, tuple[str, str | None]], where: str) -> None:
    """cues: native cue id -> (its name in the dataset, shared gesture or None). Every shared name must exist."""
    known = set(vocabulary())
    unknown = sorted({g for _, g in cues.values() if g is not None and g not in known})
    if unknown:
        raise ValueError(f"{where}: cues map to gestures not in {VOCABULARY_FILE.name}: {unknown}")


def gesture_ids(cues: dict[int, tuple[str, str | None]]) -> dict[int, int]:
    """Native cue id -> shared gesture id, for the cues that map to one. Cue 0 is rest unless the dataset says
    otherwise."""
    ids = {name: i for i, name in enumerate(vocabulary())}
    out = {0: 0}
    for cue, (_, gesture) in cues.items():
        if gesture is None:
            out.pop(cue, None)
        else:
            out[cue] = ids[gesture]
    return out


def to_shared(stimulus: np.ndarray, cues: dict[int, tuple[str, str | None]]) -> np.ndarray:
    """(T,) native cue ids (-1: no cue) -> (T,) int16 shared gesture ids (NO_GESTURE where none)."""
    stimulus = np.asarray(stimulus)
    out = np.full(stimulus.shape, NO_GESTURE, np.int16)
    for cue, gesture in gesture_ids(cues).items():
        out[stimulus == cue] = gesture
    return out
