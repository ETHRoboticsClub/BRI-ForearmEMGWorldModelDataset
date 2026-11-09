"""Deterministic, nested subject subsets: the subjects kept at fraction 0.1 are also kept at 0.5 and 1.0."""

import hashlib
import math


def subject_rank(dataset: str, subject: str) -> int:
    return int.from_bytes(hashlib.sha1(f"{dataset}/{subject}".encode()).digest()[:8], "big")


def select_subjects(dataset: str, subjects: list[str], fraction: float) -> list[str]:
    """Keep ceil(fraction * n) subjects (at least one), chosen by a stable hash, returned in input order."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    n = max(1, math.ceil(fraction * len(subjects)))
    keep = set(sorted(subjects, key=lambda s: subject_rank(dataset, s))[:n])
    return [s for s in subjects if s in keep]
