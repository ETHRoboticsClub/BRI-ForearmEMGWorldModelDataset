import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from femg.hand.joints import CANONICAL_JOINTS

INDEX_COLUMNS = [
    "key",
    "dataset",
    "subject",
    "session",
    "condition",
    "hand",
    "label_source",
    "kin_units",
    "n_channels",
    "n_samples",
    "n_labels",
    "duration_s",
    "device",
]


def session_folder(out_dir: Path, dataset_name: str, subject: str, session: str) -> Path:
    return out_dir / dataset_name / subject / session


def is_saved(folder: Path, settings: dict) -> bool:
    meta_file = folder / "meta.json"
    if not meta_file.exists():
        return False
    return json.loads(meta_file.read_text()).get("processor") == settings


def make_meta(dataset, session, processor) -> dict:
    key = f"{dataset.name}/{session.subject}/{session.name}"
    if len(dataset.config.electrodes) != session.emg.shape[1]:
        raise ValueError(
            f"{key}: {session.emg.shape[1]} channels but {len(dataset.config.electrodes)} electrodes described"
        )
    measured = [j for j in range(len(CANONICAL_JOINTS)) if np.isfinite(session.kin[:, j]).any()]
    has_cues = session.stimulus is not None and bool((session.stimulus >= 0).any())
    meta = {
        "key": key,
        "dataset": dataset.name,
        "subject": f"{dataset.name}/{session.subject}",  # globally unique subject id
        "session": session.name,
        # default without kinematics (session.info overrides it): cue (instructed movements) or none (EMG only)
        "label_source": "cue" if has_cues else "none",
        "kin_units": None,
        **session.info,
        **dataset.config.metadata(),
        "n_channels": session.emg.shape[1],
        "n_samples": len(session.emg),
        "n_labels": len(session.kin),
        "duration_s": len(session.emg) / processor.fs,
        "fs": processor.fs,
        "label_fs": processor.label_fs,
        "joints": CANONICAL_JOINTS,
        "measured_joints": measured,
        # [start, end) label frames of each continuous recording; a frame whose kin row is all NaN has no label
        "segments": [[a, b] for a, b in zip(session.segments, [*session.segments[1:], len(session.kin)], strict=True)],
        "processor": processor.settings(),
    }
    if "hand" in session.info:  # a subject recorded on the other side than the dataset's (NinaPro DB3)
        meta["hand"] = session.info["hand"]
    return meta


def save_session(folder: Path, session, meta: dict) -> None:
    """An interim session: emg (T, C) volts, kin (T / r, J) rad, stimulus (T / r,), meta.json."""
    arrays = {"emg": session.emg, "kin": session.kin, "stimulus": session.stimulus.astype(np.int16)}
    save_arrays(folder, arrays, meta)


def save_arrays(folder: Path, arrays: dict[str, np.ndarray], meta: dict) -> None:
    """<name>.npy per array (float32 unless integer) and meta.json, written to a temporary folder renamed at the end:
    a crash never leaves half a session."""
    tmp = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name, array in arrays.items():
        array = np.asarray(array)
        np.save(tmp / f"{name}.npy", array if np.issubdtype(array.dtype, np.integer) else array.astype(np.float32))
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(folder, ignore_errors=True)
    tmp.rename(folder)


def write_index(dataset_folder: Path, splits: dict[str, str] | None = None) -> pd.DataFrame:
    """sessions.parquet: one row per session. splits: subject (as the dataset names it) -> train / test, a `split`
    column when given."""
    rows = []
    for meta_file in sorted(dataset_folder.glob("*/*/meta.json")):
        meta = json.loads(meta_file.read_text())
        rows.append({column: meta.get(column) for column in INDEX_COLUMNS})
    columns = INDEX_COLUMNS
    if splits is not None:
        columns = [*INDEX_COLUMNS, "split"]
        for row in rows:
            row["split"] = splits.get(row["subject"].split("/", 1)[1])
    index = pd.DataFrame(rows, columns=columns)
    dataset_folder.mkdir(parents=True, exist_ok=True)
    index.to_parquet(dataset_folder / "sessions.parquet", index=False)
    return index


GESTURE_COLUMNS = ["dataset", "subject", "session", "gesture_id", "gesture", "family", "seconds", "repetitions"]


def write_gesture_table(dataset_folder: Path) -> pd.DataFrame:
    """gestures.parquet: per session and shared gesture, its labelled time and repetitions (runs of it inside a
    segment), from the gesture.npy of every processed session."""
    rows = []
    for meta_file in sorted(dataset_folder.glob("*/*/meta.json")):
        gesture_file = meta_file.with_name("gesture.npy")
        if not gesture_file.exists():
            continue
        meta = json.loads(meta_file.read_text())
        gesture, names, family = np.load(gesture_file), meta["gesture_names"], meta["gesture_families"]
        counts: dict[int, list] = {}
        for start, end in meta["segments"]:
            g = gesture[start:end]
            runs = np.flatnonzero(np.diff(np.r_[-2, g]) != 0)  # first frame of every run
            for value in np.unique(g[g >= 0]):
                seconds, reps = counts.setdefault(int(value), [0.0, 0])
                counts[int(value)] = [seconds + (g == value).sum() / meta["label_fs"], reps + (g[runs] == value).sum()]
        for value, (seconds, reps) in sorted(counts.items()):
            row = [meta["dataset"], meta["subject"], meta["session"], value, names[value], family[value]]
            rows.append([*row, seconds, int(reps)])
    table = pd.DataFrame(rows, columns=GESTURE_COLUMNS)
    table.to_parquet(dataset_folder / "gestures.parquet", index=False)
    return table
