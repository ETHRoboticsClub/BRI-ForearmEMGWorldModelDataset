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
    meta = {
        "key": key,
        "dataset": dataset.name,
        "subject": f"{dataset.name}/{session.subject}",  # globally unique subject id
        "session": session.name,
        "label_source": "cue",  # default when the dataset has no kinematics; session.info overrides it
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


def write_index(dataset_folder: Path) -> pd.DataFrame:
    rows = []
    for meta_file in sorted(dataset_folder.glob("*/*/meta.json")):
        meta = json.loads(meta_file.read_text())
        rows.append({column: meta.get(column) for column in INDEX_COLUMNS})
    index = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    dataset_folder.mkdir(parents=True, exist_ok=True)
    index.to_parquet(dataset_folder / "sessions.parquet", index=False)
    return index
