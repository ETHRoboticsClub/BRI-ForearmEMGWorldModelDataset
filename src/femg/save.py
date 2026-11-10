import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from femg.joints import CANONICAL_JOINTS

# Columns of sessions.parquet (all copied from meta.json).
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
    """True if the session is on disk and was built with the same processor settings."""
    meta_file = folder / "meta.json"
    if not meta_file.exists():
        return False
    return json.loads(meta_file.read_text()).get("processor") == settings


def make_meta(dataset, session, processor) -> dict:
    """Everything about a processed session except the arrays."""
    key = f"{dataset.name}/{session.subject}/{session.name}"
    measured = [j for j in range(len(CANONICAL_JOINTS)) if np.isfinite(session.kin[:, j]).any()]
    meta = {
        "key": key,
        "dataset": dataset.name,
        "subject": f"{dataset.name}/{session.subject}",  # globally unique subject id
        "session": session.name,
        "label_source": "cue",  # default when the dataset has no kinematics; session.info overrides it
        "kin_units": None,
        **session.info,
        **dataset.metadata(),
        "n_channels": session.emg.shape[1],
        "n_samples": len(session.emg),
        "n_labels": len(session.kin),
        "duration_s": len(session.emg) / processor.fs,
        "fs": processor.fs,
        "label_fs": processor.label_fs,
        "joints": CANONICAL_JOINTS,
        "measured_joints": measured,
        "processor": processor.settings(),
    }
    return meta


def save_session(folder: Path, session, meta: dict) -> None:
    """Write into a temporary folder first and rename it at the end, so a crash never leaves half a session."""
    tmp = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    np.save(tmp / "emg.npy", session.emg.astype(np.float32))
    np.save(tmp / "kin.npy", session.kin.astype(np.float32))
    np.save(tmp / "stimulus.npy", session.stimulus.astype(np.int16))
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(folder, ignore_errors=True)
    tmp.rename(folder)


def write_index(dataset_folder: Path) -> pd.DataFrame:
    """(Re)write <dataset>/sessions.parquet from the meta.json files on disk."""
    rows = []
    for meta_file in sorted(dataset_folder.glob("*/*/meta.json")):
        meta = json.loads(meta_file.read_text())
        rows.append({column: meta.get(column) for column in INDEX_COLUMNS})
    index = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    dataset_folder.mkdir(parents=True, exist_ok=True)
    index.to_parquet(dataset_folder / "sessions.parquet", index=False)
    return index
