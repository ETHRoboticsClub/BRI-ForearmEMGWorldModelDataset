"""Raw downloads and the on-disk layout of built sessions.

Built layout (one directory per session, written atomically):

    <out>/<dataset>/<subject>/<session>/
        emg.npy        (T, C)          float32, volts, at build.fs
        kin.npy        (T // r, J)     float32, canonical joints (schema.CANONICAL_JOINTS), NaN = not measured
        stimulus.npy   (T // r,)       int16, instructed movement (0 = rest, -1 = unknown)
        meta.json      everything else (subject info, electrode layout, stats, build params)
    <out>/<dataset>/sessions.parquet   one row per session (the index the loader reads)

with r = build.fs / build.label_fs. Label k describes time k / label_fs, i.e. EMG sample k * r.
"""

import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from femg.schema import RemoteFile

CHUNK = 8 << 20
USER_AGENT = "femg-dataset-builder"


def download(f: RemoteFile, raw_dir: Path, retries: int = 3) -> None:
    """Download f into raw_dir (skipped when present), extracting `f.extract` members if f is a zip."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if f.extract:
        if all((raw_dir / m).exists() for m in f.extract):
            return
    elif (raw_dir / f.name).exists():
        return
    target = raw_dir / f.name
    part = target.with_name(target.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            _fetch(f.url, part, f.md5)
            break
        except Exception:
            part.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(5 * attempt)
    if f.extract:
        _extract(part, raw_dir, f.extract)
        part.unlink()
    else:
        part.rename(target)


def _fetch(url: str, dest: Path, md5: str | None) -> None:
    h = hashlib.md5() if md5 else None
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as out:
        expected = r.headers.get("Content-Length")
        while chunk := r.read(CHUNK):
            out.write(chunk)
            if h:
                h.update(chunk)
    if expected is not None and dest.stat().st_size != int(expected):
        raise OSError(f"{url}: truncated download ({dest.stat().st_size} of {expected} bytes)")
    if h and h.hexdigest() != md5:
        raise OSError(f"{url}: md5 mismatch")


def _extract(archive: Path, raw_dir: Path, members: tuple[str, ...]) -> None:
    """Extract members (matched by basename, archives nest files in varying folders) flat into raw_dir."""
    wanted = set(members)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if name in wanted:
                with z.open(info) as src, open(raw_dir / (name + ".part"), "wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
                (raw_dir / (name + ".part")).rename(raw_dir / name)
                wanted.discard(name)
    if wanted:
        raise FileNotFoundError(f"{archive.name} lacks {sorted(wanted)}")


def session_dir(out: Path, key: str) -> Path:
    return out / key


def is_built(path: Path, build_hash: str) -> bool:
    meta = path / "meta.json"
    return meta.exists() and json.loads(meta.read_text()).get("build_hash") == build_hash


def write_session(path: Path, emg: np.ndarray, kin: np.ndarray, stimulus: np.ndarray, meta: dict) -> None:
    """Write into a temporary sibling directory and rename, so an interrupted build never leaves half a session."""
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    np.save(tmp / "emg.npy", np.ascontiguousarray(emg, np.float32))
    np.save(tmp / "kin.npy", np.ascontiguousarray(kin, np.float32))
    np.save(tmp / "stimulus.npy", stimulus.astype(np.int16))
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def read_meta(path: Path) -> dict:
    return json.loads((path / "meta.json").read_text())


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


def write_index(dataset_dir: Path) -> pd.DataFrame:
    """Rebuild <dataset>/sessions.parquet from the meta.json files on disk."""
    rows = []
    for meta_path in sorted(dataset_dir.glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text())
        rows.append({c: meta.get(c) for c in INDEX_COLUMNS})
    df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    df.to_parquet(dataset_dir / "sessions.parquet", index=False)
    return df
