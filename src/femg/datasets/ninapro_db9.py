import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import correlate

from femg.download import RemoteFile
from femg.hand.joints import CYBERGLOVE_22, JOINT_INDEX, N_JOINTS

URL = "https://ninapro.hevs.ch/files/DB9/DB2/s_{n}_angles.zip"
SUBJECT_OFFSET = 27  # DB9 subject = DB2 subject + 27

# DB9 angle names (paper, Fig. 4 and Table 6) -> canonical joints
ANGLES = {
    "CMC1_F": "thumb_cmc_flex",
    "MCP1_F": "thumb_mcp_flex",
    "IP1_F": "thumb_ip_flex",
    "CMC1_A": "thumb_cmc_abd",
    **{f"MCP{d}_F": f"{f}_mcp_flex" for d, f in [(2, "index"), (3, "middle"), (4, "ring"), (5, "pinky")]},
    **{f"PIP{d}_F": f"{f}_pip_flex" for d, f in [(2, "index"), (3, "middle"), (4, "ring"), (5, "pinky")]},
    **{f"DIP{d}_F": f"{f}_dip_flex" for d, f in [(2, "index"), (3, "middle"), (4, "ring"), (5, "pinky")]},
    "MCP2-3_A": "index_middle_abd",
    "MCP3-4_A": "middle_ring_abd",
    "MCP4-5_A": "ring_pinky_abd",
    "P_ARCH": "palm_arch",
    "CMC5_F": "palm_arch",
    "WRIST_F": "wrist_flex",
    "WRIST_D": "wrist_dev",
}
# When a file holds no names: the columns in the glove's sensor order (sensor 11 of the CyberGlove 22, MI_Abduction, is
# the MCP2-3_A the page says was left out, which supports this order; scripts/tests/check_db9.py checks it)
GLOVE_ORDER = list(CYBERGLOVE_22.values())


def subject_number(db2_subject: str) -> int:
    return int(db2_subject[1:]) + SUBJECT_OFFSET


def remote_file(db2_subject: str) -> RemoteFile:
    n = subject_number(db2_subject)
    return RemoteFile(url=URL.format(n=n), name=f"DB9_s{n}_angles.zip")


def mat_files(db2_subject: str, raw_dir: Path) -> list[Path]:
    """The .mat files of the subject's DB9 zip, extracted once next to it."""
    archive = raw_dir / remote_file(db2_subject).name
    folder = raw_dir / archive.stem
    if not folder.exists():
        if not archive.exists():
            raise FileNotFoundError(f"{archive} (run scripts/tests/check_db9.py --download, or pull_data.py)")
        tmp = folder.with_name(folder.name + ".part")
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                if entry.filename.lower().endswith(".mat"):
                    tmp.mkdir(parents=True, exist_ok=True)
                    (tmp / Path(entry.filename).name).write_bytes(z.read(entry))
        tmp.rename(folder)
    return sorted(folder.glob("*.mat"))


def read(path: Path) -> dict:
    """angles (T, 22) as in the file, glove (T, 22) raw, restimulus (T,), names (22,) or None."""
    import scipy.io

    try:
        data = scipy.io.loadmat(path)
    except NotImplementedError:  # v7.3
        import h5py

        with h5py.File(path, "r") as f:
            data = {k: np.asarray(f[k]).T for k in f.keys() if isinstance(f[k], h5py.Dataset)}
    out = {"names": find_names(data)}
    for key, options in [("angles", ["angles"]), ("glove", ["glove"]), ("restimulus", ["restimulus", "stimulus"])]:
        found = next((data[k] for k in options if k in data), None)
        out[key] = None if found is None else np.asarray(found)
    if out["restimulus"] is not None:
        out["restimulus"] = out["restimulus"].reshape(-1).astype(np.int16)
    return out


def find_names(data: dict) -> list[str] | None:
    """The 22 column names, whatever the variable is called: the first cell or char array of 22 strings."""
    for key, value in data.items():
        if key.startswith("__"):
            continue
        array = np.asarray(value)
        if array.dtype.kind not in "OU" or array.size != 22:
            continue
        names = [str(np.asarray(v).reshape(-1)[0]) if np.asarray(v).size else "" for v in array.reshape(-1)]
        if sum(normalize_name(n) in ANGLES for n in names) >= 15:
            return names
    return None


def normalize_name(name: str) -> str:
    return name.strip().upper().replace(" ", "").replace("–", "-").replace("_A_", "_A").replace("PARCH", "P_ARCH")


def column_joints(names: list[str] | None) -> list[str | None]:
    """The canonical joint of each angle column (None: unknown name)."""
    if names is None:
        return GLOVE_ORDER
    return [ANGLES.get(normalize_name(n)) for n in names]


def to_canonical(angles: np.ndarray, names: list[str] | None) -> np.ndarray:
    """(T, 22) degrees -> (T, N_JOINTS) radians; a column that is all zero or constant (sensor 11) stays NaN."""
    kin = np.full((len(angles), N_JOINTS), np.nan, np.float32)
    for column, joint in enumerate(column_joints(names)):
        values = angles[:, column].astype(np.float32)
        if joint is None or not np.isfinite(values).any() or np.nanstd(values) < 1e-6:
            continue
        kin[:, JOINT_INDEX[joint]] = np.deg2rad(values)
    return kin


def align(db2_stimulus: np.ndarray, db9_stimulus: np.ndarray, max_shift_s: float = 30.0, fs: float = 2000) -> dict:
    """Where DB9's samples sit on DB2's. ratio: DB2 samples per DB9 sample (1 if DB9 kept DB2's 2 kHz). offset: DB9
    sample k is DB2 sample (k + offset) * ratio. match: fraction of the overlap where the restimulus labels agree."""
    n2, n9 = len(db2_stimulus), len(db9_stimulus)
    ratio = max(1, round(n2 / n9)) if n9 else 1
    a = db2_stimulus[::ratio].astype(np.int16)
    b = db9_stimulus.astype(np.int16)
    if len(a) == len(b) and np.array_equal(a, b):
        offset = 0
    else:
        reach = int(max_shift_s * fs / ratio)
        moving_a, moving_b = (a > 0).astype(np.float32), (b > 0).astype(np.float32)
        corr = correlate(moving_a - moving_a.mean(), moving_b - moving_b.mean(), mode="full", method="fft")
        lags = np.arange(-len(b) + 1, len(a))
        window = np.abs(lags) <= reach
        offset = int(lags[window][np.argmax(corr[window])])
    start_a, start_b = max(offset, 0), max(-offset, 0)
    n = min(len(a) - start_a, len(b) - start_b)
    match = float((a[start_a : start_a + n] == b[start_b : start_b + n]).mean()) if n > 0 else 0.0
    return {
        "ratio": ratio,
        "offset": offset,
        "match": match,
        "overlap": n / max(len(b), 1),
        "db2_samples": n2,
        "db9_samples": n9,
    }


def pick(db2_stimulus: np.ndarray, candidates: list[Path], exercise: str) -> tuple[Path, dict, dict]:
    """The DB9 file of a DB2 session: the one whose restimulus matches best (names first, then content)."""
    named = [p for p in candidates if f"E{exercise[-1]}" in p.stem.upper().replace("_", "")]
    best = None
    for path in named + [p for p in candidates if p not in named]:
        data = read(path)
        if data["restimulus"] is None or data["angles"] is None:
            continue
        alignment = align(db2_stimulus, data["restimulus"])
        if best is None or alignment["match"] > best[1]["match"]:
            best = (path, alignment, data)
        if alignment["match"] > 0.999 and alignment["overlap"] > 0.999:
            break
    if best is None:
        raise FileNotFoundError(f"no DB9 file with angles and restimulus among {[p.name for p in candidates]}")
    return best
