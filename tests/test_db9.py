import dataclasses
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from femg.datasets import ninapro_db9 as db9
from femg.datasets.ninapro import NINAPRO_FLIPPED, NinaProDB2
from femg.hand.joints import CYBERGLOVE_22, JOINT_INDEX

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "tests"))
import check_db9  # noqa: E402

FS = 2000
NAMES = ["CMC1_F", "MCP1_F", "IP1_F", "CMC1_A", "MCP2_F", "PIP2_F", "DIP2_F", "MCP3_F", "PIP3_F", "DIP3_F", "MCP2-3_A",
         "MCP4_F", "PIP4_F", "DIP4_F", "MCP3-4_A", "MCP5_F", "PIP5_F", "DIP5_F", "MCP4-5_A", "P_Arch", "WRIST_F",
         "WRIST_D"]  # fmt: skip


def recording(n_seconds=40, seed=0):
    """Cues 5 (spread), 8 (together), 13/14 (wrist flex/ext), 15/16 (radial/ulnar), rest between; angles in degrees
    with canonical signs, and the raw glove the way NinaPro stores it (flipped sensors negated, gain and offset)."""
    rng = np.random.default_rng(seed)
    n = n_seconds * FS
    stimulus = np.zeros(n, np.int16)
    angles = rng.normal(0, 1, (n, 22))
    for i, cue in enumerate([5, 8, 13, 14, 15, 16]):
        a, b = (2 + 6 * i) * FS, (6 + 6 * i) * FS
        stimulus[a:b] = cue
    columns = {joint: c for c, joint in enumerate(CYBERGLOVE_22.values())}
    for joint, high, low in [("index_middle_abd", 5, 8), ("middle_ring_abd", 5, 8), ("ring_pinky_abd", 5, 8),
                             ("wrist_flex", 13, 14), ("wrist_dev", 15, 16)]:  # fmt: skip
        angles[stimulus == high, columns[joint]] += 20
        angles[stimulus == low, columns[joint]] -= 20
    angles[:, columns["index_mcp_flex"]] += 30 * np.sin(np.arange(n) / FS)
    signs = np.array([-1.0 if joint in NINAPRO_FLIPPED else 1.0 for joint in CYBERGLOVE_22.values()])
    glove = angles * signs * 1.7 + 40
    angles[:, 10] = 0  # sensor 11 left out of DB9
    return stimulus, angles, glove


def write_files(raw: Path, subject: str, ratio=1, offset=0):
    stimulus, angles, glove = recording()
    raw.mkdir(parents=True, exist_ok=True)
    emg = np.random.default_rng(1).normal(0, 1e-5, (len(stimulus), 12)).astype(np.float32)
    for exercise in ["E1", "E2"]:
        scipy.io.savemat(raw / f"S{int(subject[1:])}_{exercise}_A1.mat",
                         {"emg": emg, "glove": glove, "restimulus": stimulus[:, None]})  # fmt: skip
    n = db9.subject_number(subject)
    archive = raw / db9.remote_file(subject).name
    with zipfile.ZipFile(archive, "w") as z:
        for exercise in ["E1", "E2"]:  # names the repo does not assume: picked by content
            keep = slice(offset, None) if offset >= 0 else slice(None)
            mat = raw / f"tmp_{exercise}.mat"
            scipy.io.savemat(mat, {
                "angles": angles[::ratio][keep], "glove": glove[::ratio][keep],
                "restimulus": stimulus[::ratio][keep][:, None], "angle_names": np.array(NAMES, dtype=object),
            })  # fmt: skip
            z.write(mat, f"s_{n}/S{n}_{exercise}_A1_angles.mat")
            mat.unlink()


def db2_with_db9():
    dataset = NinaProDB2()
    dataset.config = dataclasses.replace(dataset.config, kin_source="db9")
    return dataset


@pytest.mark.parametrize("ratio", [1, 20])
def test_lines_up_and_loads(tmp_path, ratio):
    write_files(tmp_path, "s01", ratio=ratio)
    row = check_db9.check_session("s01", "E1", tmp_path)
    assert row["ratio"] == ratio and row["offset"] == 0 and row["match"] == 1.0
    assert row["glove_identical"] == 1.0 and row["names"] == "from file"
    assert row["joints_db2"] - row["joints_db9"] == {"index_middle_abd"}  # sensor 11
    assert all(r > 0.99 for r in row["r_glove"].values())  # the repo's flips agree with DB9's signs
    assert all(v > 0 for v in row["cue_signs"].values())

    session = db2_with_db9().load_session("s01", "E1", tmp_path)
    assert session.kin_fs == FS / ratio and session.info["kin_units"] == "rad"
    assert len(session.emg) == len(session.kin) * ratio
    assert np.isnan(session.kin[:, JOINT_INDEX["index_middle_abd"]]).all()
    assert np.nanmax(np.abs(session.kin[:, JOINT_INDEX["wrist_dev"]])) < np.deg2rad(40)


def test_shifted_file_is_refused(tmp_path):
    write_files(tmp_path, "s01", offset=3 * FS)
    row = check_db9.check_session("s01", "E1", tmp_path)
    assert row["offset"] == 3 * FS and row["match"] == 1.0  # found, and the labels agree once shifted
    with pytest.raises(ValueError, match="does not line up"):
        db2_with_db9().load_session("s01", "E1", tmp_path)


def test_names_missing_fall_back_to_glove_order():
    assert db9.column_joints(None) == list(CYBERGLOVE_22.values())
    assert db9.column_joints(NAMES) == list(CYBERGLOVE_22.values())  # the paper's names, in glove order
