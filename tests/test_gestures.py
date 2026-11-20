"""The shared gesture vocabulary, the dataset cue tables, label_source of cue-only / EMG-only sessions, the subject
split and the processed-stage gesture.npy, on synthetic sessions."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io
import yaml

from femg.datasets import DATASETS, Session
from femg.datasets.base import load_cues
from femg.datasets.ninapro import ABC_CUE_OFFSETS
from femg.gestures import NO_GESTURE, families, gesture_ids, to_shared, vocabulary
from femg.processor import Processor
from femg.save import make_meta, save_session, write_gesture_table, write_index

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import build_dataset  # noqa: E402

PARAMS = yaml.safe_load((Path(__file__).parents[1] / "params.yaml").read_text())
FS = 2000


def test_vocabulary():
    names = vocabulary()
    assert names[0] == "rest" and len(names) == len(set(names))
    assert len(families()) == len(names) and families()[0] == "rest"
    family = dict(zip(names, families(), strict=True))
    assert family["power"] == family["cylindrical"] == family["medium_wrap"] == "power_wrap"


@pytest.mark.parametrize("name", list(DATASETS))
def test_every_dataset_config_maps_into_the_vocabulary(name):
    cues = DATASETS[name]().config.cues  # load_cues raises on an unknown gesture
    assert 0 not in cues or cues[0][1] == "rest"


def test_unknown_gesture_is_refused():
    with pytest.raises(ValueError, match="not in gestures.yaml"):
        load_cues({1: ["something", "no_such_gesture"]}, "test")


def test_to_shared():
    cues = {1: ("thumb up", "thumb_up"), 2: ("not shared", None)}
    shared = to_shared(np.array([-1, 0, 1, 2, 3]), cues)
    thumb_up = vocabulary().index("thumb_up")
    assert shared.tolist() == [NO_GESTURE, 0, thumb_up, NO_GESTURE, NO_GESTURE]
    assert gesture_ids({0: ("open", "hand_open")})[0] == vocabulary().index("hand_open")


def test_ninapro_numbering():
    cues = DATASETS["ninapro_db2"]().config.cues
    assert sorted(cues) == list(range(1, 50))  # B 1-17, C 18-40, D 41-49
    assert cues[13][1] == "wrist_flex" and cues[44][1] == "index_flex"
    assert DATASETS["ninapro_db2"]().EXERCISES == ["E1", "E2", "E3"]


def test_discrete_datasets_are_built():
    assert set(PARAMS["discrete_datasets"]) <= set(PARAMS["datasets"])


def session(stimulus: np.ndarray | None, seconds=20):
    emg = np.random.default_rng(0).normal(0, 2e-5, (seconds * FS, 12)).astype(np.float32)
    return Session("s01", "E3", emg=emg, fs=FS, kin=None, kin_fs=FS, stimulus=stimulus)


def cues_every_2s(seconds=20):
    stimulus = np.zeros(seconds * FS, np.int16)
    for i, cue in enumerate([41, 44, 41, 44]):
        stimulus[(1 + 4 * i) * FS : (3 + 4 * i) * FS] = cue
    return stimulus


@pytest.mark.parametrize(("stimulus", "label_source"), [(cues_every_2s(), "cue"), (None, "none")])
def test_label_source_without_kinematics(stimulus, label_source):
    dataset, processor = DATASETS["ninapro_db2"](), Processor(**PARAMS["processor"])
    processed = processor.process(session(stimulus), dataset)
    meta = make_meta(dataset, processed, processor)
    assert meta["label_source"] == label_source and meta["measured_joints"] == []


def test_splits():
    dataset = DATASETS["ninapro_db2"]()
    splits = dataset.splits(20, 0)
    assert list(splits) == dataset.subjects() and sum(v == "test" for v in splits.values()) == 8
    assert splits == dataset.splits(20, 0) and splits != dataset.splits(20, 1)
    assert set(DATASETS["great"]().splits(0, 0).values()) == {"train"}


@pytest.mark.parametrize("name", ["ninapro_db7", "ninapro_db8"])
def test_splits_stratified_by_condition(name):
    dataset = DATASETS[name]()
    splits = dataset.splits(20, 0)
    amputees = [splits[s] for s in dataset.AMPUTEES]
    intact = [v for s, v in splits.items() if s not in dataset.AMPUTEES]
    assert sorted(amputees) == ["test", "train"]  # 20 % of 2, rounded up: one each
    assert intact.count("test") == -(-len(intact) * 20 // 100)


def test_processed_stage_writes_gestures(tmp_path):
    dataset, processor = DATASETS["ninapro_db2"](), Processor(**PARAMS["processor"])
    processed = processor.process(session(cues_every_2s()), dataset)
    interim = tmp_path / "interim" / "ninapro_db2" / "s01" / "E3"
    save_session(interim, processed, make_meta(dataset, processed, processor))
    out = tmp_path / "processed" / "ninapro_db2" / "s01" / "E3"
    assert build_dataset.prepare_session(processor, interim, out, dataset.config.cues, overwrite=False) == 0

    gesture, stimulus = np.load(out / "gesture.npy"), np.load(out / "stimulus.npy")
    meta = json.loads((out / "meta.json").read_text())
    assert np.array_equal(gesture, to_shared(stimulus, dataset.config.cues))
    names = meta["gesture_names"]
    assert {names[g] for g in np.unique(gesture)} == {"rest", "little_flex", "index_flex"}
    assert meta["stimulus_names"]["41"] == "force, flexion of the little finger"

    table = write_gesture_table(out.parents[1])
    little = table[table["gesture"] == "little_flex"].iloc[0]
    assert little["family"] == "single_digit" and meta["gesture_families"][0] == "rest"
    assert little["repetitions"] == 2 and abs(little["seconds"] - 4) < 0.1

    index = write_index(out.parents[1], dataset.splits(20, 0))
    assert index["split"].iloc[0] == dataset.splits(20, 0)["s01"]
    assert pd.read_parquet(out.parents[1] / "sessions.parquet")["label_source"].iloc[0] == "cue"


def test_ninapro_abc_matches_bcd():
    abc, bcd = DATASETS["ninapro_db5"]().config.cues, DATASETS["ninapro_db2"]().config.cues
    assert sorted(abc) == list(range(1, 53))
    assert all(abc[cue + 12] == bcd[cue] for cue in range(1, 41))  # exercises B and C, 12 ids up
    assert ABC_CUE_OFFSETS == {"E1": 0, "E2": 12, "E3": 29}  # A 1-12, B 13-29, C 30-52


def write_ninapro(path: Path, emg: np.ndarray, stimulus: np.ndarray, glove: bool = True):
    data = {"emg": emg, "restimulus": stimulus[:, None]}
    if glove:
        data["glove"] = np.random.default_rng(2).normal(0, 1, (len(emg), 22))
    scipy.io.savemat(path, data)


def test_db3_cues_only_and_flat_channels_refused(tmp_path):
    emg = np.random.default_rng(0).normal(0, 2e-5, (10 * FS, 12)).astype(np.float32)
    write_ninapro(tmp_path / "S2_E1_A1.mat", emg, cues_every_2s(10))
    session = DATASETS["ninapro_db3"]().load_session("s02", "E1", tmp_path)
    assert session.kin is None and session.info["hand"] == "left" and session.info["label_source"] == "cue"

    emg[:, [8, 9]] = 0  # s07: the forearm spots flat
    write_ninapro(tmp_path / "S2_E1_A1.mat", emg, cues_every_2s(10))
    with pytest.raises(ValueError, match="flat"):
        DATASETS["ninapro_db3"]().load_session("s02", "E1", tmp_path)
    assert "s07" not in DATASETS["ninapro_db3"]().subjects()


def test_db4_numbered_across_exercises(tmp_path):
    emg = np.random.default_rng(0).normal(0, 50, (10 * FS, 12)).astype(np.float32)
    stimulus = np.zeros(10 * FS, np.int16)
    stimulus[FS : 3 * FS] = 1  # exercise C, cue 1: large diameter
    write_ninapro(tmp_path / "S1_E3_A1.mat", emg, stimulus, glove=False)
    session = DATASETS["ninapro_db4"]().load_session("s01", "E3", tmp_path)
    assert set(np.unique(session.stimulus)) == {0, 30} and session.kin is None
    assert DATASETS["ninapro_db4"]().config.cues[30][1] == "large_diameter"


def test_ring_with_uneven_angles():
    from femg.electrodes import ElectrodeGroup

    group = ElectrodeGroup.from_config(
        {"name": "rows", "kind": "bipolar", "layout": "ring", "n": 4, "z": [0.1, 0.5], "angles": [345, 40, 180, 250]}
    )
    assert group.layout == "ring" and group.angles == (345, 40, 180, 250) * 2 and group.zs == (0.1,) * 4 + (0.5,) * 4
    with pytest.raises(ValueError, match="angles"):
        ElectrodeGroup.ring("rows", "bipolar", 3, 0.1, angles=[0, 90])


def test_fors_trials(tmp_path):
    from femg.datasets.fors_emg import GESTURES, ROOT

    fs = 985
    folder = tmp_path / ROOT / "Subject1" / "Rest"  # the case of the folder varies in the files
    folder.mkdir(parents=True)
    for gesture in GESTURES:
        for trial in range(1, 6):
            scipy.io.savemat(folder / f"{gesture}-{trial}.mat", {"value": np.full((8, 8000), 0.01)})
    session = DATASETS["fors_emg"]().load_session("s01", "neutral", tmp_path)
    assert len(session.segments) == 60 and session.segments[1] == 8000 and session.kin is None
    first = session.stimulus[:8000]
    assert (
        (first[: round(0.5 * fs)] == 0).all() and (first[round(0.5 * fs) : fs] == -1).all() and (first[fs:] == 1).all()
    )
    assert session.stimulus[-1] == 12 and session.emg.max() == pytest.approx(1e-5)  # mV -> V


def test_grabmyo_trials(tmp_path):
    import wfdb

    from femg.datasets.grabmyo import CHANNELS, GESTURES, TRIALS, record_name

    signal = np.zeros((200, 32))
    signal[:, CHANNELS] = 0.02  # mV
    for gesture in GESTURES:
        for trial in TRIALS:
            wfdb.wrsamp(record_name(2, "s05", gesture, trial), fs=2048, units=["mV"] * 32,
                        sig_name=[f"c{i}" for i in range(32)], p_signal=signal, fmt=["16"] * 32,
                        write_dir=str(tmp_path))  # fmt: skip
    session = DATASETS["grabmyo"]().load_session("s05", "day2", tmp_path)
    assert session.emg.shape == (17 * 7 * 200, 28) and len(session.segments) == 119
    assert session.emg.max() == pytest.approx(2e-5, rel=1e-3)  # mV -> V, unused columns dropped
    assert session.stimulus[0] == 1 and session.stimulus[-1] == 0  # rest (17) last, stored as 0
    assert session.info["day"] == 2


def test_epn612_labels_and_split(tmp_path):
    from femg.datasets.emg_epn612 import file_name

    n = 1000
    truth = [0] * 300 + [1] * 200 + [0] * 500
    emg = {f"ch{i}": [1] * n for i in range(1, 9)}

    def samples(labelled):
        out = {}
        for k, name in enumerate(["noGesture", "waveIn"], 1):
            s = {"emg": emg}
            if labelled:
                s |= {"gestureName": name, **({"groundTruth": truth} if name != "noGesture" else {})}
            out[f"idx_{k}"] = s
        return out

    data = {"generalInfo": {"samplingFrequencyInHertz": 200}, "userInfo": {"ArmDamage": "False"},
            "trainingSamples": samples(True), "testingSamples": samples(False)}  # fmt: skip
    (tmp_path / file_name("te007")).write_text(json.dumps(data))
    dataset = DATASETS["emg_epn612"]()
    labelled = dataset.load_session("te007", "samples1", tmp_path)
    assert labelled.info["condition"] == "intact" and labelled.info["label_source"] == "cue"
    assert (labelled.stimulus[:n] == 0).all() and labelled.stimulus[n + 300 : n + 500].tolist() == [2] * 200
    assert (labelled.stimulus[n : n + 300] == 0).all() and labelled.segments == [0, n]
    unlabelled = dataset.load_session("te007", "samples2", tmp_path)
    assert (unlabelled.stimulus == -1).all() and unlabelled.info["label_source"] == "none"
    splits = dataset.splits(20, 0)
    assert splits["te007"] == "test" and splits["tr007"] == "train" and len(splits) == 612


def test_putemg_record(tmp_path, monkeypatch):
    import h5py

    from femg.datasets import putemg

    names = [
        "emg_gestures-07-repeats_long-2018-05-01-10-00-00-000",
        "emg_gestures-07-repeats_long-2018-05-09-10-00-00-000",
    ]
    records = tmp_path / "records.txt"
    records.write_text("# header\n" + "\n".join(reversed(names)) + "\n")
    monkeypatch.setattr(putemg, "RECORDS", records)
    values = np.zeros((5120, 29))
    values[:, :24] = 100  # raw ADC
    values[:, 28] = np.repeat([0, 6, -1, 9], 1280)
    with h5py.File(tmp_path / f"{names[1]}.hdf5", "w") as f:
        f.create_dataset("data/table", data=np.array([(v,) for v in values], dtype=[("values_block_3", "<f8", (29,))]))
    dataset = DATASETS["putemg"]()
    assert dataset.subjects() == ["s07"] and dataset.sessions("s07") == ["repeats_long_1", "repeats_long_2"]
    session = dataset.load_session("s07", "repeats_long_2", tmp_path)  # the later date
    assert session.emg.shape == (5120, 24) and session.emg[0, 0] == pytest.approx(100 * 6.1035e-6)
    assert np.unique(session.stimulus).tolist() == [-1, 0, 6, 9] and session.info["experiment"] == 2


def test_capgmyo_ring_order(tmp_path):
    from femg.datasets.capgmyo import GESTURES, TRIALS, mat_name

    file_channels = np.tile(np.arange(128, dtype=float), (1000, 1))  # each channel holds its own file index
    for gesture in GESTURES:
        for trial in TRIALS:
            scipy.io.savemat(tmp_path / mat_name("s02", gesture, trial), {"data": file_channels})
    session = DATASETS["capgmyo_dba"]().load_session("s02", "dba", tmp_path)
    first = session.emg[0] / 1e-3  # mV -> V undone
    assert first[:3].tolist() == [0, 8, 16]  # ring 1: the first electrode of columns 1, 2, 3
    assert first[16] == 1  # ring 2 starts with the second electrode of column 1
    assert len(session.segments) == 80 and set(np.unique(session.stimulus)) == set(range(1, 9))
