import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from femg.calib_baseline import FOLDER, CalibBaseline
from femg.datasets import DATASETS, first_percent
from femg.gestures import families, to_shared, vocabulary
from femg.processor import Processor
from femg.sampling import sampling_table, session_seconds
from femg.save import (
    is_saved,
    make_meta,
    save_arrays,
    save_session,
    session_folder,
    write_gesture_table,
    write_index,
)

GB_PER_WORKER = 2.0  # peak RAM of one session (largest: NinaPro DB8 A1, a ~1 GB file)
STAGES = ("interim", "processed", "calibration", "sampling")
SAMPLING_FILE = "sampling.parquet"


def run(function, jobs: list[tuple], workers: int) -> list:
    if workers == 1:
        return [function(*job) for job in jobs]  # no subprocesses: easier to debug
    with Pool(workers) as pool:
        return pool.starmap(function, jobs)


def build_subject(dataset, subject, processor, raw_dir: Path, interim_dir: Path, overwrite: bool) -> int:
    n_failed = 0
    for session_name in dataset.sessions(subject):
        folder = session_folder(interim_dir, dataset.name, subject, session_name)
        if not overwrite and is_saved(folder, processor.settings()):
            print(f"  skip   {folder} (already built)", flush=True)
            continue
        try:
            start = time.time()
            session = dataset.load_session(subject, session_name, raw_dir)
            session = processor.process(session, dataset)
            save_session(folder, session, make_meta(dataset, session, processor))
            print(f"  built  {folder} in {time.time() - start:.0f} s", flush=True)
        except Exception as error:
            print(f"  FAILED {folder}: {error!r}", flush=True)
            n_failed += 1
    return n_failed


def build_interim(dataset, processor, subjects, raw_dir: Path, interim_dir: Path, workers: int, overwrite: bool) -> int:
    if dataset.config.effective_fs < processor.fs:
        rate = dataset.config.effective_fs
        print(f"[{dataset.name}] skipped: its effective rate {rate} Hz is below fs={processor.fs}")
        return 0
    ready = [s for s in subjects if dataset.is_downloaded(s, raw_dir)]
    missing = [s for s in subjects if s not in ready]
    print(f"[{dataset.name}] interim: {len(ready)} subjects to build, {workers} workers")
    if missing:
        print(f"[{dataset.name}] not downloaded, skipped: {missing} (run scripts/pull_data.py)")

    jobs = [(dataset, subject, processor, raw_dir, interim_dir, overwrite) for subject in ready]
    failures = run(build_subject, jobs, workers)
    index = write_index(interim_dir / dataset.name)
    hours = index["duration_s"].sum() / 3600
    print(f"[{dataset.name}] interim: {len(index)} sessions on disk, {hours:.1f} h of EMG, {sum(failures)} failed")
    return sum(failures)


def config_meta(cues: dict, device_group: str) -> dict:
    """The meta.json fields taken from the current configs, not the interim meta, so editing the gesture vocabulary, a
    mapping or a device group only redoes the processed stage: the shared gestures and the sampling group."""
    return {
        "stimulus_names": {str(cue): label for cue, (label, _) in cues.items()},
        "cue_gestures": {str(cue): gesture for cue, (_, gesture) in cues.items()},
        "gesture_names": list(vocabulary()),
        "gesture_families": list(families()),
        "device_group": device_group,
    }


def prepare_session(processor, source: Path, folder: Path, cues: dict, device_group: str, overwrite: bool) -> int:
    meta_file = folder / "meta.json"
    newer_source = meta_file.exists() and (source / "meta.json").stat().st_mtime > meta_file.stat().st_mtime
    from_config = config_meta(cues, device_group)
    same_config = meta_file.exists() and all(
        json.loads(meta_file.read_text()).get(k) == v for k, v in from_config.items()
    )
    if not overwrite and not newer_source and same_config and is_saved(folder, processor.prepare_settings()):
        print(f"  skip   {folder} (already prepared)", flush=True)
        return 0
    try:
        start = time.time()
        meta = json.loads((source / "meta.json").read_text())
        arrays, meta = processor.prepare(np.load(source / "emg.npy"), np.load(source / "kin.npy"), meta)
        stimulus = np.load(source / "stimulus.npy")
        gesture = to_shared(stimulus, cues)
        arrays |= {"stimulus": stimulus, "gesture": gesture}
        envelope = processor.normalizer.envelope(arrays["virtual"])
        seconds = session_seconds(envelope, gesture, np.asarray(meta["virtual_covered"], bool), meta["label_fs"])
        save_arrays(folder, arrays, meta | from_config | seconds)
        print(f"  built  {folder} in {time.time() - start:.0f} s", flush=True)
        return 0
    except Exception as error:
        print(f"  FAILED {folder}: {error!r}", flush=True)
        return 1


def build_processed(
    dataset, processor, subjects, interim_dir: Path, out_dir: Path, workers: int, overwrite: bool, split: dict
):
    sources = [m.parent for s in subjects for m in sorted((interim_dir / dataset.name / s).glob("*/meta.json"))]
    print(f"[{dataset.name}] processed: {len(sources)} interim sessions, {workers} workers")
    cues, group = dataset.config.cues, dataset.config.device_group
    jobs = [
        (processor, source, out_dir / source.relative_to(interim_dir), cues, group, overwrite) for source in sources
    ]
    failures = run(prepare_session, jobs, workers)
    index = write_index(out_dir / dataset.name, dataset.splits(split["test_percent"], split["seed"]))
    gestures = write_gesture_table(out_dir / dataset.name)
    n_test = (index["split"] == "test").sum()
    print(
        f"[{dataset.name}] processed: {len(index)} sessions on disk ({n_test} test), "
        f"{gestures['gesture_id'].nunique() if len(gestures) else 0} shared gestures, {sum(failures)} failed"
    )
    return sum(failures)


def build_calibration(names: list[str], out_dir: Path) -> None:
    """The calibration baseline over every processed session of `names` -> out_dir/calibration."""
    folders = [m.parent for name in names for m in sorted((out_dir / name).glob("*/*/meta.json"))]
    if not folders:
        print("[calibration] no processed sessions, skipped")
        return
    baseline = CalibBaseline.load().build(folders)
    baseline.save(out_dir / FOLDER)
    counts = ", ".join(f"{a} {n}" for a, n in baseline.counts.items())
    print(f"[calibration] {len(baseline.sessions)} of {len(folders)} sessions aligned ({counts}) -> {out_dir / FOLDER}")


def build_sampling(names: list[str], discrete: list[str], alpha: float, out_dir: Path) -> None:
    """out_dir/sampling.parquet: per processed session of `names`, how often to draw it for pretraining, discrete and
    continuous decoding (src/femg/sampling.py)."""
    tables = [pd.read_parquet(f) for name in names if (f := out_dir / name / "sessions.parquet").exists()]
    if not tables:
        print("[sampling] no processed sessions, skipped")
        return
    index = pd.concat(tables, ignore_index=True)
    groups = {name: DATASETS[name]().config.device_group for name in index["dataset"].unique()}
    index["device_group"] = index["device_group"].fillna(index["dataset"].map(groups))  # from the config if not built
    missing = index[index["active_s"].isna()]["dataset"].unique()
    if len(missing):
        print(f"[sampling] no seconds in {list(missing)} (processed before they were written): rebuild them")
    table = sampling_table(index, discrete, alpha)
    table.to_parquet(out_dir / SAMPLING_FILE, index=False)
    shares = table.groupby("dataset")[["p_pretrain", "p_discrete", "p_continuous"]].sum()
    hours = index.groupby("dataset")["duration_s"].sum() / 3600
    print(f"[sampling] alpha {alpha}, share of draws per dataset (train subjects) -> {out_dir / SAMPLING_FILE}")
    print(shares.assign(hours=hours).round(3).sort_values("hours", ascending=False).to_string())


def auto_workers() -> int:
    cpus = os.cpu_count() or 1
    try:
        meminfo = Path("/proc/meminfo").read_text().splitlines()
        free_kb = next(int(line.split()[1]) for line in meminfo if line.startswith("MemAvailable"))
        free_gb = free_kb / 2**20
    except (OSError, StopIteration):  # not Linux
        free_gb = 8.0
    return max(1, min(cpus - 1, int(free_gb // GB_PER_WORKER)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="data/raw -> data/interim/<dataset> -> data/processed/<dataset>, then the calibration baseline "
        "and the sampling table (resumable)."
    )
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--dataset", action="append", choices=list(DATASETS), help="repeatable; default: params")
    parser.add_argument("--stage", action="append", choices=STAGES, help="repeatable; default: all, in order")
    parser.add_argument("--percent", type=float, help="percentage of subjects per dataset; default: params")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--interim", default="data/interim")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--workers", type=int, help="default: workers in params ('auto' = from CPUs and RAM)")
    parser.add_argument("--overwrite", action="store_true", help="rebuild sessions that are already built")
    args = parser.parse_args()

    params = yaml.safe_load(Path(args.params).read_text())
    processor = Processor(**params["processor"])
    percent = args.percent if args.percent is not None else params["percent"]
    workers = args.workers or params["workers"]
    if workers == "auto":
        workers = auto_workers()
    stages = args.stage or STAGES
    interim_dir, out_dir = Path(args.interim), Path(args.out)
    not_built = sorted(set(params.get("discrete_datasets", [])) - set(params["datasets"]))
    if not_built:
        sys.exit(f"discrete_datasets must also be in datasets (the list that is built): {not_built}")
    split = params.get("split", {"test_percent": 0, "seed": 0})

    n_failed = 0
    for name in args.dataset or params["datasets"]:
        dataset = DATASETS[name]()
        subjects = first_percent(dataset.subjects(), percent)
        if "interim" in stages:
            raw_dir = Path(args.raw) / name
            n_failed += build_interim(dataset, processor, subjects, raw_dir, interim_dir, workers, args.overwrite)
        if "processed" in stages:
            n_failed += build_processed(
                dataset, processor, subjects, interim_dir, out_dir, workers, args.overwrite, split
            )
    if "calibration" in stages:
        build_calibration(params["datasets"], out_dir)  # over every dataset, not only the ones just built
    if "sampling" in stages:
        sampling = params.get("sampling", {"alpha": 0.5})
        build_sampling(params["datasets"], params.get("discrete_datasets", []), sampling["alpha"], out_dir)

    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
