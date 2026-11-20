import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

from femg.calib_baseline import FOLDER, CalibBaseline
from femg.datasets import DATASETS, first_percent
from femg.gestures import families, to_shared, vocabulary
from femg.processor import Processor
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
STAGES = ("interim", "processed", "calibration")


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


def gesture_meta(cues: dict) -> dict:
    """The meta.json fields of the shared gestures: the vocabulary and where the dataset's cues go in it (taken from the
    current configs, not the interim meta, so a vocabulary or mapping edit only redoes the processed stage)."""
    return {
        "stimulus_names": {str(cue): label for cue, (label, _) in cues.items()},
        "cue_gestures": {str(cue): gesture for cue, (_, gesture) in cues.items()},
        "gesture_names": list(vocabulary()),
        "gesture_families": list(families()),
    }


def prepare_session(processor, source: Path, folder: Path, cues: dict, overwrite: bool) -> int:
    meta_file = folder / "meta.json"
    newer_source = meta_file.exists() and (source / "meta.json").stat().st_mtime > meta_file.stat().st_mtime
    gestures = gesture_meta(cues)
    same_gestures = meta_file.exists() and all(
        json.loads(meta_file.read_text()).get(k) == v for k, v in gestures.items()
    )
    if not overwrite and not newer_source and same_gestures and is_saved(folder, processor.prepare_settings()):
        print(f"  skip   {folder} (already prepared)", flush=True)
        return 0
    try:
        start = time.time()
        meta = json.loads((source / "meta.json").read_text())
        arrays, meta = processor.prepare(np.load(source / "emg.npy"), np.load(source / "kin.npy"), meta)
        stimulus = np.load(source / "stimulus.npy")
        arrays |= {"stimulus": stimulus, "gesture": to_shared(stimulus, cues)}
        save_arrays(folder, arrays, meta | gestures)
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
    cues = dataset.config.cues
    jobs = [(processor, source, out_dir / source.relative_to(interim_dir), cues, overwrite) for source in sources]
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
        "(resumable)."
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

    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
