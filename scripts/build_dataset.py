import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import yaml

from femg.datasets import DATASETS, first_percent
from femg.processor import PROCESSORS
from femg.save import is_saved, make_meta, save_session, session_folder, write_index

GB_PER_WORKER = 2.0  # peak RAM of one session (largest: NinaPro DB8 A1, a ~1 GB file)


def build_subject(dataset, subject, processor, raw_dir: Path, out_dir: Path, overwrite: bool) -> int:
    """Load, process and save every session of one subject. Returns the number of failed sessions."""
    n_failed = 0
    for session_name in dataset.sessions(subject):
        folder = session_folder(out_dir, dataset.name, subject, session_name)
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


def build_dataset(dataset, processor, percent, raw_dir: Path, out_dir: Path, workers: int, overwrite: bool) -> int:
    """Build the selected subjects of one dataset. Returns the number of failed sessions."""
    if dataset.effective_fs < processor.fs:
        print(f"[{dataset.name}] skipped: its effective rate {dataset.effective_fs} Hz is below fs={processor.fs}")
        return 0

    subjects = first_percent(dataset.subjects(), percent)
    ready = [s for s in subjects if dataset.is_downloaded(s, raw_dir)]
    missing = [s for s in subjects if s not in ready]
    print(f"[{dataset.name}] {len(ready)} subjects to build, {workers} workers")
    if missing:
        print(f"[{dataset.name}] not downloaded, skipped: {missing} (run scripts/pull_data.py)")

    jobs = [(dataset, subject, processor, raw_dir, out_dir, overwrite) for subject in ready]
    if workers == 1:
        failures = [build_subject(*job) for job in jobs]  # no subprocesses: easier to debug
    else:
        with Pool(workers) as pool:
            failures = pool.starmap(build_subject, jobs)

    index = write_index(out_dir / dataset.name)
    hours = index["duration_s"].sum() / 3600
    print(f"[{dataset.name}] {len(index)} sessions on disk, {hours:.1f} h of EMG, {sum(failures)} failed")
    return sum(failures)


def auto_workers() -> int:
    """As many workers as CPUs and free RAM allow."""
    cpus = os.cpu_count() or 1
    try:
        meminfo = Path("/proc/meminfo").read_text().splitlines()
        free_kb = next(int(line.split()[1]) for line in meminfo if line.startswith("MemAvailable"))
        free_gb = free_kb / 2**20
    except (OSError, StopIteration):  # not Linux
        free_gb = 8.0
    return max(1, min(cpus - 1, int(free_gb // GB_PER_WORKER)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--dataset", action="append", choices=list(DATASETS), help="repeatable; default: params")
    parser.add_argument("--percent", type=float, help="percentage of subjects per dataset; default: params")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--workers", type=int, help="default: workers in params ('auto' = from CPUs and RAM)")
    parser.add_argument("--overwrite", action="store_true", help="rebuild sessions that are already built")
    args = parser.parse_args()

    params = yaml.safe_load(Path(args.params).read_text())
    processor_settings = dict(params["processor"])
    processor_class = PROCESSORS[processor_settings.pop("name")]
    processor = processor_class(**processor_settings)

    percent = args.percent if args.percent is not None else params["percent"]
    workers = args.workers or params["workers"]
    if workers == "auto":
        workers = auto_workers()

    n_failed = 0
    for name in args.dataset or params["datasets"]:
        dataset = DATASETS[name]()
        raw_dir = Path(args.raw) / name
        n_failed += build_dataset(dataset, processor, percent, raw_dir, Path(args.out), workers, args.overwrite)

    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
