"""PHASE 1: download the raw files of the datasets into data/raw/<dataset>/.

    python scripts/pull_data.py                                   # every dataset, every subject
    python scripts/pull_data.py --dataset emg_fk --percent 10     # the first 10% of the subjects
    python scripts/pull_data.py --dataset ninapro_db2 --dataset ninapro_db8 --percent 30

Files that are already downloaded are skipped, so you can run it again to resume or to grow the percentage.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from femg.datasets import DATASETS, first_percent
from femg.download import download_file, is_downloaded


def download_one(file, folder: Path) -> bool:
    """Download one file. Returns True if it worked."""
    try:
        download_file(file, folder)
        print(f"  done   {folder / file.name}", flush=True)
        return True
    except Exception as error:
        print(f"  FAILED {file.url}: {error!r}", flush=True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", choices=list(DATASETS), help="repeatable; default: all")
    parser.add_argument("--percent", type=float, default=100, help="percentage of subjects per dataset")
    parser.add_argument("--raw", default="data/raw", help="where to save the raw files")
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads")
    args = parser.parse_args()

    all_ok = True
    for name in args.dataset or list(DATASETS):
        dataset = DATASETS[name]()
        folder = Path(args.raw) / name
        subjects = first_percent(dataset.subjects(), args.percent)

        files = []
        for subject in subjects:
            files += dataset.files_to_download(subject)
        to_download = [f for f in files if not is_downloaded(f, folder)]
        n_subjects = f"{len(subjects)}/{len(dataset.subjects())}"
        print(f"[{name}] {n_subjects} subjects, {len(files)} files, {len(to_download)} to download")

        # Download with different threads
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(download_one, to_download, [folder] * len(to_download)))
        all_ok = all_ok and all(results)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
