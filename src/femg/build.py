"""Build the harmonized dataset: download -> read -> resample/filter -> canonical labels -> write.

    femg-build                                   # every dataset in params.yaml, fraction from params.yaml
    femg-build --dataset emg_fk --fraction 0.1   # quick dev build of 10% of the subjects
    femg-build --out data/dev --fraction 0.2     # keep dev builds out of the DVC-tracked data/built

Downloads run in a thread pool (I/O bound) and feed a process pool (CPU bound), so the next subject downloads
while the previous one is processed. Sessions already built with the same parameters and code are skipped, so an
interrupted build resumes and raising the fraction only builds the new subjects.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import yaml

from femg import store
from femg.schema import CANONICAL_JOINTS, N_JOINTS, SessionRef
from femg.signal import Filter, decimate_labels, emg_stats, kin_stats, resample, resample_nan
from femg.sources import get_source
from femg.subset import select_subjects

FORMAT_VERSION = 1
GB_PER_WORKER = 2.0  # peak RAM of one session (largest: NinaPro DB8 A1, ~1 GB file)


@dataclass(frozen=True)
class BuildConfig:
    fs: float = 500
    label_fs: float = 100
    highpass: float | None = 10
    notch: bool = True
    calib_s: float = 30

    def __post_init__(self):
        if self.fs % self.label_fs:
            raise ValueError(f"fs ({self.fs}) must be a multiple of label_fs ({self.label_fs})")

    @property
    def ratio(self) -> int:
        return int(self.fs // self.label_fs)


def code_hash() -> str:
    """Hash of the package source: a code change invalidates previously built sessions."""
    h = hashlib.sha1()
    for p in sorted(Path(__file__).parent.rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def build_hash(cfg: BuildConfig, code: str) -> str:
    payload = json.dumps({"cfg": asdict(cfg), "code": code, "format": FORMAT_VERSION}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def process_session(ref: SessionRef, raw_dir: Path, out: Path, cfg: BuildConfig, bhash: str) -> dict:
    """Runs in a worker process. Returns a small status dict (arrays never travel back to the parent)."""
    t0 = time.perf_counter()
    src = get_source(ref.dataset)
    if cfg.fs > src.effective_fs:
        return {"key": ref.key, "status": "skipped", "reason": f"effective fs {src.effective_fs} < {cfg.fs}"}
    rec = src.read(ref, raw_dir)

    # EMG: resample first (filters then run on fewer samples), then causal high-pass + notches.
    emg = resample(rec.emg, rec.fs, cfg.fs)
    rec.emg = None  # free the native-rate copy
    emg = Filter(cfg.fs, cfg.highpass, src.mains_hz if cfg.notch else None)(emg)

    r = cfg.ratio
    if rec.kin is not None:
        kin = resample_nan(rec.kin, rec.kin_fs, cfg.label_fs)
    else:
        kin = np.full((len(emg) // r, N_JOINTS), np.nan, np.float32)
    if rec.stimulus is not None:
        stim = decimate_labels(rec.stimulus, rec.kin_fs, cfg.label_fs)
    else:
        stim = np.full(len(kin), -1, np.int16)

    # Lag: kin[k] describes the hand at k / label_fs + lag. Crop so label k matches EMG sample k * r.
    lag = int(round(src.kin_lag_s * cfg.label_fs))
    if lag > 0:
        emg = emg[lag * r :]
    elif lag < 0:
        kin, stim = kin[-lag:], stim[-lag:]
    n = min(len(emg) // r, len(kin), len(stim))
    emg, kin, stim = emg[: n * r], kin[:n], stim[:n]

    meta = {
        "key": ref.key,
        "dataset": ref.dataset,
        "subject": f"{ref.dataset}/{ref.subject}",
        "session": ref.session,
        "hand": src.hand,
        **{k: v for k, v in ref.info.items() if k != "file"},
        **rec.meta,
        **src.static_meta(),
        "n_channels": emg.shape[1],
        "n_samples": len(emg),
        "n_labels": n,
        "duration_s": len(emg) / cfg.fs,
        "fs": cfg.fs,
        "label_fs": cfg.label_fs,
        "joints": list(CANONICAL_JOINTS),
        "measured_joints": [j for j, name in enumerate(CANONICAL_JOINTS) if np.isfinite(kin[:, j]).any()],
        "emg_stats": emg_stats(emg, cfg.fs, cfg.calib_s),
        "kin_stats": kin_stats(kin),
        "build": asdict(cfg),
        "build_hash": bhash,
    }
    meta.setdefault("label_source", "cue")
    meta.setdefault("kin_units", None)
    store.write_session(store.session_dir(out, ref.key), emg, kin, stim, meta)
    return {
        "key": ref.key,
        "status": "built",
        "seconds": round(time.perf_counter() - t0, 1),
        "hours": n / cfg.label_fs / 3600,
    }


def auto_workers() -> int:
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    try:
        meminfo = Path("/proc/meminfo").read_text()
        avail_gb = int(next(ln for ln in meminfo.splitlines() if ln.startswith("MemAvailable")).split()[1]) / 2**20
    except (OSError, StopIteration):
        avail_gb = 8.0
    return max(1, min(cpus - 1, int(avail_gb // GB_PER_WORKER)))


def build_dataset(name: str, out: Path, raw: Path, cfg: BuildConfig, fraction: float, workers: int, dl_workers: int):
    src = get_source(name)
    subjects = select_subjects(name, src.subjects(), fraction)
    raw_dir, ds_out = raw / name, out / name
    bhash = build_hash(cfg, code_hash())
    plan = {s: src.sessions(s) for s in subjects}
    todo = {
        s: [r for r in refs if not store.is_built(store.session_dir(out, r.key), bhash)] for s, refs in plan.items()
    }
    todo = {s: refs for s, refs in todo.items() if refs}
    n_sessions = sum(len(v) for v in plan.values())
    print(
        f"[{name}] {len(subjects)}/{len(src.subjects())} subjects, {n_sessions} sessions, "
        f"{sum(len(v) for v in todo.values())} to build, {workers} workers",
        flush=True,
    )

    _prune(ds_out, {r.key for refs in plan.values() for r in refs}, out)
    results, failures = [], []
    t0 = time.perf_counter()
    # spawn: forking a process that runs download threads can deadlock
    with ThreadPoolExecutor(dl_workers) as dl, ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as cpu:
        downloads: dict[Future, str] = {dl.submit(_download_subject, refs, raw_dir): s for s, refs in todo.items()}
        jobs: dict[Future, str] = {}
        pending = set(downloads)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                if f in downloads:
                    s = downloads[f]
                    if f.exception():
                        failures.append({"key": f"{name}/{s}", "error": f"download: {f.exception()!r}"})
                        continue
                    for ref in todo[s]:
                        job = cpu.submit(process_session, ref, raw_dir, out, cfg, bhash)
                        jobs[job] = ref.key
                        pending.add(job)
                else:
                    if f.exception():
                        failures.append({"key": jobs[f], "error": repr(f.exception())})
                    else:
                        results.append(res := f.result())
                        print(
                            f"[{name}] {res['key']}: {res['status']} {res.get('seconds', res.get('reason', ''))}",
                            flush=True,
                        )

    ds_out.mkdir(parents=True, exist_ok=True)
    index = store.write_index(ds_out)
    summary = {
        "dataset": name,
        "fraction": fraction,
        "subjects": subjects,
        "sessions": len(index),
        "hours": float(index["duration_s"].sum() / 3600) if len(index) else 0.0,
        "skipped": [r for r in results if r["status"] == "skipped"],
        "failures": failures,
        "build": asdict(cfg),
        "build_hash": bhash,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    (ds_out / "build.json").write_text(json.dumps(summary, indent=1))
    print(f"[{name}] {len(index)} sessions, {summary['hours']:.1f} h, {len(failures)} failed, {summary['wall_s']} s")
    return summary


def _download_subject(refs: list[SessionRef], raw_dir: Path) -> None:
    seen = set()
    for ref in refs:
        for f in ref.files:
            if f.name not in seen:
                seen.add(f.name)
                store.download(f, raw_dir)


def _prune(ds_out: Path, keep: set[str], out: Path) -> None:
    """Remove sessions of this dataset that are no longer planned (e.g. after lowering the fraction)."""
    if not ds_out.exists():
        return
    for d in ds_out.glob("*/*"):
        if d.is_dir() and d.relative_to(out).as_posix() not in keep:
            shutil.rmtree(d)
    for d in ds_out.glob("*"):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--dataset", action="append", help="repeatable; default: `datasets` in params.yaml")
    ap.add_argument("--fraction", type=float, help="fraction of subjects per dataset (overrides params)")
    ap.add_argument("--out", default="data/built")
    ap.add_argument("--raw", default="data/raw", help="download cache, one folder per dataset")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args(argv)

    params = yaml.safe_load(Path(args.params).read_text())
    b = dict(params["build"])
    fraction = args.fraction if args.fraction is not None else b.pop("fraction", 1.0)
    b.pop("fraction", None)
    w = b.pop("workers", "auto")
    workers = args.workers or (auto_workers() if w == "auto" else int(w))
    dl_workers = int(b.pop("download_workers", 4))
    cfg = BuildConfig(**b)
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # inherited by the spawned workers: one core each
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    failed = False
    for name in args.dataset or params["datasets"]:
        s = build_dataset(name, Path(args.out), Path(args.raw), cfg, fraction, workers, dl_workers)
        failed |= bool(s["failures"])
        for f in s["failures"]:
            print(f"FAILED {f['key']}: {f['error']}", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
