import csv
import gzip
import json
import re
import urllib.request
from functools import cache
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from femg.datasets.base import Dataset, Session, concat_trials
from femg.datasets.ninapro import glove_to_canonical
from femg.download import RemoteFile
from femg.hand.joints import CYBERGLOVE_18

# The files are on OSF under opaque ids, split over three projects: subjects 1-10, 11-20, 21-25.
# seeds_files.csv.gz pins name, id and md5 of every trial; rebuild it with `python -m femg.datasets.seeds`.
OSF_PROJECTS = ["wa3qk", "e7m5k", "yp4vq"]
MANIFEST = Path(__file__).with_name("seeds_files.csv.gz")
TRIAL = re.compile(r"detop_exp01_subj(\d\d)_Sess(\d)_(\d\d)_(\d\d)\.mat")  # subject, session, movement, repetition

# No sensor is flipped. Checked on s01 (3 sessions), on the frames of each trial that move furthest from its start:
# thumbFlex raises thumb MCP/IP flexion and thumb roll (thumb_cmc_flex) and lowers thumb abduction (thumb_cmc_abd),
# as the canonical thumb labels; index_flex / middle_flex raise their MCP. The spread sensors gave no consistent
# answer (disc: +0.3 / -0.5 / 0.0 across sessions) and the wrist was not instructed.
FLIPPED: tuple[str, ...] = ()
# Trial movement ids (CC in the file names) -> the `movement` string stored in each file
MOVEMENTS = {
    1: "point",
    2: "thumbExt",
    3: "thumbAdd",
    4: "thumbFlex",
    5: "fist",
    6: "disc",
    7: "pinch",
    8: "cilinder",
    9: "index",
    10: "middle",
    11: "3digit",
    12: "index_flex",
    13: "middle_flex",
}


@cache
def manifest() -> dict[str, list[tuple[str, str, str]]]:
    """subject -> [(file name, OSF id, md5)] of its trials."""
    files: dict[str, list[tuple[str, str, str]]] = {}
    with gzip.open(MANIFEST, "rt", newline="") as f:
        for name, osf_id, md5 in csv.reader(f):
            match = TRIAL.fullmatch(name)
            if match:
                files.setdefault(f"s{match[1]}", []).append((name, osf_id, md5))
    return files


class SEEDS(Dataset):
    name = "seeds"

    def subjects(self) -> list[str]:
        return sorted(manifest())

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        return [
            RemoteFile(url=f"https://osf.io/download/{osf_id}/", name=name, md5=md5)
            for name, osf_id, md5 in manifest()[subject]
        ]

    def sessions(self, subject: str) -> list[str]:
        return sorted({f"Sess{TRIAL.fullmatch(name)[2]}" for name, _, _ in manifest()[subject]})

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        """The 13 movements x 6 repetitions of one experimental session, joined in that order."""
        import scipy.io

        names = sorted(name for name, _, _ in manifest()[subject] if f"_{session}_" in name)
        trials = []
        for name in names:
            data = scipy.io.loadmat(
                raw_dir / name, variable_names=["emg", "glove", "fs_emg", "fs_glove", "channels_glove"]
            )
            fs, kin_fs = float(data["fs_emg"][0, 0]), float(data["fs_glove"][0, 0])
            if (fs, kin_fs) != (self.config.native_fs, 256.0) or [c.strip() for c in data["channels_glove"]] != list(
                CYBERGLOVE_18
            ):
                raise ValueError(f"{name}: unexpected rates ({fs}, {kin_fs}) or glove channels")

            emg = data["emg"].T.astype(np.float32) * np.float32(self.config.emg_unit_scale)
            glove = data["glove"].T.astype(np.float32)
            kin = glove_to_canonical(glove, CYBERGLOVE_18, FLIPPED, scale=1.0)
            stimulus = np.zeros(len(kin), np.int16)
            stimulus[movement_span(kin, kin_fs)] = int(TRIAL.fullmatch(name)[3])
            trials.append((emg, kin, stimulus))

        emg, kin, stimulus, starts = concat_trials(trials, round(self.config.native_fs / 256.0))
        info = {
            "condition": "intact",
            "label_source": "measured",
            "kin_units": "rad_uncalibrated",  # 0-1 per sensor, not radians
            "task": "13 instructed finger/hand movements, 6 repetitions each, trials joined",
            "stimulus_source": "the trial's movement where the glove moves (movement_span), rest elsewhere",
            "stimulus_names": MOVEMENTS,
        }
        return Session(
            subject,
            session,
            emg=np.ascontiguousarray(emg),
            fs=fs,
            kin=kin,
            kin_fs=256.0,
            stimulus=stimulus,
            info=info,
            segments=starts,
        )


def movement_span(kin: np.ndarray, fs: float) -> slice:
    """Where the hand moves in a trial. The files have no cue: every trial starts at rest (3-6 s on s01, s02) and
    the hand moves or holds until it ends or returns. The span runs from the first to the last frame further from
    the trial's start pose than 20% of its furthest (mean over the sensors, smoothed over 0.1 s; the peaks are 0.05
    to 0.6 against a quantization step of 1/255)."""
    kin = kin[:, np.isfinite(kin).any(axis=0)]
    start_pose = np.median(kin[: round(0.5 * fs)], axis=0)
    distance = uniform_filter1d(np.abs(kin - start_pose).mean(axis=1), round(0.1 * fs))
    far = np.flatnonzero(distance > 0.2 * np.percentile(distance, 99))
    return slice(far[0], far[-1] + 1)


def build_manifest() -> None:
    """Lists every subject folder of the three OSF projects and writes seeds_files.csv.gz."""

    def pages(url: str):
        url += ("&" if "?" in url else "?") + "page[size]=100"
        while url:
            request = urllib.request.Request(url, headers={"User-Agent": "femg-dataset-builder"})
            with urllib.request.urlopen(request, timeout=60) as response:
                page = json.load(response)
            yield from page["data"]
            url = page["links"]["next"]

    rows = []
    for project in OSF_PROJECTS:
        for folder in pages(f"https://api.osf.io/v2/nodes/{project}/files/osfstorage/"):
            if folder["attributes"]["kind"] != "folder" or not re.fullmatch(r"subj\d\d", folder["attributes"]["name"]):
                continue
            for file in pages(folder["relationships"]["files"]["links"]["related"]["href"]):
                if file["attributes"]["kind"] == "file":
                    osf_id = file["links"]["download"].rstrip("/").rsplit("/", 1)[1]
                    rows.append((file["attributes"]["name"], osf_id, file["attributes"]["extra"]["hashes"]["md5"]))
    with gzip.open(MANIFEST, "wt", newline="") as f:
        csv.writer(f).writerows(sorted(rows))
    print(f"{MANIFEST}: {len(rows)} files")


if __name__ == "__main__":
    build_manifest()
