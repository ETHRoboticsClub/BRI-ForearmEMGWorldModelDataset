import csv
import gzip
import json
import re
import urllib.request
from functools import cache
from pathlib import Path

import numpy as np

from femg.datasets.base import Dataset, Session, concat_trials
from femg.datasets.ninapro import CYBERGLOVE_18, glove_to_canonical
from femg.download import RemoteFile

# The files are on OSF under opaque ids, split over three projects: subjects 1-10, 11-20, 21-25.
# seeds_files.csv.gz pins name, id and md5 of every trial; rebuild it with `python -m femg.datasets.seeds`.
OSF_PROJECTS = ["wa3qk", "e7m5k", "yp4vq"]
MANIFEST = Path(__file__).with_name("seeds_files.csv.gz")
TRIAL = re.compile(r"detop_exp01_subj(\d\d)_Sess(\d)_(\d\d)_(\d\d)\.mat")  # subject, session, movement, repetition

# channels_glove of the files, checked on load. Same sensors and order as the NinaPro 18-DOF CyberGlove.
GLOVE = [
    "Thumb_Roll",
    "Thumb_Inner",
    "Thumb_Outer",
    "TI_Abduction",
    "Index_Inner",
    "Index_Middle",
    "Middle_Inner",
    "Middle_Middle",
    "MI_Abduction",
    "Ring_Inner",
    "Ring_Middle",
    "RM_Abduction",
    "Pinky_Inner",
    "Pinky_Middle",
    "PR_Abduction",
    "Palm_Arch",
    "Wrist_Flex",
    "Wrist_Abduction",
]


# No sensor is flipped. Checked on s01 (3 sessions) with scripts/check_signs.py, on the frames of each trial that
# move furthest from its start: thumbFlex raises thumb MCP/IP flexion and thumb roll (thumb_cmc_flex) and lowers
# thumb abduction (thumb_cmc_abd), as the canonical thumb labels; index_flex / middle_flex raise their MCP. The spread
# sensors gave no consistent answer (disc: +0.3 / -0.5 / 0.0 across sessions) and the wrist was not instructed.
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
    citation = "Matran-Fernandez et al. 2019, Scientific Data 6:186, doi:10.17605/OSF.IO/WA3QK"
    license = "see the OSF project (the paper is CC-BY-4.0)"
    url = "https://osf.io/wa3qk/"
    device = (
        "BioSemi ActiveTwo: 126-electrode HD grid (9x14, 4 mm) over the finger flexors + 8 monopolar electrodes over "
        "the extensors, left forearm"
    )
    electrode_type = "monopolar"
    native_fs = 2048.0  # recorded at 16384 Hz, stored at 2048 Hz
    effective_fs = 2048.0
    mains_hz = 50.0  # recorded in the UK
    emg_unit_scale = 1e-6  # assumed microvolts (BioSemi): HD rest RMS 1-2, extensors up to 30
    hand = "left"
    # No ring around the forearm: the grid covers the flexors only, the 8 extra electrodes the extensors
    montages = {"hd_flexor126": list(range(126)), "extensor8": list(range(126, 134))}
    verified = {
        "manifest": "OSF API crawl: 25 subjects x (234 trials + calibration file), 59.7 GB",
        "mat_variables": "subj02 Sess2_01_01 opened: emg (134,T) float32, glove (18,T/8) float32, fs_emg=2048, "
        "fs_glove=256, channels_emg MA1..MN9 + EXG1..8, channels_glove = GLOVE",
        "emg_preprocessing": "authors: common average reference + zero-phase 10-500 Hz band-pass (not causal)",
        "glove_units": "normalized 0-1 per sensor with each subject's calibration min/max (seen 0.08-1.5)",
        "glove_signs": "thumb and finger flexions canonical as stored (s01, 2/3 sessions); spreads and wrist not "
        "verifiable from the instructed movements",
        "movements": "s01: the movement string of each file matches MOVEMENTS",
    }

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
            if (fs, kin_fs) != (self.native_fs, 256.0) or [c.strip() for c in data["channels_glove"]] != GLOVE:
                raise ValueError(f"{name}: unexpected rates ({fs}, {kin_fs}) or glove channels")

            emg = data["emg"].T.astype(np.float32) * np.float32(self.emg_unit_scale)
            glove = data["glove"].T.astype(np.float32)
            kin = glove_to_canonical(glove, CYBERGLOVE_18, FLIPPED, scale=1.0)
            movement = int(TRIAL.fullmatch(name)[3])
            trials.append((emg, kin, np.full(len(kin), movement, np.int16)))

        emg, kin, stimulus, starts = concat_trials(trials, round(self.native_fs / 256.0))
        info = {
            "condition": "intact",
            "label_source": "measured",
            "kin_units": "rad_uncalibrated",  # 0-1 per sensor, not radians
            "task": "13 instructed finger/hand movements, 6 repetitions each, trials joined",
            "stimulus_names": MOVEMENTS,
            "trial_starts_s": [s / 256.0 for s in starts],
        }
        return Session(
            subject, session, emg=np.ascontiguousarray(emg), fs=fs, kin=kin, kin_fs=256.0, stimulus=stimulus, info=info
        )


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
