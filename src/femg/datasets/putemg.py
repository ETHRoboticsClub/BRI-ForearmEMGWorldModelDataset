from pathlib import Path

import numpy as np

from femg.datasets.base import CONFIG_DIR, Dataset, Session
from femg.download import RemoteFile

# The putEMG share (putemg-downloader's BASE_URL), read over its public WebDAV path; the project page's TLS
# certificate does not match its host (2026-11), the share works
URL = "https://chmura.put.poznan.pl/public.php/dav/files/45NY5snj0U4tgQz/Data-HDF5"
RECORDS = CONFIG_DIR / "putemg_records.txt"
TRAJECTORIES = ("repeats_long", "sequential", "repeats_short")
# values_block_3 of the HDF5 (a pandas frame_table): EMG_1..EMG_24, TRAJ_1, subject, TRAJ_GT_NO_FILTER, VIDEO_STAMP,
# TRAJ_GT
N_EMG = 24
TRAJ_GT = 28  # the labels checked on video; -1: rejected (contradictory) or not labelled


def records() -> list[str]:
    return [line for line in RECORDS.read_text().splitlines() if line and not line.startswith("#")]


class PutEMG(Dataset):
    name = "putemg"

    def subjects(self) -> list[str]:
        return [f"s{n}" for n in sorted({r.split("-")[1] for r in records()})]

    def subject_records(self, subject: str) -> dict[str, str]:
        """session -> record: <trajectory>_<1|2>, the two experiments of a participant (a week apart) by date."""
        own = sorted((r for r in records() if r.split("-")[1] == subject[1:]), key=lambda r: r.split("-", 3)[3])
        out = {}
        for trajectory in TRAJECTORIES:
            for k, record in enumerate((r for r in own if r.split("-")[2] == trajectory), 1):
                out[f"{trajectory}_{k}"] = record
        return out

    def sessions(self, subject: str) -> list[str]:
        return list(self.subject_records(subject))

    def files_to_download(self, subject: str) -> list[RemoteFile]:
        return [RemoteFile(url=f"{URL}/{r}.hdf5", name=f"{r}.hdf5") for r in self.subject_records(subject).values()]

    def load_session(self, subject: str, session: str, raw_dir: Path) -> Session:
        """One recording (one trajectory), continuous. Cues from TRAJ_GT."""
        import h5py

        record = self.subject_records(subject)[session]
        with h5py.File(raw_dir / f"{record}.hdf5", "r") as f:
            values = f["data/table"]["values_block_3"]
            emg = np.asarray(values[:, :N_EMG], np.float32) * np.float32(self.config.emg_unit_scale)
            stimulus = np.nan_to_num(np.asarray(values[:, TRAJ_GT]), nan=-1).astype(np.int16)
        info = {
            "condition": "intact",
            "label_source": "cue",
            "kin_units": None,
            "trajectory": session.rsplit("_", 1)[0],
            "experiment": int(session.rsplit("_", 1)[1]),
            "record": record,
        }
        fs = self.config.native_fs
        return Session(subject, session, emg=emg, fs=fs, kin=None, kin_fs=fs, stimulus=stimulus, info=info)
