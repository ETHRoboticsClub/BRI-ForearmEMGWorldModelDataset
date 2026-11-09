from pathlib import Path

from femg.schema import Recording, SessionRef


class Source:
    """One public dataset. Subclasses list its sessions and read one session into a `Recording`.

    Class attributes are static metadata copied into every session's meta.json. Keep them factual and
    cite where they come from; unknowns are None, not guesses.
    """

    name: str
    citation: str
    license: str
    url: str
    device: str
    electrode_type: str  # "bipolar" (double-differential) or "monopolar" (common reference)
    native_fs: float  # EMG rate of the files
    effective_fs: float  # EMG rate before any upsampling done by the dataset authors
    mains_hz: float | None  # power-line frequency where it was recorded
    emg_unit_scale: float = 1.0  # multiply the file's EMG by this to get volts
    hand: str = "right"  # hand whose EMG is recorded
    # Channel subsets the loader can request to get the same channel count across datasets.
    # ring8: 8 electrodes evenly spaced around the forearm. Their rotation is NOT aligned across datasets.
    montages: dict[str, list[int]] = {}
    kin_lag_s: float = 0.0  # labels[t] describe the hand at t + kin_lag_s; the builder shifts them by this
    verified: dict[str, str] = {}  # what was checked against the real files, and how

    def subjects(self) -> list[str]:
        raise NotImplementedError

    def sessions(self, subject: str) -> list[SessionRef]:
        raise NotImplementedError

    def read(self, ref: SessionRef, raw_dir: Path) -> Recording:
        raise NotImplementedError

    def static_meta(self) -> dict:
        keys = (
            "citation",
            "license",
            "url",
            "device",
            "electrode_type",
            "native_fs",
            "effective_fs",
            "mains_hz",
            "hand",
            "montages",
            "kin_lag_s",
            "verified",
        )
        return {k: getattr(self, k) for k in keys}
