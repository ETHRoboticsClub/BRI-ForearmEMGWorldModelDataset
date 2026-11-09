"""Sliding-window access to a built dataset. Numpy only: torch's DataLoader accepts it as a map-style dataset.

    from femg.loader import EMGWindows
    ds = EMGWindows("data/built", datasets=["emg_fk", "ninapro_db8"], fraction=0.1, window_s=0.48)
    train, val, test = ds.split_subjects(val=0.1, test=0.1)          # subject-disjoint
    loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, num_workers=8)

Windows are cut on the fly from memory-mapped session arrays (nothing overlapping is stored), so the dataset on
disk stays ~stride/window times smaller than pre-windowed arrays and any window/stride can be tried without a rebuild.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from femg.subset import select_subjects

NORMALIZERS = ("none", "calib_rms", "rms", "median_abs")
LABEL_NORMALIZERS = ("none", "session_zscore", "uncalibrated_zscore")


class EMGWindows:
    """Each item: emg (C, W) float32, kin (J,) float32, mask (J,) bool, stimulus int, session int.

    The target is the hand at the last EMG sample of the window (causal decoding).

    Args:
        root: build output directory (contains <dataset>/sessions.parquet).
        datasets: dataset names; default all built ones under root.
        fraction: fraction of subjects per dataset, nested and identical to the builder's selection.
        subjects / exclude_subjects: global subject ids ("emg_fk/s01").
        conditions: keep only these conditions, e.g. ["amputee_transradial"].
        require_kin: drop sessions without any measured joint (cue-only data).
        window_s, stride_s: window length, and step between consecutive targets.
        montage: channel subset from the session's meta (e.g. "ring8"), or "all" (needs equal channel counts).
        normalize: per-session per-channel EMG scaling, from the stats stored at build time.
        label_norm: "session_zscore" standardizes every joint per session; "uncalibrated_zscore" only does it for
            sessions whose kin_units is rad_uncalibrated (raw glove) and leaves calibrated angles in radians.
        in_memory: load all sessions in RAM instead of memory-mapping (fast random access when RAM allows).
    """

    def __init__(
        self,
        root,
        datasets=None,
        fraction=1.0,
        subjects=None,
        exclude_subjects=None,
        conditions=None,
        require_kin=True,
        window_s=0.48,
        stride_s=0.08,
        montage="ring8",
        normalize="calib_rms",
        label_norm="uncalibrated_zscore",
        in_memory=False,
        _sessions=None,
    ) -> None:
        if normalize not in NORMALIZERS:
            raise ValueError(f"normalize must be one of {NORMALIZERS}")
        if label_norm not in LABEL_NORMALIZERS:
            raise ValueError(f"label_norm must be one of {LABEL_NORMALIZERS}")
        self.root = Path(root)
        self.montage, self.normalize, self.label_norm, self.in_memory = montage, normalize, label_norm, in_memory
        self.window_s, self.stride_s = window_s, stride_s

        if _sessions is None:
            _sessions = self._select(datasets, fraction, subjects, exclude_subjects, conditions, require_kin)
        self.sessions: pd.DataFrame = _sessions.reset_index(drop=True)
        if self.sessions.empty:
            raise ValueError("no sessions match the selection")
        self._index()
        self._cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if in_memory:
            for s in range(len(self.sessions)):
                self._arrays(s)

    # ---- selection -------------------------------------------------------------------------------------------
    def _select(self, datasets, fraction, subjects, exclude_subjects, conditions, require_kin) -> pd.DataFrame:
        if datasets is None:
            datasets = sorted(p.parent.name for p in self.root.glob("*/sessions.parquet"))
        frames = []
        for name in datasets:
            path = self.root / name / "sessions.parquet"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing: build it with `femg-build --dataset {name}`")
            df = pd.read_parquet(path)
            local = df["subject"].str.split("/").str[1]
            keep = select_subjects(name, sorted(local.unique()), fraction)
            frames.append(df[local.isin(keep)])
        df = pd.concat(frames, ignore_index=True)
        if subjects is not None:
            df = df[df["subject"].isin(subjects)]
        if exclude_subjects is not None:
            df = df[~df["subject"].isin(exclude_subjects)]
        if conditions is not None:
            df = df[df["condition"].isin(conditions)]
        metas = [json.loads((self.root / k / "meta.json").read_text()) for k in df["key"]]
        if require_kin:
            has = [bool(m["measured_joints"]) for m in metas]
            df, metas = df[has], [m for m, h in zip(metas, has, strict=True) if h]
        df = df.copy()
        df["_meta"] = metas
        return df

    def _index(self) -> None:
        metas = self.sessions["_meta"].tolist()
        rates = {(m["fs"], m["label_fs"]) for m in metas}
        if len(rates) != 1:
            raise ValueError(f"sessions built at different rates {rates}; rebuild with one params.yaml")
        self.fs, self.label_fs = rates.pop()
        self.ratio = int(self.fs // self.label_fs)
        self.window = int(round(self.window_s * self.fs))
        self.stride = max(1, int(round(self.stride_s * self.label_fs)))
        self.k0 = -(-(self.window - 1) // self.ratio)  # first label whose window fits

        self.channels, self.scale, self.kin_mean, self.kin_std = [], [], [], []
        for m in metas:
            ch = list(range(m["n_channels"])) if self.montage == "all" else m["montages"].get(self.montage)
            if ch is None:
                raise ValueError(f"{m['key']} has no montage {self.montage!r} (has {list(m['montages'])})")
            self.channels.append(np.asarray(ch))
            if self.normalize == "none":
                scale = np.ones(len(ch), np.float32)
            else:
                scale = 1.0 / np.maximum(np.asarray(m["emg_stats"][self.normalize], np.float32)[ch], 1e-12)
            self.scale.append(scale[:, None])
            zscore = self.label_norm == "session_zscore" or (
                self.label_norm == "uncalibrated_zscore" and m.get("kin_units") == "rad_uncalibrated"
            )
            st = m["kin_stats"]
            mean = np.array([v if v is not None else 0.0 for v in st["mean"]], np.float32)
            std = np.array([v if v else 1.0 for v in st["std"]], np.float32)
            self.kin_mean.append(mean if zscore else np.zeros_like(mean))
            self.kin_std.append(std if zscore else np.ones_like(std))
        n_ch = {len(c) for c in self.channels}
        if len(n_ch) != 1:
            raise ValueError(f"channel counts differ across sessions {n_ch}: choose a montage, e.g. 'ring8'")
        self.n_channels = n_ch.pop()

        n_windows = [max(0, -(-(m["n_labels"] - self.k0) // self.stride)) for m in metas]
        self.offsets = np.concatenate([[0], np.cumsum(n_windows)]).astype(np.int64)

    # ---- access ----------------------------------------------------------------------------------------------
    def _arrays(self, s: int):
        """Opened lazily, so each DataLoader worker maps its own files and nothing big is pickled."""
        if s not in self._cache:
            d = self.root / self.sessions.at[s, "key"]
            mode = None if self.in_memory else "r"
            self._cache[s] = (
                np.load(d / "emg.npy", mmap_mode=mode),
                np.load(d / "kin.npy"),
                np.load(d / "stimulus.npy"),
            )
        return self._cache[s]

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, i: int) -> dict:
        if i < 0:
            i += len(self)
        s = int(np.searchsorted(self.offsets, i, side="right") - 1)
        k = self.k0 + (i - int(self.offsets[s])) * self.stride
        emg, kin, stim = self._arrays(s)
        end = k * self.ratio + 1
        x = np.asarray(emg[end - self.window : end])[:, self.channels[s]].T * self.scale[s]
        y = kin[k]
        mask = np.isfinite(y)
        y = np.where(mask, (y - self.kin_mean[s]) / self.kin_std[s], 0.0).astype(np.float32)
        return {
            "emg": np.ascontiguousarray(x, np.float32),
            "kin": y,
            "mask": mask,
            "stimulus": int(stim[k]),
            "session": s,
        }

    # ---- splits ----------------------------------------------------------------------------------------------
    @property
    def subjects(self) -> list[str]:
        return sorted(self.sessions["subject"].unique())

    def subset(self, subjects) -> "EMGWindows":
        """Same windowing and normalization, restricted to the given global subject ids."""
        df = self.sessions[self.sessions["subject"].isin(set(subjects))]
        return EMGWindows(
            self.root,
            window_s=self.window_s,
            stride_s=self.stride_s,
            montage=self.montage,
            normalize=self.normalize,
            label_norm=self.label_norm,
            in_memory=self.in_memory,
            _sessions=df,
        )

    def split_subjects(self, val=0.1, test=0.1, seed=0) -> tuple["EMGWindows", "EMGWindows", "EMGWindows"]:
        """Subject-disjoint train/val/test, stratified per dataset (every dataset with >= 3 subjects appears in all)."""
        rng = np.random.default_rng(seed)
        parts = {"train": [], "val": [], "test": []}
        for _, g in self.sessions.groupby("dataset"):
            subs = np.array(sorted(g["subject"].unique()))
            rng.shuffle(subs)
            n_test = int(round(test * len(subs))) if len(subs) >= 3 else 0
            n_val = int(round(val * len(subs))) if len(subs) >= 3 else 0
            n_test, n_val = max(n_test, int(test > 0 and len(subs) >= 3)), max(n_val, int(val > 0 and len(subs) >= 3))
            parts["test"] += subs[:n_test].tolist()
            parts["val"] += subs[n_test : n_test + n_val].tolist()
            parts["train"] += subs[n_test + n_val :].tolist()
        return tuple(self.subset(parts[p]) for p in ("train", "val", "test"))
