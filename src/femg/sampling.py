"""How much of a session is worth drawing, and how often to draw each session so that no dataset or recording setup
dominates: setup (device_group) -> dataset -> subject -> session, the first two levels with a temperature.

A level's options get weight (their seconds)^alpha, normalised: alpha 1 draws in proportion to the data, 0 every option
alike; between them large datasets still weigh more, small ones are not repeated without end. Each subject of a
dataset is drawn alike (subject variety over hours); a subject's sessions in proportion to their seconds."""

import numpy as np
import pandas as pd

# A label frame is active when the mean envelope of the virtual band is above this many times its floor (the
# session's 10th percentile, ~rest): it measures how much of an unlabelled session holds muscle activity
ACTIVE_OVER_FLOOR = 2.0
FLOOR_PERCENTILE = 10
LEVELS = ("device_group", "dataset", "subject")


def session_seconds(envelope: np.ndarray, gesture: np.ndarray, covered: np.ndarray, label_fs: float) -> dict:
    """envelope (T, C): moving RMS of the virtual band at label_fs; gesture (T,) shared ids; covered (C,) bool.
    -> gesture_s (a gesture other than rest), rest_s, active_s (EMG above ACTIVE_OVER_FLOOR x its floor)."""
    n = min(len(envelope), len(gesture))
    out = {"gesture_s": float((gesture[:n] > 0).sum() / label_fs), "rest_s": float((gesture[:n] == 0).sum() / label_fs)}
    if not covered.any() or n == 0:
        return out | {"active_s": 0.0}
    level = envelope[:n][:, covered].mean(axis=1)
    floor = np.percentile(level, FLOOR_PERCENTILE)
    return out | {"active_s": float((level > ACTIVE_OVER_FLOOR * floor).sum() / label_fs)}


def probabilities(sessions: pd.DataFrame, seconds: str, alpha: float) -> pd.Series:
    """Per session (rows of `sessions`, with LEVELS and the `seconds` column): the probability of drawing it, summing
    to 1 over the sessions with seconds > 0. Sampling: device_group and dataset by (seconds)^alpha, subjects alike,
    sessions by seconds."""
    s = sessions[seconds].fillna(0).clip(lower=0)
    out = pd.Series(0.0, index=sessions.index)
    used = sessions[s > 0]
    if used.empty:
        return out
    s = s[used.index]
    group_s = s.groupby(used["device_group"]).sum()
    p_group = group_s**alpha / (group_s**alpha).sum()
    dataset_s = s.groupby([used["device_group"], used["dataset"]]).sum()
    tempered = dataset_s**alpha
    p_dataset = tempered / tempered.groupby(level=0).transform("sum")
    n_subjects = used.groupby("dataset")["subject"].nunique()
    subject_s = s.groupby([used["dataset"], used["subject"]]).transform("sum")
    out[used.index] = (
        used["device_group"].map(p_group).to_numpy()
        * p_dataset.loc[list(zip(used["device_group"], used["dataset"], strict=True))].to_numpy()
        / used["dataset"].map(n_subjects).to_numpy()
        * (s / subject_s).to_numpy()
    )
    return out


def sampling_table(index: pd.DataFrame, discrete: list[str], alpha: float) -> pd.DataFrame:
    """index: the sessions.parquet of every dataset. -> per session the probability of drawing it for pretraining
    (active EMG, every dataset), discrete decoding (labelled gestures and rest, discrete datasets) and continuous
    decoding (sessions with kinematics, their duration); test subjects never (split == test: 0)."""
    table = index[["key", "dataset", "device_group", "subject", "session", "split", "label_source"]].copy()
    train = index["split"] != "test"
    uses = {
        "pretrain": (index["active_s"], train),
        "discrete": (index["gesture_s"] + index["rest_s"], train & index["dataset"].isin(discrete)),
        "continuous": (index["duration_s"], train & index["label_source"].isin(["measured", "mirrored"])),
    }
    for use, (seconds, keep) in uses.items():
        frame = index.assign(_s=seconds.where(keep, 0))
        table[f"p_{use}"] = probabilities(frame, "_s", alpha)
    return table
