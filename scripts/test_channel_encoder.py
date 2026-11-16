import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from femg.channel_encoder import ChannelEncoder, ElectrodeCalibration
from femg.electrodes import ElectrodeLayout, distance_cm
from femg.normalize import as_array, display_montage, hand_flexion, has_kinematics, normalize_kin
from femg.signal_utils import envelope

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100", "#8a5cd1", "#1aa3b8", "#b5651d", "#6b8e23"]
INK = "#28282a"
DISTANCE_BINS = np.arange(0, 16.5, 1.5)  # cm
LAG_S = 0.2  # the pose trails the EMG (scripts/estimate_kin_lag.py)
FOLDS = 5


def load(folder: Path, max_s: float) -> dict:
    meta = json.loads((folder / "meta.json").read_text())
    per_label = meta["fs"] // meta["label_fs"]
    n = int(max_s * meta["label_fs"]) * per_label
    emg = np.asarray(np.load(folder / "emg.npy", mmap_mode="r")[:n], np.float32)
    window = round(meta["processor"]["rms_window_s"] * meta["label_fs"])
    out = {
        "meta": meta,
        "emg": emg / as_array(meta["emg_stats"]["active"]),  # the encoder's input: EMG / active per channel
        "log_env": np.log(envelope(emg, per_label, window).astype(np.float64) + 1e-9),
        "positions": ElectrodeLayout.from_meta(meta).positions(),
        "groups": np.array([e["group"] for e in meta["electrodes"]]),
    }
    if has_kinematics(meta):
        kin = normalize_kin(np.load(folder / "kin.npy")[: len(out["log_env"])], meta["kin_stats"])
        out["pose"] = kin[:, meta["measured_joints"]].astype(np.float64)
        out["flexion"] = hand_flexion(kin).astype(np.float64)
    return out


# Model checks


def latents(model, s: dict, seconds: float = 4.0, positions_=None, mask=None, calibration=None) -> torch.Tensor:
    n = int(seconds * s["meta"]["fs"])
    emg = torch.from_numpy(s["emg"][:n].T.copy())[None]
    pos = torch.from_numpy(s["positions"] if positions_ is None else positions_)[None]
    return model(emg, pos, mask, calibration)


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def model_checks(model, sessions: dict[str, list[dict]]) -> tuple[list, dict]:
    checks = []
    one = {name: runs[0] for name, runs in sessions.items()}
    shapes = {tuple(latents(model, s, 1.0).shape[2:]) for s in one.values()}
    counts = sorted({s["meta"]["n_channels"] for s in one.values()})
    checks.append(("same latent shape for every layout", len(shapes) == 1, f"{counts} channels -> {shapes}"))

    s = one.get("ninapro_db2") or next(iter(one.values()))
    reference = latents(model, s)
    order = np.random.default_rng(0).permutation(s["meta"]["n_channels"])
    shuffled = dict(s, emg=s["emg"][:, order], positions=s["positions"][order])
    diff = relative(latents(model, shuffled), reference)
    checks.append(("channel order ignored", diff < 1e-4, f"relative change {diff:.1e}"))

    noise = np.random.default_rng(1).normal(size=(len(s["emg"]), 4))
    fake = np.tile(np.float32([[90.0, 0.5, 1.0]]), (4, 1))
    padded = dict(s, emg=np.c_[s["emg"], noise].astype(np.float32), positions=np.r_[s["positions"], fake])
    mask = torch.ones(1, padded["positions"].shape[0], dtype=torch.bool)
    mask[0, -4:] = False
    diff = relative(latents(model, padded, mask=mask), reference)
    checks.append(("padded channels ignored", diff < 1e-4, f"relative change {diff:.1e}"))

    rotated = s["positions"].copy()
    rotated[:, 0] += 45
    diff = relative(latents(model, s, positions_=rotated), reference)
    checks.append(("positions change the output", diff > 0.01, f"45 deg rotation: relative change {diff:.2f}"))

    if "kin_mus_uji" in one:
        finite = bool(torch.isfinite(latents(model, one["kin_mus_uji"], 1.0)).all())
        checks.append(("unknown positions handled (KIN-MUS)", finite, "finite latents"))
    return checks


def calibration_landscape(model, s: dict, error_deg: float = 20.0, steps: int = 200) -> dict:
    """The band is worn rotated by error_deg from where the encoder expects it. Target: the latents of the right
    placement (a stand-in for the pose loss). Sweep the correction, then fit it by gradient descent from 0.
    For an evenly spaced ring the sweep repeats every electrode spacing: geometry alone fixes the rotation only up to
    a whole number of electrodes; only what the encoder learns about anatomy can break that tie."""
    target = latents(model, s).detach()
    worn = s["positions"].copy()
    worn[:, 0] += error_deg
    sweep = np.arange(-180, 181, 3)
    loss = []
    with torch.no_grad():
        for correction in sweep:
            moved = worn.copy()
            moved[:, 0] += correction
            loss.append(relative(latents(model, s, positions_=moved), target) ** 2)

    calibration = ElectrodeCalibration(s["meta"]["n_channels"])
    optimizer = torch.optim.Adam([calibration.rotation, calibration.slide], lr=2.0)  # the gains are right here
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    path = []
    for _ in range(steps):
        optimizer.zero_grad()
        moved = latents(model, s, positions_=worn, calibration=calibration)
        error = (moved - target).pow(2).mean() / target.pow(2).mean()
        error.backward()
        optimizer.step()
        schedule.step()
        path.append(calibration.rotation.item())
    return {"sweep": sweep, "loss": np.array(loss), "path": np.array(path), "true": -error_deg}


def embedding_kernel(n_angles=(3, 6, 12)) -> dict:
    """Cosine similarity of the position features between an electrode at 0 deg and one at d deg (same z): how far
    apart two electrodes must be for the encoder to tell them apart, before training."""
    d = np.arange(0, 181, 2.0)
    out = {}
    for n in n_angles:
        model = ChannelEncoder(n_angle=n)
        pos = torch.tensor(np.c_[d, np.full_like(d, 0.2), np.ones_like(d)], dtype=torch.float32)
        f = model.embed.features(pos)
        out[n] = torch.nn.functional.cosine_similarity(f, f[:1]).detach().numpy()
    return {"degrees": d, "similarity": out}


# Data checks


def spatial_decay(runs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Envelope correlation of every channel pair of a group (known relative placement) vs their distance (cm)."""
    distances, correlations = [], []
    for s in runs:
        known = np.isfinite(s["positions"][:, :2]).all(axis=1)
        corr = np.corrcoef(s["log_env"].T)
        for i in range(len(known)):
            for j in range(i + 1, len(known)):
                if known[i] and known[j] and s["groups"][i] == s["groups"][j]:
                    distances.append(distance_cm(s["positions"][i], s["positions"][j]))
                    correlations.append(corr[i, j])
    return np.array(distances), np.array(correlations)


def ring_of(s: dict) -> np.ndarray:
    """The first 8 channels of the band (ring8 montage), the ones the rings of all datasets share."""
    return np.array(s["meta"]["montages"].get("ring8", []))


def flexion_profile(runs: list[dict]) -> np.ndarray | None:
    """Per ring channel: correlation of its envelope with hand flexion, minus the mean over the ring."""
    profiles = []
    for s in runs:
        ring = ring_of(s)
        if len(ring) != 8 or "flexion" not in s:
            return None
        ok = np.isfinite(s["flexion"])
        r = np.array([np.corrcoef(s["log_env"][ok, c], s["flexion"][ok])[0, 1] for c in ring])
        profiles.append(r - r.mean())
    return np.mean(profiles, axis=0)


def align(profile: np.ndarray, reference: np.ndarray) -> tuple[int, int, float]:
    """The ring rotation (deg) and direction (+1/-1) that best match a profile to the reference: (offset, sign, r)."""
    angles = np.arange(8) * 45.0
    best = (0, 1, -np.inf)
    for sign in (1, -1):
        for offset in range(360):
            moved = (sign * angles + offset) % 360
            order = np.argsort(moved)
            at_reference = np.interp(angles, moved[order], profile[order], period=360)
            r = np.corrcoef(at_reference, reference)[0, 1]
            if r > best[2]:
                best = (offset, sign, r)
    return best


def shifted(profile: np.ndarray, offset: int, sign: int, at: np.ndarray) -> np.ndarray:
    moved = (sign * np.arange(8) * 45.0 + offset) % 360
    order = np.argsort(moved)
    return np.interp(at, moved[order], profile[order], period=360)


def decoding_r2(s: dict, channels: np.ndarray, ridge: float = 10.0) -> float:
    """Cross-validated R2 of a ridge decoder: log envelopes of `channels` at t -> 0-1 pose at t + LAG_S."""
    lag = round(LAG_S * s["meta"]["label_fs"])
    x, y = s["log_env"][:-lag, channels], s["pose"][lag:]
    ok = np.isfinite(y).all(axis=1) & np.isfinite(x).all(axis=1)
    x, y = x[ok], y[ok]
    fold = np.arange(len(x)) * FOLDS // len(x)
    error = total = 0.0
    for f in range(FOLDS):
        train, test = fold != f, fold == f
        mean, std = x[train].mean(axis=0), x[train].std(axis=0) + 1e-9
        a = np.c_[(x[train] - mean) / std, np.ones(train.sum())]
        b = np.c_[(x[test] - mean) / std, np.ones(test.sum())]
        weights = np.linalg.solve(a.T @ a + ridge * np.eye(a.shape[1]), a.T @ y[train])
        error += ((y[test] - b @ weights) ** 2).sum()
        total += ((y[test] - y[train].mean(axis=0)) ** 2).sum()
    return 1 - error / total


# Report


def layouts(sessions: dict[str, list[dict]]) -> dict[str, np.ndarray]:
    """Datasets with the same electrode layout share one entry: {"db2, db7": positions}."""
    by_layout = defaultdict(list)
    for name, runs in sessions.items():
        by_layout[runs[0]["positions"].round(2).tobytes()].append((name, runs[0]["positions"]))
    return {", ".join(n for n, _ in entries): entries[0][1] for entries in by_layout.values()}


def main():
    parser = argparse.ArgumentParser(description="Checks of the channel encoder (untrained) and of the data for it.")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("output/channel_encoder"))
    parser.add_argument("--per-dataset", type=int, default=2, help="sessions per dataset")
    parser.add_argument("--max-s", type=float, default=300, help="seconds read per session")
    parser.add_argument("--reference", default="emg_fk", help="dataset the rings are aligned to (your band)")
    args = parser.parse_args()
    torch.manual_seed(0)

    sessions = {}
    for dataset in sorted(p for p in args.data.iterdir() if p.is_dir()):
        folders = sorted(dataset.glob("*/*/meta.json"))[: args.per_dataset]
        runs = [load(f.parent, args.max_s) for f in folders]
        if runs and "electrodes" in runs[0]["meta"]:
            sessions[dataset.name] = runs
        elif runs:
            print(f"{dataset.name}: no electrode positions in meta.json, rebuild it (build_dataset.py --overwrite)")
    model = ChannelEncoder().eval()
    color = {name: COLORS[i % len(COLORS)] for i, name in enumerate(sessions)}

    with torch.no_grad():
        checks = model_checks(model, sessions)
    landscape = calibration_landscape(model, sessions.get(args.reference, next(iter(sessions.values())))[0])
    found, true = landscape["path"][-1], landscape["true"]
    checks.append(("calibration recovers a 20 deg band rotation", abs(found - true) < 5, f"fitted {found:.0f} deg"))

    decay = {name: spatial_decay(runs) for name, runs in sessions.items()}
    for name, (d, c) in decay.items():
        if len(d) > 10:
            rho = spearmanr(d, c).statistic
            checks.append((f"{name}: correlation falls with distance", rho < -0.2, f"Spearman {rho:.2f}"))

    profiles = {name: p for name, runs in sessions.items() if (p := flexion_profile(runs)) is not None}
    reference = profiles.get(args.reference)
    alignment = {name: align(p, reference) for name, p in profiles.items()} if reference is not None else {}
    for name, (offset, sign, r) in alignment.items():
        if name != args.reference:
            way = "same direction" if sign > 0 else "reversed"
            checks.append((f"{name}: ring aligns with {args.reference}", r > 0.7, f"{offset} deg, {way}, r {r:.2f}"))

    r2 = {}
    for name, runs in sessions.items():
        if "pose" in runs[0] and runs[0]["meta"]["n_channels"] > 8:
            small = np.array(display_montage(runs[0]["meta"])[1])
            every = np.arange(runs[0]["meta"]["n_channels"])
            r2[name] = (np.mean([decoding_r2(s, small) for s in runs]), np.mean([decoding_r2(s, every) for s in runs]))

    kernel = embedding_kernel()

    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    (ax_layout, ax_decay, ax_ring), (ax_r2, ax_kernel, ax_calib) = axes

    for i, (names, pos) in enumerate(layouts(sessions).items()):
        known = np.isfinite(pos[:, :2]).all(axis=1)
        label = names + (f" (+{(~known).sum()} unknown)" if (~known).any() else "")
        ax_layout.scatter(pos[known, 0], pos[known, 1], s=10, color=COLORS[i % len(COLORS)], label=label)
    ax_layout.set(xlim=(0, 360), xticks=[0, 90, 180, 270, 360], ylabel="z: 0 elbow, 1 wrist")
    ax_layout.set(xlabel="angle (deg): 0 palmar, 90 radial, 180 dorsal", title="a  Electrode layouts (unrolled arm)")
    ax_layout.invert_yaxis()
    ax_layout.legend(fontsize=7, loc="lower right")

    for name, (d, c) in decay.items():
        if len(d) > 10:
            which = np.digitize(d, DISTANCE_BINS)
            bins = [b for b in np.unique(which) if (which == b).sum() > 2]
            centers = [(d[which == b].mean(), np.median(c[which == b])) for b in bins]
            kind = sessions[name][0]["meta"]["electrode_type"]
            ax_decay.plot(*np.array(centers).T, "o-", ms=3, color=color[name], label=f"{name} ({kind})")
    ax_decay.set(xlabel="distance between electrodes (cm)", ylabel="envelope correlation (median)")
    ax_decay.set_title("b  Neighbouring channels see the same muscles")
    ax_decay.legend(fontsize=7)

    fine = np.arange(0, 361, 2.0)
    for name, profile in profiles.items():
        offset, sign, r = alignment.get(name, (0, 1, np.nan))
        style = dict(color=INK, lw=2) if name == args.reference else dict(color=color[name], lw=1)
        ax_ring.plot(fine, shifted(profile, offset, sign, fine), label=f"{name} ({offset} deg, r {r:.2f})", **style)
    ax_ring.axhline(0, color="#a8a7a2", lw=0.5)
    ax_ring.set(xlim=(0, 360), xticks=np.arange(0, 361, 45), xlabel=f"angle on the {args.reference} ring (deg)")
    ax_ring.set(ylabel="channel-flexion correlation − ring mean", title="c  Ring profiles after the best rotation")
    ax_ring.legend(fontsize=7)

    names = list(r2)
    ax_r2.bar(np.arange(len(names)) - 0.2, [r2[n][0] for n in names], 0.4, color="#a8a7a2", label="display montage")
    ax_r2.bar(np.arange(len(names)) + 0.2, [r2[n][1] for n in names], 0.4, color=COLORS[0], label="all channels")
    for i, n in enumerate(names):
        ax_r2.text(i + 0.2, r2[n][1] + 0.01, f"+{r2[n][1] - r2[n][0]:.2f}", ha="center", fontsize=7)
    ax_r2.set_xticks(range(len(names)), names, rotation=20, fontsize=8)
    ax_r2.set(ylabel=f"pose R² (ridge, 5-fold, pose at t + {LAG_S} s)")
    ax_r2.set_title("d  Do the extra channels carry hand information?")
    ax_r2.legend(fontsize=7)

    for n, similarity in kernel["similarity"].items():
        ax_kernel.plot(kernel["degrees"], similarity, label=f"{n} harmonics (~{180 / n:.0f} deg)")
    ax_kernel.axvline(45, color="#a8a7a2", ls="--", lw=0.8)
    ax_kernel.text(46, 0.95, "ring8 spacing", fontsize=7, color="#6b6a66")
    ax_kernel.set(xlabel="angle between two electrodes (deg)", ylabel="cosine similarity of position features")
    ax_kernel.set_title("e  Spatial resolution of the position embedding")
    ax_kernel.legend(fontsize=7)

    ax_calib.plot(landscape["sweep"], landscape["loss"], color=INK, lw=1)
    for k in range(-4, 5):
        ax_calib.axvline(true + 45 * k, color=COLORS[2], ls=":" if k else "--", lw=1)
    ax_calib.plot([], [], color=COLORS[2], ls="--", label=f"true correction {true:.0f} (dotted: + whole electrodes)")
    ax_calib.axvline(found, color=COLORS[1], lw=1, label=f"fitted {found:.0f} (gradient descent from 0)")
    ax_calib.set(xlim=(-180, 180), xlabel="rotation correction (deg)", ylabel="latent error (relative²)")
    ax_calib.set_title(f"f  Calibration: {args.reference} band worn {-true:.0f} deg off")
    ax_calib.legend(fontsize=7)

    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "report.png", dpi=130)

    width = max(len(c[0]) for c in checks)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:{width}}  {detail}")
    if r2:
        print("\npose R2, display montage -> all channels:")
        for name, (a, b) in r2.items():
            print(f"  {name:14} {a:.2f} -> {b:.2f}")
    print(f"\nwrote {args.out / 'report.png'}")


if __name__ == "__main__":
    main()
