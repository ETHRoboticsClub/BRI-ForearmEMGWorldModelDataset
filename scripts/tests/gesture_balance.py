"""How balanced the gestures of the discrete datasets are: per dataset (time, repetitions, how unequal its classes are),
across datasets (which shared gestures and families more than one dataset gives), and per split (whether every
gesture of a dataset has test subjects, and train ones).

Reads what the processed stage writes: data/processed/<dataset>/gestures.parquet (seconds and repetitions per
session and shared gesture) and sessions.parquet (split). Cue-only sessions (DB2 E3) count, kinematics are not read.
The movement_balance.py counterpart for the joints.

    python scripts/tests/gesture_balance.py                      # the discrete_datasets of params.yaml
    python scripts/tests/gesture_balance.py --datasets ninapro_db2 ninapro_db5 --detail
-> docs/figures/gesture_balance_families.png, gesture_balance_gestures.png and tables on stdout
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from femg.gestures import families, vocabulary

FAMILY_ORDER = list(dict.fromkeys(families()))  # in vocabulary order


def load(processed: Path, names: list[str]) -> pd.DataFrame:
    """One row per session x gesture, with the subject's split."""
    tables = []
    for name in names:
        folder = processed / name
        if not (folder / "gestures.parquet").exists():
            print(f"{name}: no gestures.parquet in {folder} (build its processed stage)")
            continue
        table = pd.read_parquet(folder / "gestures.parquet")
        sessions = pd.read_parquet(folder / "sessions.parquet")
        split = sessions.drop_duplicates("subject").set_index("subject").get("split")
        table["split"] = table["subject"].map(split) if split is not None else None
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def effective_classes(seconds: pd.Series) -> float:
    """exp(entropy) of the time shares: the number of equally large classes that would be as balanced."""
    p = seconds[seconds > 0] / seconds.sum()
    return float(np.exp(-(p * np.log(p)).sum())) if len(p) else np.nan


def dataset_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, t in table.groupby("dataset", sort=False):
        moving = t[t["gesture_id"] > 0]
        per_gesture = moving.groupby("gesture")[["seconds", "repetitions"]].sum()
        subjects = t.drop_duplicates("subject")
        rows.append(
            {
                "dataset": name,
                "subjects": len(subjects),
                "test": int((subjects["split"] == "test").sum()),
                "hours": t["seconds"].sum() / 3600,
                "rest": t.loc[t["gesture_id"] == 0, "seconds"].sum() / t["seconds"].sum(),
                "gestures": len(per_gesture),
                "families": moving["family"].nunique(),
                "reps": per_gesture["repetitions"].agg(["min", "median", "max"]).astype(int).tolist(),
                "s_per_rep": per_gesture["seconds"].sum() / max(per_gesture["repetitions"].sum(), 1),
                "max_min": per_gesture["seconds"].max() / per_gesture["seconds"].min(),
                "effective": effective_classes(per_gesture["seconds"]),
            }
        )
    return pd.DataFrame(rows)


def print_datasets(table: pd.DataFrame):
    rows = dataset_rows(table)
    print("\nPER DATASET (rest apart unless named; reps per gesture over all subjects)")
    print(
        f"{'':14}{'subjects':>9}{'test':>5}{'hours':>7}{'rest':>6}{'gestures':>9}{'families':>9}"
        f"{'reps min/med/max':>18}{'s/rep':>7}{'max/min':>8}{'eff. classes':>13}"
    )
    for r in rows.itertuples():
        reps = "/".join(map(str, r.reps))
        print(
            f"{r.dataset:14}{r.subjects:>9}{r.test:>5}{r.hours:>7.1f}{r.rest:>6.0%}{r.gestures:>9}{r.families:>9}"
            f"{reps:>18}{r.s_per_rep:>7.1f}{r.max_min:>7.1f}x{r.effective:>8.1f} / {r.gestures}"
        )


def pooled(table: pd.DataFrame, level: str) -> pd.DataFrame:
    """Per gesture or family, rest apart: minutes over every dataset, its share, and how many datasets / subjects."""
    moving = table[table["gesture_id"] > 0]
    out = moving.groupby(level).agg(
        minutes=("seconds", lambda s: s.sum() / 60),
        repetitions=("repetitions", "sum"),
        datasets=("dataset", "nunique"),
        subjects=("subject", "nunique"),
    )
    out["share"] = out["minutes"] / out["minutes"].sum()
    order = FAMILY_ORDER if level == "family" else list(vocabulary())
    return out.reindex([g for g in order if g in out.index])


def print_pooled(table: pd.DataFrame, detail: bool):
    for level in ("family", "gesture"):
        p = pooled(table, level)
        single = p[p["datasets"] == 1]
        title = "FAMILIES" if level == "family" else "GESTURES"
        print(
            f"\nPOOLED {title}, all datasets together (rest apart): {len(p)} classes, "
            f"{len(p) - len(single)} in more than one dataset; max/min {p['minutes'].max() / p['minutes'].min():.0f}x, "
            f"effective classes {effective_classes(p['minutes']):.1f}"
        )
        if level == "family" or detail:
            shown = p
        else:  # what makes the balance: the largest and smallest classes; the single-dataset ones by name
            shown = pd.concat([p.nlargest(5, "minutes"), p.nsmallest(5, "minutes")]).drop_duplicates()
            shown = shown.reindex([g for g in p.index if g in shown.index])
            if len(single):
                by_dataset = table[table["gesture"].isin(single.index)].groupby("dataset")["gesture"].unique()
                for name, gestures in by_dataset.items():
                    print(f"  only in {name}: {', '.join(g for g in p.index if g in gestures)}")
            print(f"  5 largest and 5 smallest (--detail for all {len(p)}):")
        print(f"  {'':20}{'minutes':>9}{'share':>7}{'reps':>7}{'datasets':>9}{'subjects':>9}")
        for name, r in shown.iterrows():
            print(
                f"  {name:20}{r.minutes:>9.1f}{r.share:>7.1%}{int(r.repetitions):>7}{int(r.datasets):>9}"
                f"{int(r.subjects):>9}"
            )


def print_coverage(table: pd.DataFrame):
    """Per dataset: gestures some subjects lack, and gestures without test (or train) subjects."""
    print("\nCOVERAGE PER DATASET (gestures some subjects lack; gestures absent from a split)")
    for name, t in table[table["gesture_id"] > 0].groupby("dataset", sort=False):
        n_subjects = t["subject"].nunique()
        per = t.groupby("gesture")["subject"].nunique()
        lacking = {g: n_subjects - n for g, n in per.items() if n < n_subjects}
        splits = {s: set(t.loc[t["split"] == s, "gesture"]) for s in ("train", "test")}
        everything = set(t["gesture"])
        line = f"  {name:14} {n_subjects} subjects"
        line += (
            f"; lacking: {', '.join(f'{g} ({n})' for g, n in lacking.items())}"
            if lacking
            else "; every gesture in every subject"
        )
        print(line)
        for split, present in splits.items():
            if t["split"].isna().all():
                print("    no split column (rebuild the processed stage)")
                break
            if present and everything - present:
                print(f"    not in {split}: {', '.join(sorted(everything - present))}")
            elif not present:
                print(f"    no {split} subjects built")


def datasets(table: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(table["dataset"]))  # in the order asked for


def heatmap(ax, values: np.ndarray, rows: list[str], columns: list[str], text, vmax: float, log: bool = False):
    shown = np.log1p(values) if log else values
    top = np.log1p(vmax) if log else vmax
    ax.imshow(np.ma.masked_invalid(shown), cmap="Blues", vmin=0, vmax=top * 1.1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(columns)):
            v = values[i, j]
            if np.isfinite(v) and v > 0:
                dark = shown[i, j] > top * 0.6
                ax.text(j, i, text(v), ha="center", va="center", fontsize=7, color="white" if dark else "#2b2b2b")
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(rows)), rows)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_families(table: pd.DataFrame, path: Path):
    """Each family's share of a dataset's gesture time (rest apart), and of the pooled time."""
    moving = table[table["gesture_id"] > 0]
    minutes = moving.pivot_table(index="family", columns="dataset", values="seconds", aggfunc="sum") / 60
    minutes = minutes.reindex(index=[f for f in FAMILY_ORDER if f in minutes.index], columns=datasets(table))
    minutes["all"] = minutes.sum(axis=1)
    share = (minutes / minutes.sum()).to_numpy()
    style = {"font.size": 8.5}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(1.6 + 1.1 * minutes.shape[1], 0.9 + 0.32 * len(minutes)))
        heatmap(ax, share, list(minutes.index), list(minutes.columns), lambda v: f"{v:.0%}", np.nanmax(share))
        ax.axvline(minutes.shape[1] - 1.5, color="white", lw=3)
        ax.set_title(
            "Gesture time per family, share of each dataset\n(rest apart; all: the datasets pooled)",
            loc="left",
            fontsize=9.5,
            fontweight="bold",
            pad=22,
        )
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)


def plot_gestures(table: pd.DataFrame, path: Path):
    """Minutes of every shared gesture per dataset, grouped by family; a gesture in several datasets is marked."""
    moving = table[table["gesture_id"] > 0]
    minutes = moving.pivot_table(index="gesture", columns="dataset", values="seconds", aggfunc="sum") / 60
    family = dict(zip(vocabulary(), families(), strict=True))
    order = sorted(minutes.index, key=lambda g: (FAMILY_ORDER.index(family[g]), vocabulary().index(g)))
    minutes = minutes.reindex(index=order, columns=datasets(table))
    n_datasets = minutes.notna().sum(axis=1)
    labels = [f"{g}  [{n}]" if n > 1 else g for g, n in zip(order, n_datasets, strict=True)]
    style = {"font.size": 8}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(2.6 + 1.0 * minutes.shape[1], 1.0 + 0.19 * len(order)))
        values = minutes.to_numpy()
        heatmap(ax, values, labels, list(minutes.columns), lambda v: f"{v:.0f}" if v >= 1 else "<1", np.nanmax(values),
                log=True)  # fmt: skip
        previous = None
        for i, g in enumerate(order):  # a line between families, the family named on the right
            if family[g] != previous:
                if previous is not None:
                    ax.axhline(i - 0.5, color="white", lw=2.5)
                ax.text(len(minutes.columns) - 0.4, i - 0.4, family[g], fontsize=7, color="#6b6b6b", va="top")
                previous = family[g]
        ax.set_title(
            "Minutes per shared gesture and dataset\n(log colour; [n]: in n datasets)",
            loc="left",
            fontsize=9.5,
            fontweight="bold",
            pad=22,
        )
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--datasets", nargs="+", help="default: discrete_datasets in --params")
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    parser.add_argument("--detail", action="store_true", help="every gesture in the pooled table")
    args = parser.parse_args()

    names = args.datasets or yaml.safe_load(args.params.read_text())["discrete_datasets"]
    table = load(args.processed, names)
    if table.empty:
        raise SystemExit(f"no gestures.parquet for {names} in {args.processed}")
    print_datasets(table)
    print_pooled(table, args.detail)
    print_coverage(table)

    args.out.mkdir(parents=True, exist_ok=True)
    plot_families(table, args.out / "gesture_balance_families.png")
    plot_gestures(table, args.out / "gesture_balance_gestures.png")
    print(f"\nsaved gesture_balance_families.png and gesture_balance_gestures.png in {args.out}")


if __name__ == "__main__":
    main()
