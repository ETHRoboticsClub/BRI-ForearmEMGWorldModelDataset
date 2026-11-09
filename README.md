# BRI Forearm EMG World Model Dataset

Public forearm EMG + hand-kinematics datasets combined into one format (one EMG rate, units, filtering, joint
space and metadata), for training a generic EMG → hand-pose model that can transfer to amputees.

**Read [docs/data-spec.md](docs/data-spec.md) first**: requirements, decisions, dataset status and roadmap.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dvc]"
pre-commit install
dvc pull            # the built dataset of this commit (S3 access: see HRI-ContinuousTraining/docs/dvc-access.md)
```

## Build

```bash
femg-build --dataset emg_fk --fraction 0.1 --out data/dev   # dev: 10% of subjects, seconds to minutes
femg-build --fraction 0.3 --out data/dev                    # grow it: the 10% already built is reused
dvc repro                                                   # full build -> data/built (params.yaml)
dvc push && git add dvc.lock && git commit                  # share it
```

- `--fraction` picks subjects with a stable hash. The subsets are nested (0.1 ⊂ 0.3 ⊂ 1.0) and identical between
  the builder and the loader.
- The build resumes: sessions already built with the same `params.yaml` and code are skipped.
- Raw downloads are cached in `data/raw/<dataset>` (`--raw` to point elsewhere). They are not versioned because
  they can be downloaded again.
- On a big CPU machine next to the bucket: `sky launch -c femg-build sky/build.yaml` (see the file).

Output, one folder per session (`emg.npy` (T, C) volts, `kin.npy` (T/r, 22) radians/NaN, `stimulus.npy`,
`meta.json`) plus `<dataset>/sessions.parquet`. Details in `src/femg/store.py`.

## Load (from the training repo)

```python
from femg.loader import EMGWindows

ds = EMGWindows("data/forearm", datasets=["emg_fk", "ninapro_db2", "ninapro_db8"], fraction=0.2,
                window_s=0.48, stride_s=0.08, montage="ring8", normalize="calib_rms")
train, val, test = ds.split_subjects(val=0.1, test=0.1)
loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, num_workers=8)
# batch: emg (B, 8, 240), kin (B, 22), mask (B, 22) -> loss = ((pred - kin) ** 2 * mask).sum() / mask.sum()
```

In `HRI-ContinuousTraining`:

```bash
pip install "femg @ git+ssh://git@github.com/ETHRoboticsClub/BRI-ForearmEMGWorldModelDataset.git"
dvc import git@github.com:ETHRoboticsClub/BRI-ForearmEMGWorldModelDataset data/built -o data/forearm
```

`dvc import` pins the dataset version in the training repo's git history, and `dvc update data/forearm.dvc` moves
it to a newer build. On a machine with lots of RAM (e.g. the GPU box), pass `in_memory=True` to skip mmap page faults.

## Add a dataset

1. `src/femg/sources/<name>.py`: subclass `Source`, fill in the metadata (cite where every value comes from) and
   implement `subjects`, `sessions` (download URLs) and `read` (→ volts, canonical joints in radians).
2. Register it in `src/femg/sources/__init__.py` and add it to `datasets` in `params.yaml`.
3. Open one real file and record what you checked in `verified`. Then `femg-build --dataset <name> --fraction 0.1`.

## Structure

```
params.yaml              build parameters (fs, label_fs, filters, fraction, workers)
dvc.yaml                 one build stage per dataset -> data/built/<dataset>
src/femg/
  schema.py              canonical joints, label sources, Recording/SessionRef
  sources/               one reader per public dataset
  signal.py              resampling, causal filters, stats
  build.py               parallel builder (femg-build)
  store.py               downloads + on-disk layout
  loader.py              EMGWindows: windows on the fly, montages, normalization, subject splits
  subset.py              nested subject fractions
sky/build.yaml           full build on an AWS CPU machine
```
