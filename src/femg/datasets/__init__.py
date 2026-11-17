from femg.datasets.base import Dataset, DatasetConfig, Session, first_percent
from femg.datasets.emg_fk import EMGFK
from femg.datasets.great import GREAT
from femg.datasets.ninapro import NinaProDB2, NinaProDB7, NinaProDB8

DATASETS = {
    "emg_fk": EMGFK,
    "ninapro_db2": NinaProDB2,
    "ninapro_db8": NinaProDB8,
    "ninapro_db7": NinaProDB7,
    "great": GREAT,
}

__all__ = ["DATASETS", "Dataset", "DatasetConfig", "Session", "first_percent"]
