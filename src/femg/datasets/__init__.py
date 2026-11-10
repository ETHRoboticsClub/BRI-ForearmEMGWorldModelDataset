"""All available datasets. To add one, import its class here and add it to DATASETS."""

from femg.datasets.base import Dataset, Session, first_percent
from femg.datasets.emg_fk import EMGFK
from femg.datasets.ninapro import NinaProDB2, NinaProDB8

DATASETS = {
    "emg_fk": EMGFK,
    "ninapro_db2": NinaProDB2,
    "ninapro_db8": NinaProDB8,
}

__all__ = ["DATASETS", "Dataset", "Session", "first_percent"]
