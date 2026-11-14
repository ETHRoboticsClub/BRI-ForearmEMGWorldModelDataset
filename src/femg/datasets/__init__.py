from femg.datasets.base import Dataset, Session, first_percent
from femg.datasets.emg_fk import EMGFK
from femg.datasets.great import GREAT
from femg.datasets.kin_mus_uji import KinMusUJI
from femg.datasets.ninapro import NinaProDB2, NinaProDB3, NinaProDB4, NinaProDB7, NinaProDB8
from femg.datasets.seeds import SEEDS

DATASETS = {
    "emg_fk": EMGFK,
    "ninapro_db2": NinaProDB2,
    "ninapro_db8": NinaProDB8,
    "ninapro_db7": NinaProDB7,
    "ninapro_db3": NinaProDB3,
    "ninapro_db4": NinaProDB4,
    "seeds": SEEDS,
    "kin_mus_uji": KinMusUJI,
    "great": GREAT,
}

__all__ = ["DATASETS", "Dataset", "Session", "first_percent"]
