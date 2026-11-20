from femg.datasets.base import Dataset, DatasetConfig, Session, first_percent
from femg.datasets.capgmyo import CapgMyoDBa
from femg.datasets.emg_epn612 import EMGEPN612
from femg.datasets.emg_fk import EMGFK
from femg.datasets.fors_emg import FORSEMG
from femg.datasets.grabmyo import GRABMyo
from femg.datasets.great import GREAT
from femg.datasets.ninapro import NinaProDB2, NinaProDB3, NinaProDB4, NinaProDB5, NinaProDB7, NinaProDB8
from femg.datasets.putemg import PutEMG

DATASETS = {
    "emg_fk": EMGFK,
    "ninapro_db2": NinaProDB2,
    "ninapro_db8": NinaProDB8,
    "ninapro_db7": NinaProDB7,
    "great": GREAT,
    "ninapro_db5": NinaProDB5,
    "ninapro_db3": NinaProDB3,
    "ninapro_db4": NinaProDB4,
    "fors_emg": FORSEMG,
    "grabmyo": GRABMyo,
    "emg_epn612": EMGEPN612,
    "putemg": PutEMG,
    "capgmyo_dba": CapgMyoDBa,
}

__all__ = ["DATASETS", "Dataset", "DatasetConfig", "Session", "first_percent"]
