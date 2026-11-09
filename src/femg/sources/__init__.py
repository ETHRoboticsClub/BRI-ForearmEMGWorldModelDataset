from femg.sources.base import Source
from femg.sources.emg_fk import EMGFK
from femg.sources.ninapro import NinaProDB2, NinaProDB8

SOURCES: dict[str, type[Source]] = {s.name: s for s in (EMGFK, NinaProDB2, NinaProDB8)}


def get_source(name: str) -> Source:
    if name not in SOURCES:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(SOURCES)}")
    return SOURCES[name]()
