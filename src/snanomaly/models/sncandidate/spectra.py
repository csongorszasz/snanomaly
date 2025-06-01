from __future__ import annotations

from attrs import define, field

from snanomaly.models.sncandidate.spectraobs import SpectraObs


@define
class Spectra:
    raw_observations: list[SpectraObs] = field(default=None)
