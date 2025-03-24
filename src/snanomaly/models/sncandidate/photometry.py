from __future__ import annotations

from attrs import define, field

from snanomaly.models.sncandidate.bands import Bands
from snanomaly.models.sncandidate.photometryobs import PhotometryObs


@define
class Photometry:
    raw_observations: list[PhotometryObs] = field(default=None)
    bin_width: float = field(default=None)
    bands: Bands = field(default=None)
