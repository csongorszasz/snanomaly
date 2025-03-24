from typing import Optional

import numpy as np
from attrs import define, field

from snanomaly.models.sncandidate.observation import Observation
from snanomaly.models.sncandidate.suggestion import Suggestion


@define
class SpectraObs(Observation):
    """
    Represents a spectra observation.
    """

    data: np.ndarray = field(default=None)
    u_wavelengths: str = field(default=None)
    u_fluxes: str = field(default=None)
    u_errors: Optional[str] = field(default=None)
    snr: Optional[float] = field(default=None)
    filename: Optional[str] = field(default=None)
    deredshifted: Optional[bool] = field(default=None)
    dereddened: Optional[bool] = field(default=None)
    vacuumwavelengths: Optional[bool] = field(default=None)
    exclude: Optional[Suggestion] = field(default=None)
