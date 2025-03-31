from typing import Optional

import numpy as np
from attrs import define, field

from snanomaly.models.sncandidate.observation import Observation


@define
class PhotometryObs(Observation):
    """
    Represents a photometric observation.

    Note:
        Radio and X-ray specific fields are not included as current supernova datasets do not contain radio/X-ray data.

    """

    countrate: Optional[float] = field(default=None)
    e_countrate: Optional[float] = field(default=np.nan)
    e_lower_countrate: Optional[float] = field(default=np.nan)
    e_upper_countrate: Optional[float] = field(default=np.nan)

    magnitude: Optional[float] = field(default=None)
    flux: Optional[float] = field(default=None)
    e_flux: Optional[float] = field(default=np.nan)  # TODO: set in post init
    e_magnitude: Optional[float] = field(default=np.nan)
    e_lower_magnitude: Optional[float] = field(default=np.nan)
    e_upper_magnitude: Optional[float] = field(default=np.nan)
    zeropoint: Optional[float] = field(default=None)
    band: Optional[str] = field(default=None) # TODO: check for presence, if not present -> skip
    bandset: Optional[str] = field(default=None) # TODO: check if `band` is in `bandset`, if not -> skip
    system: Optional[str] = field(default=None)
    upperlimit: Optional[bool] = field(default=False)
    upperlimitsigma: Optional[float] = field(default=None)
    kcorrected: Optional[bool] = field(default=None)
    scorrected: Optional[bool] = field(default=None)
    mcorrected: Optional[bool] = field(default=None)

    def __attrs_post_init__(self):
        self._init_magnitude_and_flux()

    def _init_magnitude_and_flux(self):
        """
        Initialize the magnitude and flux by deriving from available data (e.g.: countrate, zeropoint, flux density).

        Todo:
            - convert using zero-point
            - convert count rate to flux
            - convert flux density to flux
            - factor in errors

        """
        if self.zeropoint is None:
            self.zeropoint = 1  # standard flux for convenience (not physically meaningful) # TODO: fact check
        if self.magnitude is not None and self.flux is None:
            self.flux = self._flux_from_magnitude(self.magnitude, self.zeropoint)
        elif self.magnitude is None and self.flux is not None:
            self.magnitude = self._magnitude_from_flux(self.flux, self.zeropoint)
        # TODO: finish the rest of the conversions (e.g.: e_flux)

    @staticmethod
    def _flux_from_magnitude(mag: float, zp: float) -> float:
        return zp * 10 ** (-0.4 * mag)

    @staticmethod
    def _magnitude_from_flux(flux: float, zp: float) -> float:
        return -2.5 * np.log10(flux / zp)


