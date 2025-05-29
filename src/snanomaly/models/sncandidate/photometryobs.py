from typing import Optional

import numpy as np
from attrs import define, field, validators

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
    e_flux: Optional[float] = field(default=np.nan, init=False)
    e_magnitude: Optional[float] = field(default=0, validator=validators.ge(0))
    e_lower_magnitude: Optional[float] = field(default=0, validator=validators.optional(validators.ge(0)))
    e_upper_magnitude: Optional[float] = field(default=0, validator=validators.optional(validators.ge(0)))
    zeropoint: Optional[float] = field(default=None)
    band: Optional[str] = field(default=None) # TODO: make required
    bandset: Optional[str] = field(default=None) # TODO: handle differences between band set systems (e.g.: AB vs UBVRI vs VEGA)
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

        """
        if self.zeropoint is None:
            self.zeropoint = 1  # standard flux for convenience (not physically meaningful) # TODO: fact check
        if self.magnitude is not None and self.flux is None:
            self.flux = self._flux_from_magnitude(self.magnitude, self.zeropoint)
        elif self.magnitude is None and self.flux is not None:
            self.magnitude = self._magnitude_from_flux(self.flux, self.zeropoint)

        # self._init_errors() # TODO: reenable

    @staticmethod
    def _flux_from_magnitude(mag: float, zp: float) -> float:
        return zp * 10 ** (-0.4 * mag)

    @staticmethod
    def _magnitude_from_flux(flux: float, zp: float) -> float:
        return -2.5 * np.log10(flux / zp)

    def _init_errors(self):
        if not np.isnan(self.e_lower_magnitude) and not np.isnan(self.e_upper_magnitude):
            e_lower_flux = self._flux_from_magnitude(self.magnitude + self.e_lower_magnitude, self.zeropoint)
            e_upper_flux = self._flux_from_magnitude(self.magnitude - self.e_upper_magnitude, self.zeropoint)
            self.e_flux = 0.5 * (e_upper_flux - e_lower_flux)
            if not np.isfinite(self.e_flux):
                raise ValueError("e_flux is not finite")
        elif not np.isnan(self.e_magnitude):
            self.e_flux = 0.4 * np.log(10) * self.flux * self.e_magnitude
            if not np.isfinite(self.e_flux):
                raise ValueError("e_flux is not finite")


# cattrs.structure({"source": "a", "time": 1, "magnitude": 1}, PhotometryObs)
# cattrs.structure({"source": "a", "time": 1, "magnitude": 1, "band": "V"}, PhotometryObs)

