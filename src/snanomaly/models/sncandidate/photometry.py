from typing import Optional

from attrs import define, field

from snanomaly.models.sncandidate.observation import Observation


@define
class Photometry(Observation):
    """
    Represents a photometric observation.
    """

    countrate: Optional[float] = field(default=None)
    e_countrate: Optional[float] = field(default=None)
    e_lower_countrate: Optional[float] = field(default=None)
    e_upper_countrate: Optional[float] = field(default=None)

    magnitude: float = field(default=None)
    e_magnitude: Optional[float] = field(default=None)
    e_lower_magnitude: Optional[float] = field(default=None)
    e_upper_magnitude: Optional[float] = field(default=None)
    zeropoint: Optional[float] = field(default=None)
    band: Optional[str] = field(default=None)
    bandset: Optional[str] = field(default=None)
    system: Optional[str] = field(default=None)
    upperlimit: Optional[bool] = field(default=None)
    upperlimitsigma: Optional[float] = field(default=None)
    kcorrected: Optional[bool] = field(default=None)
    scorrected: Optional[bool] = field(default=None)
    mcorrected: Optional[bool] = field(default=None)

    # radio and X-ray specific fields are not included as current supernova datasets do not contain radio/X-ray data
