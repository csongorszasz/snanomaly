from typing import Optional

import numpy as np
from attrs import define, field
from attrs.converters import optional


@define
class Observation:
    """
    Represents the base of an observation (e.g.: photometric/spectra measurement).
    """

    source: str = field()
    time: Optional[float | list] = field(
        default=None,
        converter=optional(lambda x: (float(x[0]) + float(x[1])) / 2 if isinstance(x, list) else float(x)),
    )  # TODO: check for presence, if not present -> skip
    e_time: Optional[float] = field(default=np.nan)
    e_lower_time: Optional[float] = field(default=np.nan)
    e_upper_time: Optional[float] = field(default=np.nan)
    u_time: Optional[str] = field(default=None)
    survey: Optional[str] = field(default=None)
    instrument: Optional[str] = field(default=None)
    telescope: Optional[str] = field(default=None)
    observatory: Optional[str] = field(default=None)
    observer: Optional[str] = field(default=None)
    reducer: Optional[str] = field(default=None)
    airmass: Optional[float] = field(default=None)
    host: Optional[bool] = field(default=None)  # TODO: check for presence, if present -> skip
    includeshost: Optional[bool] = field(default=None)
    model: Optional[str] = field(default=None) # TODO: check for presence, if present -> skip
    realization: Optional[int] = field(default=None) # TODO: check for presence, if present -> skip
