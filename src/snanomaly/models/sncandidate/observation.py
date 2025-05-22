from typing import Any, Optional

import numpy as np
from attrs import define, field
from attrs.converters import optional


def validate_not_true(instance: Any, attribute: Any, value: Any) -> None:
    if value is True:
        raise ValueError(f"{attribute.name} cannot be True for valid observations")


def validate_must_be_none(instance: Any, attribute: Any, value: Any) -> None:
    if value is not None:
        raise ValueError(f"{attribute.name} must be None for valid observations")


def validate_non_negative_finite_or_nan(instance: Any, attribute: Any, value: Any) -> None:
    if np.isnan(value):
        return
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{attribute.name} must be a non-negative finite number")


@define
class Observation:
    """
    Represents the base of an observation (e.g.: photometric/spectra measurement).
    """

    source: str = field()
    time: float = field(
        converter=optional(lambda x: np.mean([float(t) for t in x]) if isinstance(x, list) else float(x)),
    )
    e_time: Optional[float] = field(default=np.nan, validator=validate_non_negative_finite_or_nan)
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
    host: Optional[bool] = field(default=None, validator=validate_not_true)
    includeshost: Optional[bool] = field(default=None)
    model: Optional[str] = field(default=None, validator=validate_must_be_none)
    realization: Optional[int] = field(default=None, validator=validate_must_be_none)
