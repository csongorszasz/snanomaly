from typing import Optional

from attrs import define, field

from snanomaly.models.sncandidate.correlation import Correlation


@define
class Datum:
    """
    Represents the base of a datum (data quantity).
    """

    source: str = field()
    e_value: Optional[float] = field(default=None)
    e_lower_value: Optional[float] = field(default=None)
    e_upper_value: Optional[float] = field(default=None)
    lowerlimit: Optional[bool] = field(default=None)
    upperlimit: Optional[bool] = field(default=None)
    u_value: Optional[str] = field(default=None)
    u_e_value: Optional[str] = field(default=None)
    correlation: Optional[Correlation] = field(default=None)
    kind: Optional[str] = field(default=None)
    derived: Optional[bool] = field(default=None)
    description: Optional[str] = field(default=None)
    probability: Optional[float] = field(default=None)
    model: Optional[str] = field(default=None)
    realization: Optional[int] = field(default=None)
