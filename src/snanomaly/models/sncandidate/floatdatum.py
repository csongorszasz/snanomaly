from attrs import define, field

from snanomaly.models.sncandidate.datum import Datum


@define
class FloatDatum(Datum):
    """
    A Datum with value of float type.
    """

    value: float = field(default=None)
