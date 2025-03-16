from attrs import define, field

from snanomaly.models.sncandidate.datum import Datum


@define
class StringDatum(Datum):
    """
    A Datum with value of string type.
    """

    value: str = field(default=None)
