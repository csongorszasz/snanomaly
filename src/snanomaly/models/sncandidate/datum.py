from attrs import Converter, define, field
from attrs.validators import instance_of, optional

from snanomaly.models.sncandidate.correlation import Correlation


def comma_separated_integers(value: str) -> list[int]:
    if isinstance(value, list):
        return value
    try:
        return list(map(int, value.split(",")))
    except ValueError as ex:
        raise ValueError(
            f"could not convert string to list of integers: {value}",
        ) from ex

@define
class Datum:
    """
    Represents the base of a datum (data quantity).
    """

    source: list[int] = field(converter=Converter(comma_separated_integers))
    e_value: float = field(factory=float, converter=float, validator=optional(instance_of(float)))
    e_lower_value: float = field(default=-float("inf"), converter=float, validator=optional(instance_of(float)))
    e_upper_value: float = field(default=float("inf"), converter=float, validator=optional(instance_of(float)))
    lowerlimit: bool = field(default=False, converter=bool, validator=optional(instance_of(bool)))
    upperlimit: bool = field(default=False, converter=bool, validator=optional(instance_of(bool)))
    u_value: str = field(factory=str, converter=str, validator=optional(instance_of(str)))
    u_e_value: str = field(factory=str, converter=str, validator=optional(instance_of(str)))
    correlation: Correlation = field(default=None)
    kind = field(factory=str, converter=str, validator=optional(instance_of(str)))
    derived: bool = field(default=False, converter=bool, validator=optional(instance_of(bool)))
    description: str = field(factory=str, converter=str, validator=optional(instance_of(str)))
    probability: float = field(factory=float, converter=float, validator=optional(instance_of(float)))
    model: list[int] = field(factory=list, converter=Converter(comma_separated_integers))
    realization: int = field(factory=int, converter=int, validator=optional(instance_of(int)))
