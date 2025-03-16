from typing import Optional

from attrs import define, field


@define
class Correlation:
    """
    Represents the correlation of a datum with another datum.
    """

    quantity: str = field()
    value: str | float = field()
    kind: Optional[str] = field(default=None)
    derived: Optional[bool] = field(default=None)
