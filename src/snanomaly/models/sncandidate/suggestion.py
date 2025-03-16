from typing import Optional

from attrs import define, field


@define
class Suggestion:
    """
    Represents a suggestion for plotting (e.g.: exclude data from wavelengths greater than `x` Angstroms).
    """

    above: Optional[float] = field(default=None)
    below: Optional[float] = field(default=None)
    range: Optional[list] = field(default=None)
