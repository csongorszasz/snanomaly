from typing import Optional

from attrs import define, field


@define
class Source:
    """
    Represents a source from where data was obtained.
    """

    alias: int = field()
    name: str = field(default=None)
    url: Optional[str] = field(default=None)
    bibcode: Optional[str] = field(default=None)
    doi: Optional[str] = field(default=None)
    arxivid: Optional[str] = field(default=None)
    secondary: Optional[bool] = field(default=None)
    acknowledgement: Optional[str] = field(default=None)
