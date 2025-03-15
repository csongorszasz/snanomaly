from attrs import define, field


@define
class Source:
    """
    Represents a source from where data was obtained.
    """

    name: str = field()
    alias: int = field()
    url: str = field(factory=str)
    bibcode: str = field(factory=str)
    doi: str = field(factory=str)
    arxivid: str = field(factory=str)
    secondary: bool = field(factory=bool)
    acknowledgement: str = field(factory=str)
