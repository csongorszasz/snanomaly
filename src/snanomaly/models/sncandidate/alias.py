from attrs import define, field

from snanomaly.models.sncandidate.stringdatum import StringDatum


@define
class Alias(StringDatum):

    def __repr__(self):
        return f"{self.value}"