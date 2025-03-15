from attrs import define, field

from snanomaly.models.sncandidate.stringdatum import StringDatum


@define
class RA(StringDatum):
    """
    Represents the right ascension (RA) of a celestial object.
    """

    hours: int = field(default=0, init=False)
    minutes: int = field(default=0, init=False)
    seconds: int = field(default=0, init=False)

    def __attrs_post_init__(self):
        """
        Post-initialization processing to convert the RA string into hours, minutes, and seconds.
        """
        if self.value:
            try:
                hms = self.value.split(":")
                self.hours = int(hms[0])
                self.minutes = int(hms[1])
                self.seconds = int(hms[2])
            except (ValueError, IndexError) as ex:
                raise ValueError(f"Invalid RA format: {self.value}") from ex
