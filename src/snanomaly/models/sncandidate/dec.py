from attrs import define, field

from snanomaly.models.sncandidate.stringdatum import StringDatum


@define
class Dec(StringDatum):
    """
    Represents the declination (Dec) of a celestial object.
    """

    degrees: int = field(init=False)
    minutes: int = field(init=False)
    seconds: float = field(init=False)


    def __attrs_post_init__(self):
        """
        Post-initialization processing to convert the Dec string into degrees.
        """
        sign = -1 if self.value.startswith("-") else 1
        try:
            dms = self.value.split(":")
            self.degrees = sign * int(dms[0])
            self.minutes = int(dms[1])
            self.seconds = float(dms[2])
        except (ValueError, IndexError) as ex:
            raise ValueError(f"Invalid Dec format: {self.value}") from ex
