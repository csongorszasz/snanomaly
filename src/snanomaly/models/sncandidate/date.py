from datetime import datetime

from attrs import define, field

from snanomaly.models.sncandidate.stringdatum import StringDatum


@define
class Date(StringDatum):
    """
    Represents a date in the format YYYY/MM/DD.
    """

    date_value: datetime = field(init=False)

    def __attrs_post_init__(self):
        """
        Post-initialization processing to convert the date string into a datetime object.
        """
        try:
            self.date_value = datetime.strptime(self.value, "%Y/%m/%d")
        except ValueError as ex:
            raise ValueError(f"Invalid date format: {self.value}") from ex
