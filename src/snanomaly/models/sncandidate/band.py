import attrs
import numpy as np
from attrs import define, field


@define
class Band:
    """
    Represents a band of photometry data.
    """

    _name = field(init=False)
    time: np.array = field(default=np.array([]))
    e_time: np.array = field(default=np.array([]))
    flux: np.array = field(default=np.array([]))
    e_flux: np.array = field(default=np.array([]))
    upperlimit: bool = field(default=False)

    @property
    def matrix(self):
        """
        Returns the band data as a 2D matrix.
        """
        return np.vstack([self.time, self.e_time, self.flux, self.e_flux])

    @property
    def nr_observations(self):
        """
        Returns the number of observations in the band.
        """
        return len(self.time)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str):
        """
        Sets the name of the band.
        """
        self._name = value

    @classmethod
    def get_public_field_names(cls):
        """
        Returns a list of all public field names in the Band class.
        """
        return [field.name for field in attrs.fields(cls) if not field.name.startswith("_")]
