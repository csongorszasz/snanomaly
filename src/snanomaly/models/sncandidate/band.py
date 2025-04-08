import attrs
import numpy as np
from attrs import define, field


@define(repr=False)
class Band:
    """
    Represents a band of photometry data.
    """

    _name = field(default=None)
    time: np.array = field(default=np.array([], dtype=np.float64))
    e_time: np.array = field(default=np.array([], dtype=np.float64))
    flux: np.array = field(default=np.array([], dtype=np.float64))
    e_flux: np.array = field(default=np.array([], dtype=np.float64))
    upperlimit: np.array = field(default=np.array([], dtype=bool))
    _is_binned = field(default=False)

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

    @property
    def is_binned(self):
        return self._is_binned

    @is_binned.setter
    def is_binned(self, value: bool):
        self._is_binned = value

    @classmethod
    def get_public_field_names(cls):
        """
        Returns a list of all public field names in the Band class.
        """
        return [field.name for field in attrs.fields(cls) if not field.name.startswith("_")]

    def binned(self, bin_width: int, discrete_time: bool = True):
        """
        Returns a binned version of the band.
        """
        from snanomaly.preprocessing.binning import Binning
        return Binning(self, bin_width, discrete_time)()

    def __repr__(self):
        return f"Band({self.name}, {self.nr_observations} observations)"
