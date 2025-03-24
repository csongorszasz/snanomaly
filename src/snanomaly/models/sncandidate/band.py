import numpy as np
from attrs import define, field


@define
class Band:
    """
    Represents a band of photometry data.
    """

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
