from __future__ import annotations

import numpy as np
from attrs import define, field

from snanomaly.models.sncandidate.band import Band


@define
class Binning:
    """
    Binning of photometry data for a given band.
    """

    bin_width: float = field(default=None)
    bins: np.ndarray = field(default=None)
    bin_centers: np.ndarray = field(default=None)
    bin_edges: np.ndarray = field(default=None)
    bin_counts: np.ndarray = field(default=None)
    band: Band = field(default=None)

    def __attrs_post_init__(self):
        if self.band is not None:
            self.bins = np.arange(
                self.band.time.min(),
                self.band.time.max() + self.bin_width,  # ensuring the last data point is included
                self.bin_width,
            )
            self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
            self.bin_edges = np.vstack([self.bins[:-1], self.bins[1:]]).T
            self.bin_counts, _ = np.histogram(self.band.time, bins=self.bins)
