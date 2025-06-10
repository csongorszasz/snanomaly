from __future__ import annotations

import attrs
import numpy as np
from attrs import define, field
from loguru import logger


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
    _is_upperlimits_converted = field(default=False)
    _ignored_upperlimits_time: np.array = field(default=np.array([], dtype=np.float64))
    _ignored_upperlimits_flux: np.array = field(default=np.array([], dtype=np.float64))
    _is_normalized = field(default=False)
    _norm_factor = field(default=None)

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
        self._name = value

    @property
    def is_binned(self):
        return self._is_binned

    @is_binned.setter
    def is_binned(self, value: bool):
        self._is_binned = value

    @property
    def norm_factor(self):
        return self._norm_factor

    @norm_factor.setter
    def norm_factor(self, value: float):
        self._norm_factor = value

    @property
    def is_upperlimits_converted(self):
        return self._is_upperlimits_converted

    @is_upperlimits_converted.setter
    def is_upperlimits_converted(self, value: bool):
        self._is_upperlimits_converted = value

    @property
    def ignored_upperlimits_time(self):
        return self._ignored_upperlimits_time

    @ignored_upperlimits_time.setter
    def ignored_upperlimits_time(self, value: np.ndarray):
        self._ignored_upperlimits_time = value

    @property
    def ignored_upperlimits_flux(self):
        return self._ignored_upperlimits_flux

    @ignored_upperlimits_flux.setter
    def ignored_upperlimits_flux(self, value: np.ndarray):
        self._ignored_upperlimits_flux = value

    @property
    def is_normalized(self):
        return self._is_normalized

    @classmethod
    def get_public_field_names(cls):
        """
        Returns a list of all public field names in the Band class.
        """
        return [field.name for field in attrs.fields(cls) if not field.name.startswith("_")]

    def normalize(self):
        """
        Normalizes the flux and error in flux of the band.
        """
        if self.flux.size == 0:
            logger.warning("Band has no flux data to normalize.")
            return

        max_flux = np.max(self.flux)
        if max_flux == 0:
            logger.warning("Band has zero (0) maximum flux, normalization skipped.")
            return

        self._norm_factor = max_flux
        self.flux /= max_flux
        self.e_flux /= max_flux
        self._is_normalized = True

    def denormalize(self):
        """
        Denormalizes the flux and error in flux of the band.
        """
        if not self._is_normalized or self._norm_factor is None:
            logger.warning("Band is not normalized or normalization factor is not set.")
            return

        self.flux *= self._norm_factor
        self.e_flux *= self._norm_factor
        self._is_normalized = False

    def binned(self, bin_width: int, discrete_time: bool = True) -> Band:
        """
        Returns a binned version of the band.
        """
        from snanomaly.preprocessing.binning import Binning
        return Binning(self, bin_width, discrete_time)()

    def process_upper_limits(self) -> Band:
        """
        Only keep upper limits that are either earlier than the earliest real detection or later than the latest real
        detection.

        Convert the kept upper limits to real observations by assigning them to `0` with a `3 * upperlimit` error.
        """
        min_real_time = self.time[~self.upperlimit].min()
        max_real_time = self.time[~self.upperlimit].max()
        keep_condition = self.upperlimit & ((self.time < min_real_time) | (self.time > max_real_time))
        upperlimit_indices_to_keep = np.where(keep_condition)[0]
        if upperlimit_indices_to_keep.size == 0:
            return self.filter_by_condition(~self.upperlimit)

        self.e_flux[upperlimit_indices_to_keep] = 3 * self.flux[upperlimit_indices_to_keep]
        self.flux[upperlimit_indices_to_keep] = 0

        # remove the rest of the upper limits
        all_indices_to_keep = np.sort(
            np.concatenate((upperlimit_indices_to_keep, np.where(~self.upperlimit)[0])),
        )
        self._ignored_upperlimits_time = self.time[~all_indices_to_keep]
        self._ignored_upperlimits_flux = self.flux[~all_indices_to_keep]
        self._is_upperlimits_converted = True
        return self.filter_by_indices(all_indices_to_keep)

    def filter_by_indices(self, indices_to_keep: np.ndarray) -> Band:
        self.time = self.time[indices_to_keep]
        self.e_time = self.e_time[indices_to_keep]
        self.flux = self.flux[indices_to_keep]
        self.e_flux = self.e_flux[indices_to_keep]
        self.upperlimit = self.upperlimit[indices_to_keep]
        return self

    def filter_by_condition(self, cond) -> Band:
        self.time = self.time[cond]
        self.e_time = self.e_time[cond]
        self.flux = self.flux[cond]
        self.e_flux = self.e_flux[cond]
        self.upperlimit = self.upperlimit[cond]
        return self

    def __repr__(self):
        return f"Band({self.name}, {self.nr_observations} observations)"
