from __future__ import annotations

import numpy as np
from attrs import define, field

from snanomaly.models.sncandidate.band import Band


@define
class Binning:
    """
    Binning of photometry data for a given band.

    Creates a new `Band` object.
    """

    band: Band = field()
    bin_width: float = field()
    discrete_time: bool = field(default=True)
    bins: np.ndarray = field(init=False)

    def __attrs_post_init__(self):
        if not self.discrete_time:
            raise NotImplementedError("Continuous time binning is not implemented.")
        self.bins = np.unique(self.band.time // self.bin_width * self.bin_width)

    @staticmethod
    def _mean_sigma(y: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
        """Calculate weighted mean and standard error."""
        if y.size == 1:
            return y[0], np.nan
        # TODO : Check if this is the correct way to handle zero errors
        # if np.sum(weight) == 0:  # Check for zero-sum weights
        #     logger.trace("Weights sum to zero in _mean_sigma. Returning NaN.")
        #     return np.nan, np.nan
        mean, sum_weight = np.average(y, weights=weight, returned=True)
        sgm_mean = np.sqrt(np.sum(weight * (y - mean) ** 2) / sum_weight / (y.size - 1))
        return mean, sgm_mean

    @staticmethod
    def _typical_err(err: np.ndarray) -> float:
        """Calculate typical error."""
        return 1 / np.sqrt(np.sum(1 / err**2))

    @staticmethod
    def _mean_error(y: np.ndarray, err: np.ndarray) -> tuple[float, float]:
        """Calculate weighted mean and appropriate error."""
        if y.size == 1:
            return y[0], err[0]
        weight = 1 / err**2
        # TODO : Check if this is the correct way to handle zero errors
        # if np.sum(weight) == 0:  # Additional check for zero-sum weights
        #     logger.trace("Weights sum to zero in _mean_error. Returning NaN.")
        #     return np.nan, np.nan
        mean, sgm_mean = Binning._mean_sigma(y, weight)
        typical_err = Binning._typical_err(err)
        if typical_err > sgm_mean:
            return mean, 0.5 * (sgm_mean + typical_err)
        return mean, sgm_mean

    def _do_binning(self) -> Band:
        """
        Calculate aggregated statistics for each bin.
        Does not modify the original band. Returns a new Band object with binned data.
        """
        binned_flux = []
        binned_e_flux = []
        binned_upperlimit = []

        bin_centers = self.bins + self.bin_width / 2
        for i in range(len(bin_centers)):
            bin_start = self.bins[i]
            bin_end = bin_start + self.bin_width if i < len(self.bins) - 1 else np.inf

            # Get data points in this bin
            mask = (self.band.time >= bin_start) & (self.band.time < bin_end)
            flux = self.band.flux[mask]
            e_flux = self.band.e_flux[mask]
            is_upperlimit = self.band.upperlimit[mask]

            if np.all(is_upperlimit):
                # If all points are upper limits, take the one with the lowest flux
                best = np.argmin(flux)
                binned_flux.append(flux[best])
                binned_e_flux.append(e_flux[best])
                binned_upperlimit.append(True)
            elif not np.any(np.isfinite(e_flux)):
                # If no errors are available, take the mean of the fluxes
                no_upperlimit = np.where(~is_upperlimit)[0]
                binned_flux.append(np.mean(flux[no_upperlimit]))
                binned_e_flux.append(np.nan)
                binned_upperlimit.append(False)
            else:
                # Calculate the mean and error for the fluxes
                sample = np.where(np.logical_not(is_upperlimit) & np.isfinite(e_flux))
                flux, e_flux = self._mean_error(flux[sample], e_flux[sample])
                binned_flux.append(flux)
                binned_e_flux.append(e_flux)
                binned_upperlimit.append(False)

        return Band(
            name=self.band.name,
            is_binned=True,
            time=bin_centers,
            e_time=np.array(0.5 * self.bin_width),
            flux=np.array(binned_flux),
            e_flux=np.array(binned_e_flux),
            upperlimit=self.band.upperlimit,
        )

    def __call__(self) -> Band:
        return self._do_binning()


__all__ = ["Binning"]
