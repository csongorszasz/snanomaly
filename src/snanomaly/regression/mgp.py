from __future__ import annotations

import warnings
from collections.abc import Callable
from enum import Enum

import numpy as np
from attrs import define, field
from loguru import logger
from multistate_kernel import MultiStateKernel
from scipy import optimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel

from snanomaly.models.results.mgp_result import MGPResult
from snanomaly.models.sncandidate import Bandset
from snanomaly.models.sncandidate.band import Band
from snanomaly.models.sncandidate.bands import BandEnum, Bands
from snanomaly.regression.exception import (
    BandNotFoundError,
    PeakTimeNotSetError,
    PredictionIntervalOutOfBoundsError,
)


class Optimizer(Enum):
    L_BFGS_B = "fmin_l_bfgs_b"
    TRUST_CONSTR = "trust-constr"

class PreparedData:
    """
    Represents the data format required by a MultiStateKernel.

    Although the actual peak point might not be available in the train data, find the peak from what is available
    and do the preprocessing relative to that approximate peak.
    """

    def __init__(self, bands: list[Band], peak_band: BandEnum, prediction_interval_from_peak: tuple[int, int]):
        self.bands: list[Band] = self._init_bands(bands)
        self.X: np.ndarray = np.concatenate(
            [np.column_stack((i * np.ones(band.nr_observations), band.time)) for i, band in enumerate(self.bands)],
        )
        self.y: np.array = np.concatenate([band.flux for band in self.bands])
        self.err: np.array = np.concatenate([band.e_flux for band in self.bands])

        # self.norm_factor = (self.y.mean(), self.y.std())
        self.norm_factor = (0, self.y.max())

        self.y = self.normalize(self.y)

        self._keep_interval_relative_to_peak(peak_band, prediction_interval_from_peak)

    def normalize(self, y: np.array) -> np.array:
        return (y - self.norm_factor[0]) / self.norm_factor[1]

    def denormalize(self, y: np.array) -> np.array:
        return y * self.norm_factor[1] + self.norm_factor[0]

    @property
    def bandset(self) -> str:
        return "".join([b.name for b in self.bands])

    def _init_bands(self, bands: list[Band]) -> list[Band]:
        bands_binned = [band.binned(bin_width=1) if not band.is_binned else band for band in bands]
        # TODO: filter
        # TODO: Handle upper limits
        # TODO: filter again
        return bands_binned

    def _keep_interval_relative_to_peak(self, peak_band: BandEnum, prediction_interval_from_peak: tuple[int, int]) -> None:
        """Discards points outside the target interval relative to the peak."""
        peak_time = self.find_peak_time(peak_band)
        mask = ((self.X[:, 1] >= peak_time + prediction_interval_from_peak[0]) &
                (self.X[:, 1] <= peak_time + prediction_interval_from_peak[1]))
        self.X = self.X[mask]
        self.y = self.y[mask]
        self.err = self.err[mask]

    def find_peak_time(self, peak_band: BandEnum) -> float:
        band = filter(lambda band: band.name == peak_band.value, self.bands)
        try:
            band = next(band)
            return band.time[np.argmax(band.flux)]
        except StopIteration:
            raise BandNotFoundError(f"Peak band `{peak_band}` not found in bandset `{self.bandset}`")

@define
class MGPInterpolator:
    """Interpolate light curves in available band sets with Multivariate Gaussian Process Regression."""

    sn_name: str = field()
    bandset: Bandset = field()
    bands: Bands = field()
    peak_band: BandEnum = field()
    prediction_interval_from_peak: tuple[int, int] = field()  # e.g.: (-20,+100)
    normalize_y: bool = field()
    n_restarts_optimizer: int = field()
    kernel: Kernel = field(default=RBF)
    length_scale_min_bounds: tuple[float, float] = field(default=(0, np.inf))
    optimize_method: Optimizer = field(default=Optimizer.L_BFGS_B)
    random_state: int = field(default=None)
    peak_time: float = field(init=False, default=None)
    regressor: GaussianProcessRegressor = field(init=False)
    prepared_bands: PreparedData = field(init=False)

    def __attrs_post_init__(self):
        self.prepared_bands = PreparedData(self.bands.get_bands(self.bandset), self.peak_band,
                                           self.prediction_interval_from_peak)
        self.regressor = self._init_regressor()
        self.peak_time = self._find_peak_time_in_predicted_data()

    def _init_regressor(self) -> GaussianProcessRegressor:
        def construct_regressor(kernel_min_bound_offset: float) -> GaussianProcessRegressor:
            kernels = self._init_kernels(kernel_min_bound_offset)
            scale, scale_bounds = self._init_scale_matrix()
            ms_kernel = MultiStateKernel(kernels=kernels, scale=scale, scale_bounds=scale_bounds)
            # alpha = self._init_alpha() # TODO
            optimizer = self._init_optimizer()
            return GaussianProcessRegressor(
                kernel=ms_kernel,
                # alpha=alpha, # TODO
                optimizer=optimizer,
                n_restarts_optimizer=self.n_restarts_optimizer,
                # normalize_y=self.normalize_y, # TODO
                normalize_y=False, # TODO
                random_state=self.random_state,
            )

        def search_optimal_min_length_scale(offset_low: float):
            """Find optimal lower bound for kernel length-scale through trial and error."""
            kernel_min_bound_offset = offset_low
            step = 0.1
            while True:
                regressor = construct_regressor(kernel_min_bound_offset)
                try:
                    with warnings.catch_warnings():
                        # Treat ConvergenceWarning as an error
                        warnings.filterwarnings("error", category=ConvergenceWarning)
                        regressor.fit(self.prepared_bands.X, self.prepared_bands.y)
                    return regressor
                except ConvergenceWarning as ex:
                    logger.debug(f"! ConvergenceWarning during MGP fit: {ex}")
                    kernel_min_bound_offset -= step
                    logger.debug(f"Retrying with kernel length-scale lower bound offset = {kernel_min_bound_offset:.2f}")
                    continue

        return search_optimal_min_length_scale(0.0)

    def _init_kernels(self, min_bound_offset: float = 0.0) -> tuple[Kernel]:
        time_diffs = (
            np.max(np.diff(band.time)) if band.nr_observations > 1 else 0 for band in self.bands.get_bands(self.bandset)
        )
        min_bounds = [
            max(self.length_scale_min_bounds[0], min(diff, self.length_scale_min_bounds[1])) for diff in time_diffs
        ]
        return tuple(self.kernel(length_scale_bounds=(max(0.01, min_bound + min_bound_offset), 1e4)) for min_bound in min_bounds)

    def _init_scale_matrix(self) -> tuple[np.ndarray, tuple(np.ndarray, np.ndarray)]:
        """
        Creates a lower triangular matrix that defines the correlation between GP states.
        Returns the scale matrix and its bounds.
        """
        m = np.eye(len(self.bandset.value))
        m[np.tril_indices_from(m, k=-1)] = 0.5

        m_lower_bound = np.zeros_like(m)
        m_lower_bound[np.tril_indices_from(m_lower_bound)] = -1e3
        m_lower_bound[0, 0] = 1e-4

        m_upper_bound = np.zeros_like(m)
        m_upper_bound[np.tril_indices_from(m_upper_bound)] = 1e3
        m_upper_bound[0, 0] = 1e4

        return m, (m_lower_bound, m_upper_bound)

    def _init_alpha(self) -> np.ndarray:
        """
        Returns an array representing the variance of the observational noise in the MGP.
        """
        return self.prepared_bands.err ** 2

    def _init_optimizer(self) -> str | Callable:
        if self.optimize_method == Optimizer.TRUST_CONSTR:
            return self._trust_constr_optimizer
        return Optimizer.L_BFGS_B.value

    @staticmethod
    def _trust_constr_optimizer(obj_func, initial_theta, bounds):
        constraints = [optimize.LinearConstraint(np.eye(initial_theta.shape[0]), bounds[:, 0], bounds[:, 1])]
        res = optimize.minimize(
            lambda theta: obj_func(theta=theta, eval_gradient=False),
            initial_theta,
            constraints=constraints,
            method=Optimizer.TRUST_CONSTR.value,
            jac=lambda theta: obj_func(theta=theta, eval_gradient=True)[1],
            hess=optimize.BFGS(),
            options=dict(gtol=1e-6),
        )
        return res.x, res.fun

    def _find_peak_time_in_predicted_data(self) -> float:
        min_time = self.prepared_bands.X[:, 1].min()
        max_time = self.prepared_bands.X[:, 1].max()
        y_mean = self._predict_explicit(min_time, max_time, return_std=False, band=self.peak_band)
        return min_time + y_mean.argmax()

    def _predict_explicit(self, x_low: float, x_high: float, return_std: bool, band: BandEnum = None) -> np.array | tuple[np.array, np.array]:
        """
        Predict y values independently of peak point.
        If specified, only return predictions to one band.
        Returns prediction mean and standard deviation (optional).
        """
        x = np.linspace(x_low, x_high, int(x_high - x_low) + 1)
        X = np.concatenate([np.column_stack((i * np.ones(len(x)), x)) for i in range(self.nr_bands)])
        y = self.regressor.predict(X, return_std=return_std)
        if band:
            band_index = self.get_band_index(band)
            mask = (X[:, 0] == band_index)
            y = (y[0][mask], y[1][mask]) if return_std else y[mask]
        return y

    def predict_from_peak(self, prediction_interval_from_peak: tuple[int, int]) -> MGPResult:
        if self.peak_time is None:
            raise PeakTimeNotSetError("Set time of peak brightness before making predictions")

        if (prediction_interval_from_peak[0] < self.prediction_interval_from_peak[0] or
            prediction_interval_from_peak[1] > self.prediction_interval_from_peak[1]):
            raise PredictionIntervalOutOfBoundsError(f"Given interval `{prediction_interval_from_peak}` is not within `{self.prediction_interval_from_peak}`")

        days_pre_peak, days_post_peak = -prediction_interval_from_peak[0], prediction_interval_from_peak[1]
        range_width = days_pre_peak + days_post_peak

        # predict
        y_means, y_stds = self._predict_explicit(self.peak_time - days_pre_peak, self.peak_time + days_post_peak, return_std=True)
        # denormalize
        y_means, y_stds = self.prepared_bands.denormalize(y_means), self.prepared_bands.denormalize(y_stds)
        # reorganize to one row per band
        y_means, y_stds = y_means.reshape(self.nr_bands, range_width+1), y_stds.reshape(self.nr_bands, range_width+1)
        # handle negative values
        y_means, y_stds = self._y_negative_to_zero_until_infinity(days_pre_peak, y_means, y_stds)

        return MGPResult(
            sn_name=self.sn_name,
            bandset=self.bandset,
            days_pre_peak=days_pre_peak,
            days_post_peak=days_post_peak,
            log_likelihood=self.regressor.log_marginal_likelihood(),
            thetas=self.regressor.kernel_.theta,
            pred_means={band_name: y_means[i, :] for i, band_name in enumerate(self.bandset.value)},
            pred_stds={band_name: y_stds[i, :] for i, band_name in enumerate(self.bandset.value)},
        )

    def get_interval_relative_to_peak(self, days_pre: int, days_post: int) -> np.array:
        return np.linspace(self.peak_time - days_pre, self.peak_time + days_post, days_pre + days_post + 1)

    def _y_negative_to_zero_until_infinity(self, peak_idx: int, y_means: np.ndarray, y_stds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Relative to the peak point, finds the two closest points (1 to the left, 1 to the right) that are zero or
        negative and zeroes all subsequents values until infinity.
        """
        # Process each band
        for i in range(self.nr_bands):
            # Find first negative or zero value to the left of peak
            for j in range(peak_idx-1, -1, -1):
                if y_means[i, j] <= 0:
                    # Zero out all values to the left including this point
                    y_means[i, :j+1] = 0
                    y_stds[i, :j+1] = 0
                    break

            # Find first negative or zero value to the right of peak
            for j in range(peak_idx + 1, len(y_means[i])):
                if y_means[i, j] <= 0:
                    # Zero out all values to the right including this point
                    y_means[i, j:] = 0
                    y_stds[i, j:] = 0
                    break

        return y_means, y_stds


    @property
    def nr_bands(self):
        return len(self.bandset.value)

    def get_band_index(self, band: BandEnum) -> int:
        try:
            return self.bandset.value.index(band.value)
        except ValueError:
            raise BandNotFoundError(f"Band `{band}` not found in bandset `{self.bandset}`")
