from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterable
from enum import Enum
from typing import Optional

import numpy as np
import scipy
from attrs import define, field
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
    CouldNotConvergeError,
    PeakTimeNotSetError,
    PredictionIntervalOutOfBoundsError,
)


class Optimizer(Enum):
    L_BFGS_B = "fmin_l_bfgs_b"
    TRUST_CONSTR = "trust-constr"

class LengthScaleBoundsInitStrategy(Enum):
    STATIC = 0
    DYNAMIC_SET_ONCE = 1
    DYNAMIC_BIN_SEARCH = 2


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
        # print("y After concatenation:", self.y)
        self.err: np.array = np.concatenate([band.e_flux for band in self.bands])
        # print("err After concatenation:", self.y)

        # self.norm_factor = (self.y.mean(), self.y.std())
        self.norm_factor = (0, self.y.max())
        # print("Norm factor:",self.norm_factor)

        self.y = self.normalize(self.y)
        # print("y After normalization:", self.y)

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
    n_restarts_optimizer: int = field()
    length_scale_bounds_init_strategy: LengthScaleBoundsInitStrategy = field()
    normalize_y: bool = field(default=False)
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
        self._verify_prepared_bands()

        self.regressor = self._init_regressor()
        self._verify_fit()

        self.peak_time = self._find_peak_time_in_predicted_data()

    def _verify_prepared_bands(self):
        assert not np.isnan(self.prepared_bands.X).any(), "X contains NaNs"
        # print(self.prepared_bands.y)
        assert not np.isnan(self.prepared_bands.y).any(), "y contains NaNs"
        assert np.isfinite(self.prepared_bands.X).all(), "X contains non-finite values"
        assert np.isfinite(self.prepared_bands.y).all(), "y contains non-finite values"

    def _verify_fit(self):
        assert self.regressor is not None, "Regressor is not initialized"
        assert self.regressor.kernel_ is not None, "Kernel is not initialized"
        assert self.regressor.kernel_.state_kernels is not None, "State kernels are not initialized"
        assert self.regressor.kernel_.theta is not None, "Theta is not initialized"

    def _init_regressor(self):
        strategies = {
            LengthScaleBoundsInitStrategy.STATIC.value: self._init_regressor_static_length_scale_bounds,
            LengthScaleBoundsInitStrategy.DYNAMIC_SET_ONCE.value: self._init_regressor_dynamic_length_scale_bounds_set_once,
            LengthScaleBoundsInitStrategy.DYNAMIC_BIN_SEARCH.value: self._init_regressor_dynamic_length_scale_bounds_binary_search,

            ####################################################
            # TODO: 3-SIGMA szabaly a length scale beallitasahoz
            ####################################################
            # channel-enkent kiszamolni a 3sigma szorast
            # -> az RBF kernel length scale parameteret beallitani 3*sigmara (megjegyzes: a length scale az adatok 99.7%-ara illeszkednie kell)
            # az OSSZES szupernova menten
            # ? van-e hiperparameter tuning alapertelmezetten
        }
        return strategies[self.length_scale_bounds_init_strategy.value]()

    def _init_regressor_static_length_scale_bounds(self) -> GaussianProcessRegressor:
        kernels = [self.kernel(length_scale_bounds=(self.length_scale_min_bounds[0], 1e4)) for _ in range(self.nr_bands)]
        # kernels = [self.kernel(length_scale=1e-7) for _ in range(self.nr_bands)]
        regressor = self.construct_regressor(kernels=kernels)
        regressor, kernel_idx = self.fit_regressor(regressor=regressor)
        if kernel_idx is not None:
            raise CouldNotConvergeError(
                f"Could not converge for kernel {kernel_idx} with length-scale lower bound "
                f"{self.length_scale_min_bounds[0]}, , length-scale: {regressor.kernel_.state_kernels[kernel_idx].length_scale_bounds[0]}",
            )
        return regressor

    def _init_regressor_dynamic_length_scale_bounds_set_once(self) -> GaussianProcessRegressor:
        regressor = self.construct_regressor()
        regressor, kernel_idx = self.fit_regressor(regressor=regressor)
        if kernel_idx is not None:
            raise CouldNotConvergeError(
                f"Could not converge for kernel {kernel_idx} with length-scale lower bound "
                f"{self.length_scale_min_bounds[0]}, length-scale: {regressor.kernel_.state_kernels[kernel_idx].length_scale_bounds[0]}",
            )
        return regressor

    def _init_regressor_dynamic_length_scale_bounds_binary_search(self) -> GaussianProcessRegressor:
        def search_optimal_min_length_scale(
                low: float,
                high: float,
                fixed_lower_bounds: dict,
                curr_kernel_idx: int,
                prev_kernel_warning_idx: Optional[int] = None,
                prev_optimal_regressor: Optional[GaussianProcessRegressor] = None,
        ) -> tuple[GaussianProcessRegressor, int]:
            """
            Find optimal lower bounds for kernel length-scales with binary search until no ConvergenceWarning is given.
            Return the fitted GP regressor and optionally, the index of the kernel that raised a convergence warning on
            the last fit.
            """
            print(f"Searching for optimal length-scale lower bound for kernel {curr_kernel_idx} in range [{low}, {high}]")
            mid = (low + high) / 2
            fixed_lower_bounds[curr_kernel_idx] = mid
            kernels = self._init_kernels(fixed_lower_bounds)
            regressor = self.construct_regressor(kernels)
            regressor, kernel_idx = self.fit_regressor(regressor=regressor)
            print(f"Convergence warning raised for kernel {kernel_idx} with length-scale lower bound {mid}")
            if kernel_idx is None or kernel_idx != curr_kernel_idx:
                # 5. don't stop the search as soon as no error is given for that kernel
                # 6. continue the search until there is an error again for that kernel
                #   and take the previous value as the optimum (or the search interval has closed up)
                print(f"Found an optima: lengt-scale lower bound = {mid}")
                return search_optimal_min_length_scale(low=mid, high=high,
                                                       fixed_lower_bounds=fixed_lower_bounds,
                                                       curr_kernel_idx=curr_kernel_idx,
                                                       prev_kernel_warning_idx=kernel_idx,
                                                       prev_optimal_regressor=regressor)
            # convergence warning is raised
            if low >= high:
                # highest possible lower length-scale bound has been found
                if prev_optimal_regressor is not None:
                    # an optimum with no warning has been found already
                    print(f"Found an optima: length-scale lower bound = {prev_optimal_regressor.kernel_.state_kernels[curr_kernel_idx].length_scale_bounds[0]}")
                    return prev_optimal_regressor, prev_kernel_warning_idx
                # return regressor without global optima found
                print("! Could not converge")
                return regressor, curr_kernel_idx
            # the optimum may be to the left
            return search_optimal_min_length_scale(low=low, high=mid,
                                                   fixed_lower_bounds=fixed_lower_bounds,
                                                   curr_kernel_idx=curr_kernel_idx,
                                                   prev_kernel_warning_idx=kernel_idx,
                                                   prev_optimal_regressor=regressor)

        # 1. do a test fit
        regressor, kernel_idx = self.fit_regressor(regressor=self.construct_regressor())
        if kernel_idx is None:
            # 2. if there is no convergence warning, simply return
            return regressor

        fixed_lower_bounds: dict = {}
        while True:
            # 3. save the problematic kernel's lower length-scale bound as the upper limit of a binary search
            ls_low_bound: float = regressor.kernel_.state_kernels[kernel_idx].length_scale_bounds[0]
            print(f"Kernels: {regressor.kernel_.state_kernels}")
            print(f"Problematic kernel (idx={kernel_idx}) length-scale lower bound: {ls_low_bound}")
            # 4. use binary search to find a length-scale lower bound for the problematic kernel that doesn't cause error for the that kernel
            old_kernel_idx = kernel_idx
            regressor, kernel_idx = search_optimal_min_length_scale(low=self.length_scale_min_bounds[0], high=ls_low_bound,
                                                                    fixed_lower_bounds=fixed_lower_bounds,
                                                                    curr_kernel_idx=kernel_idx,
                                                                    prev_kernel_warning_idx=kernel_idx,
                                                                    prev_optimal_regressor=None)
            ls_low_bound: float = regressor.kernel_.state_kernels[kernel_idx].length_scale_bounds[0]
            print(f"Optimized kernel (idx={kernel_idx}) length-scale lower bound: {ls_low_bound}")
            # 7. if upon finding the optimum for the initial problematic kernel another kernel is giving convergence warning, repeat the search on that kernel
            if kernel_idx is not None and kernel_idx != old_kernel_idx:
                # fixate previously found length-scale lower bound
                fixed_lower_bounds[kernel_idx] = ls_low_bound
                # 8. repeat the search for the new kernel
                continue
            break
        return regressor

    def construct_regressor(self, kernels: Optional[Iterable[Kernel]] = None) -> GaussianProcessRegressor:
        """Create a GP instance with the option to explicitly pass the kernels."""
        kernels = self._init_kernels() if kernels is None else kernels
        scale, scale_bounds = self._init_scale_matrix()
        ms_kernel = MultiStateKernel(kernels=kernels, scale=scale, scale_bounds=scale_bounds)
        # alpha = self._init_alpha() # TODO
        optimizer = self._init_optimizer()
        return GaussianProcessRegressor(
            kernel=ms_kernel,
            # alpha=1e-5, # TODO
            # optimizer=optimizer,
            optimizer=self._fmin_opt,
            n_restarts_optimizer=self.n_restarts_optimizer,
            # normalize_y=self.normalize_y, # TODO
            normalize_y=False,  # TODO
            random_state=self.random_state,
        )

    def fit_regressor(self, regressor: GaussianProcessRegressor) -> tuple[GaussianProcessRegressor, Optional[int]]:
        """
        Returns a fitted GP regressor and the index of the kernel that failed to converge (if any).
        """
        try:
            with warnings.catch_warnings():
                # Treat ConvergenceWarning as an error
                warnings.filterwarnings("error", category=ConvergenceWarning)
                regressor.fit(self.prepared_bands.X, self.prepared_bands.y)
                return regressor, None
        except ConvergenceWarning as ex:
            # extract dimension number from warning message with regex
            kernel_idx = re.search(r"dimension (\d+)", str(ex))
            if kernel_idx:
                kernel_idx = int(kernel_idx.group(1))
                return regressor, kernel_idx
            raise CouldNotConvergeError(ex)

    def _init_kernels(self, fixed_length_scale_low_bounds: Optional[dict] = None) -> tuple[Kernel]:
        """
        The argument is an optional dictionary of (<kernel_index>, <length_scale_lower_bound>), in case of the
        length-scale bounds need to be explicitly given instead of dynamically found.
        """
        min_bounds = [None] * self.nr_bands
        if fixed_length_scale_low_bounds:
            for kernel_idx, low_bound in fixed_length_scale_low_bounds.items():
                min_bounds[kernel_idx] = low_bound

        for i, band in enumerate(self.bands.get_bands(self.bandset)):
            if min_bounds[i] is None:
                time_diff = np.max(np.diff(band.time)) if band.nr_observations > 1 else 0
                min_bounds[i] = max(self.length_scale_min_bounds[0], min(time_diff, self.length_scale_min_bounds[1]))

        return tuple(self.kernel(length_scale_bounds=(min_bound, 1e4)) for min_bound in min_bounds)

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

    @staticmethod
    def _fmin_opt(obj_func, initial_theta, bounds):
        res = scipy.optimize.minimize(
            obj_func,
            initial_theta,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 50000},
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
            band_index = self.get_band_index(band, self.bandset)
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
            peak_time=self.peak_time,
            days_pre_peak=days_pre_peak,
            days_post_peak=days_post_peak,
            log_likelihood=self.regressor.log_marginal_likelihood(),
            thetas=self.regressor.kernel_.theta,
            pred_means=[y_means[i, :] for i in range(self.nr_bands)],
            pred_stds=[y_stds[i, :] for i in range(self.nr_bands)],
        )

    @staticmethod
    def get_interval_relative_to_peak(peak_time: float, days_pre: int, days_post: int) -> np.array:
        return np.linspace(peak_time - days_pre, peak_time + days_post, days_pre + days_post + 1)

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

    @staticmethod
    def get_band_index(band: BandEnum, bandset: Bandset) -> int:
        try:
            return bandset.value.index(band.value)
        except ValueError:
            raise BandNotFoundError(f"Band `{band}` not found in bandset `{bandset}`")
