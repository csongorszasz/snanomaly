from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import numpy as np
from attrs import define, field
from multistate_kernel import MultiStateKernel
from scipy import optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel

from snanomaly.models.results.mgp_result import MGPResult
from snanomaly.models.sncandidate import Bandset
from snanomaly.models.sncandidate.band import Band
from snanomaly.models.sncandidate.bands import BandEnum, Bands


class Optimizer(Enum):
    L_BFGS_B = "fmin_l_bfgs_b"
    TRUST_CONSTR = "trust-constr"

@define
class PreparedData:
    """Represents the data format required by a MultiStateKernel."""

    bands: list[Band] = field()
    X: np.ndarray = field(init=False)
    y: np.array = field(init=False)
    err: np.array = field(init=False)
    # norm_factor: np.float64 = field(init=False)
    norm_y_mean: np.float64 = field(init=False)
    norm_y_std: np.float64 = field(init=False)

    def __attrs_post_init__(self):
        self.bands = [band.binned(bin_width=1) if not band.is_binned else band for band in self.bands]
        self.X = np.concatenate(
            [np.column_stack((i * np.ones(band.nr_observations), band.time)) for i, band in enumerate(self.bands)],
        )
        self.y = np.concatenate([band.flux for band in self.bands])
        self.err = np.concatenate([band.e_flux for band in self.bands])
        # self.norm_factor = self.y.std()
        # self.norm_factor = 1 # TODO
        self.standardize()

    def standardize(self):
        """
        Standardizes the data by subtracting the mean and dividing by the standard deviation.
        """
        # self.norm_y_mean = 0
        # self.norm_y_mean = np.mean(self.y)
        # self.norm_y_std = np.std(self.y)
        # self.y = (self.y - self.norm_y_mean) / self.norm_y_std
        # self.err = (self.err - np.mean(self.err)) / np.std(self.err)

        # self.norm_y_std = np.std(self.y) or self.y[0] or 1
        self.norm_y_std = np.max(self.y)
        self.y /= self.norm_y_std
        self.err /= self.norm_y_std

    def destandardize(self, y: np.ndarray) -> np.ndarray:
        """
        Reverses the standardization process.
        """
        # return y * self.norm_y_std + self.norm_y_mean
        return y * self.norm_y_std

@define
class MGPInterpolator:
    """Interpolate light curves in available band sets with Multivariate Gaussian Process Regression."""

    sn_name: str = field()
    bandset: Bandset = field()
    bands: Bands = field()
    peak_band: BandEnum = field()
    normalize_y: bool = field()
    n_restarts_optimizer: int = field()
    kernel: Kernel = field(default=RBF)
    length_scale_min_bounds: tuple[float, float] = field(default=(0, np.inf))
    optimize_method: Optimizer = field(default=Optimizer.L_BFGS_B)
    random_state: int = field(default=None)
    regressor: GaussianProcessRegressor = field(init=False)
    prepared_bands: PreparedData = field(init=False)

    def __attrs_post_init__(self):
        self.prepared_bands = PreparedData(self.bands.get_bands(self.bandset))
        self.regressor = self._init_regressor()
        self.regressor.fit(self.prepared_bands.X, self.prepared_bands.y)

    def _init_regressor(self) -> GaussianProcessRegressor:
        kernels = self._init_kernels()
        scale, scale_bounds = self._init_scale_matrix()
        ms_kernel = MultiStateKernel(kernels=kernels, scale=scale, scale_bounds=scale_bounds)
        # alpha = self._init_alpha() # TODO
        optimizer = self._init_optimizer()
        return GaussianProcessRegressor(
            kernel=ms_kernel,
            # alpha=alpha, # TODO
            optimizer=optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,
            random_state=self.random_state,
        )

    def _init_kernels(self) -> tuple[Kernel]:
        time_diffs = (
            np.max(np.diff(band.time)) if band.nr_observations > 1 else 0 for band in self.bands.get_bands(self.bandset)
        )
        min_bounds = [
            max(self.length_scale_min_bounds[0], min(diff, self.length_scale_min_bounds[1])) for diff in time_diffs
        ]
        return tuple(self.kernel(length_scale_bounds=(min_bounds, 1e4)) for min_bounds in min_bounds)

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
            return self.trust_constr_optimizer
        return Optimizer.L_BFGS_B.value

    @staticmethod
    def trust_constr_optimizer(obj_func, initial_theta, bounds):
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

    def predict(self, days_pre_peak: int, days_post_peak: int) -> MGPResult:
        # time_new = np.linspace(np.min(prepped_bands[0].time), np.max(prepped_bands[0].time), 100)
        time_new = np.linspace() # TODO get peak day
        X_new_all = np.concatenate([np.vstack((i * np.ones(len(time_new)), time_new)).T for i in range(3)])
        y_mean_all, y_std_all = self.regressor.predict(X_new_all, return_std=True)
        y_mean_all = y_mean_all.reshape(len(self.bandset.value), len(time_new))
        y_std_all = y_std_all.reshape(len(self.bandset.value), len(time_new))
        return MGPResult(
            sn_name=self.sn_name,
            bandset=self.bandset,
            days_pre_peak=days_pre_peak,
            days_post_peak=days_post_peak,
            derivs=None, # TODO
            weight_derivs=None, # TODO
            log_likelihood=self.regressor.log_marginal_likelihood(),
            thetas=self.regressor.kernel_.theta,
            pred_means={band_name: y_mean_all[i, :] for i, band_name in enumerate(self.bandset.value)},
            pred_stds={band_name: y_std_all[i, :] for i, band_name in enumerate(self.bandset.value)},
        )
