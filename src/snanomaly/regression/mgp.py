from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import numpy as np
from attrs import define, field
from multistate_kernel import MultiStateKernel
from scipy import optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel

from snanomaly.models.sncandidate import Bandset
from snanomaly.models.sncandidate.band import Band
from snanomaly.models.sncandidate.bands import Bands


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
    norm_factor: np.float64 = field(init=False)

    def __attrs_post_init__(self):
        self.bands = [band.binned(bin_width=1) if not band.is_binned else band for band in self.bands]
        self.X = np.concatenate(
            [np.column_stack((i * np.ones(band.nr_observations), band.time)) for i, band in enumerate(self.bands)],
        )
        self.y = np.concatenate([band.flux for band in self.bands])
        self.norm_factor = self.y.std()
        self.y = self.y / self.norm_factor
        self.err = np.concatenate([band.e_flux for band in self.bands]) / self.norm_factor

@define
class MGPInterpolator:
    """Interpolate light curves in available band sets with Multivariate Gaussian Process Regression."""

    bandset: Bandset = field()
    bands: Bands = field()
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
        alpha = self._init_alpha()
        optimizer = self._init_optimizer()
        return GaussianProcessRegressor(
            kernel=ms_kernel,
            alpha=alpha,
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

    # def predict(self, time_lower_bound: int, time_upper_bound: int):
    #     time_new = np.linspace(np.min(prepped_bands[0].time), np.max(prepped_bands[0].time), 100)
    #     X_new_all = np.concatenate([np.vstack((i * np.ones(len(time_new)), time_new)).T for i in range(3)])
    #     y_pred_all, y_std_all = gp.predict(X_new_all, return_std=True)
    #     y_pred_all = y_pred_all.reshape(3, len(time_new)).T
    #     y_std_all = y_std_all.reshape(3, len(time_new)).T
