
import numpy as np
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from snanomaly.interpolation.names import Method
from snanomaly.models.sncandidate.band import Band
from snanomaly.regression.mgp import MGPInterpolator


class RegressorFactory:
    TYPES = (
        Method.GAUSS_UNI.value,
        Method.GAUSS_MULTI.value,
        Method.KERNEL_RIDGE.value,
        Method.GRADIENT_BOOST.value,
    )

    def __new__(cls, regressor_type: str, band: Band = None, **kwargs):
        regressor = {
            cls.TYPES[0]: cls._gpr_uni,
            cls.TYPES[1]: cls._gpr_multi,
            cls.TYPES[2]: cls._krr,
            cls.TYPES[3]: cls._gbr,
        }
        return regressor[regressor_type](band=band, **kwargs)

    @classmethod
    def _krr(cls, band: Band = None, **kwargs) -> KernelRidge:
        verbose = 0
        if "verbose" in kwargs:
            verbose = kwargs.pop("verbose")
        if "kernel" not in kwargs:
            kwargs["kernel"] = "rbf"

        alpha, gamma = cls._tune_alpha_gamma(band, verbose=verbose)

        kwargs["alpha"] = alpha
        kwargs["gamma"] = gamma

        return KernelRidge(**kwargs)

    @classmethod
    def _tune_alpha_gamma(cls, band: Band, verbose: int = 0) -> tuple[float, float]:
        """
        Tune alpha and gamma parameters for Kernel Ridge Regression.
        Returns the tuned alpha and gamma values.
        """
        X = band.time.reshape(-1, 1)
        y = band.flux

        # grid search
        param_grid = {
            "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3],
            "gamma": [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0, 1e1, 5e1, 1e2, 5e2, 1e3],
        }
        krr = KernelRidge(kernel="rbf")
        tscv = TimeSeriesSplit(n_splits=min(5, len(band.time)-1))
        grid_search = GridSearchCV(krr, param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1, verbose=verbose)
        grid_search.fit(X=X, y=y)

        logger.debug(f"Best parameters: {grid_search.best_params_}")
        logger.debug(f"Best score: {grid_search.best_score_}")

        return grid_search.best_params_["alpha"], grid_search.best_params_["gamma"]

    @classmethod
    def _gbr(cls, band: Band = None, **kwargs) -> GradientBoostingRegressor:
        if "n_estimators" not in kwargs:
            kwargs["n_estimators"] = 500
        if "max_depth" not in kwargs:
            kwargs["max_depth"] = 5
        if "learning_rate" not in kwargs:
            kwargs["learning_rate"] = 0.1
        if "random_state" not in kwargs:
            kwargs["random_state"] = 42
        return GradientBoostingRegressor(**kwargs)

    @classmethod
    def _gpr_uni(cls, band: Band = None, **kwargs) -> GaussianProcessRegressor:
        if "kernel" not in kwargs:
            avg_adjacent_time_dist = np.mean(np.diff(band.time))
            data_range = band.time.max() - band.time.min() + 1
            logger.debug(f"band={band.name} length_scale_bounds=({float(avg_adjacent_time_dist), float(data_range)})")
            kwargs["kernel"] = RBF(length_scale_bounds=(avg_adjacent_time_dist, max(avg_adjacent_time_dist * 1.1, data_range))) # TODO
        if "alpha" not in kwargs and band:
            kwargs["alpha"] = band.e_flux
        if "n_restarts_optimizer" not in kwargs:
            kwargs["n_restarts_optimizer"] = 0
        if "random_state" not in kwargs:
            kwargs["random_state"] = 42
        if "verbose" in kwargs:
            # not supported
            kwargs.pop("verbose")
        return GaussianProcessRegressor(**kwargs)

    @classmethod
    def _gpr_multi(cls, band: Band = None, **kwargs) -> MGPInterpolator:
        raise NotImplementedError
