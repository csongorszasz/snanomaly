import numpy as np
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge

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
        if "kernel" not in kwargs:
            kwargs["kernel"] = "rbf"
        if "alpha" not in kwargs:
            # TODO: set dynamically
            kwargs["alpha"] = 0.1
        if "gamma" not in kwargs:
            # TODO: set dynamically
            #  - sparse data => lower gamma (around 0.0005) for smoothness
            #  - dense data => higher gamma (around 0.01) for tighter fitting
            # kwargs["gamma"] = 0.01
            kwargs["gamma"] = 0.0005
        if "verbose" in kwargs:
            # not supported
            kwargs.pop("verbose")
        return KernelRidge(**kwargs)

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
            kwargs["kernel"] = RBF(length_scale_bounds=(avg_adjacent_time_dist, data_range))
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
