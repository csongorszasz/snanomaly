
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge

from snanomaly.models.sncandidate.band import Band


class RegressorFactory:
    TYPES = (
        "kernel_ridge",
        "grad_boost",
        "gauss",
        "multigauss",
    )

    def __new__(cls, regressor_type: str, band: Band = None, **kwargs):
        regressor = {
            "kernel_ridge": cls._krr,
            "grad_boost": cls._gbr,
            "gauss": cls._gpr,
            "multigauss": cls._multigpr,
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
        return KernelRidge(**kwargs)

    @classmethod
    def _gbr(cls, band: Band = None, **kwargs) -> GradientBoostingRegressor:
        if "n_estimators" not in kwargs:
            kwargs["n_estimators"] = 200
        if "max_depth" not in kwargs:
            kwargs["max_depth"] = 4
        if "learning_rate" not in kwargs:
            kwargs["learning_rate"] = 0.1
        if "random_state" not in kwargs:
            kwargs["random_state"] = 42
        return GradientBoostingRegressor(**kwargs)

    @classmethod
    def _gpr(cls, band: Band = None, **kwargs) -> GaussianProcessRegressor:
        if "kernel" not in kwargs:
            kwargs["kernel"] = RBF(length_scale=1, length_scale_bounds="fixed")
        if "alpha" not in kwargs and band:
            kwargs["alpha"] = band.e_flux
        if "n_restarts_optimizer" not in kwargs:
            kwargs["n_restarts_optimizer"] = 0
        if "random_state" not in kwargs:
            kwargs["random_state"] = 42
        return GaussianProcessRegressor(**kwargs)

    @classmethod
    def _multigpr(cls, band: Band = None, **kwargs):
        raise NotImplementedError
