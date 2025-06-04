
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge


class RegressorFactory:
    TYPES = (
        "kernel_ridge",
        "grad_boost",
        "gauss",
    )

    def __new__(cls, regressor_type: str, **kwargs):
        regressor = {
            "kernel_ridge": cls._krr,
            "grad_boost": cls._gbr,
            "gauss": cls._gpr,
        }
        return regressor[regressor_type](**kwargs)

    @classmethod
    def _krr(cls, **kwargs) -> KernelRidge:
        if "kernel" not in kwargs:
            kwargs["kernel"] = "rbf"
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.1
        if "gamma" not in kwargs:
            kwargs["gamma"] = 0.01
        return KernelRidge(**kwargs)

    @classmethod
    def _gbr(cls, **kwargs) -> GradientBoostingRegressor:
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
    def _gpr(cls, **kwargs) -> GaussianProcessRegressor:
        if "kernel" not in kwargs:
            kwargs["kernel"] = RBF(length_scale=1, length_scale_bounds="fixed")
        if "n_restarts_optimizer" not in kwargs:
            kwargs["n_restarts_optimizer"] = 10
        if "random_state" not in kwargs:
            kwargs["random_state"] = 42
        return GaussianProcessRegressor(**kwargs)
