import os

import numpy as np
from scipy.interpolate import BSpline, interp1d, make_interp_spline

from snanomaly.interpolation.names import Method


class InterpolatorFactory:
    TYPES = (
        Method.LINEAR.value,
        Method.BSPLINE.value,
    )

    def __new__(cls, x: np.ndarray, y: np.ndarray, interpolator_type: str, **kwargs):
        interpolator = {
            cls.TYPES[0]: cls._linear,
            cls.TYPES[1]: cls._bspline,
        }
        return interpolator[interpolator_type](x=x, y=y, **kwargs)

    @classmethod
    def _linear(cls, x: np.ndarray, y: np.ndarray, **kwargs) -> interp1d:
        return interp1d(x, y, kind="linear", fill_value="extrapolate")

    @classmethod
    def _bspline(cls, x: np.ndarray, y: np.ndarray, **kwargs) -> BSpline:
        if "BSPLINE_K" in os.environ:
            kwargs["k"] = int(os.environ["BSPLINE_K"])
        elif "k" not in kwargs:
            kwargs["k"] = 3
        t, c, k = make_interp_spline(x, y, **kwargs).tck
        return BSpline(t, c, k, extrapolate=True)
