from enum import Enum


class Method(Enum):
    LINEAR = "linear"
    BSPLINE = "bspline"
    GAUSS_UNI = "gauss_uni"
    GAUSS_MULTI = "gauss_multi"
    KERNEL_RIDGE = "kernel_ridge"
    GRADIENT_BOOST = "grad_boost"

display_names = {
    Method.LINEAR.value: "Linear interpolation",
    Method.BSPLINE.value: "B-spline interpolation",
    Method.GAUSS_UNI.value: "Gaussian Process Regression (one kernel)",
    Method.GAUSS_MULTI.value: "Gaussian Process Regression (multiple kernels)",
    Method.KERNEL_RIDGE.value: "Kernel Ridge Regression",
    Method.GRADIENT_BOOST.value: "Gradient Boost Regression",
}

def get_display_name(method: str) -> str:
    return display_names[method]
