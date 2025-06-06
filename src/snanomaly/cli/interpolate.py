from __future__ import annotations

import os

import click

from snanomaly.dataset.exception import DataPointNotFoundError
from snanomaly.dataset.factory import OSCFactory
from snanomaly.interpolation.baseinterpolator import BaseInterpolator
from snanomaly.interpolation.interpolatorfactory import InterpolatorFactory
from snanomaly.interpolation.names import Method, get_display_name
from snanomaly.interpolation.regressioninterpolator import RegressionInterpolator
from snanomaly.interpolation.regressorfactory import RegressorFactory
from snanomaly.interpolation.simpleinterpolator import SimpleInterpolator
from snanomaly.models.results.interpolation_result import InterpolationResult
from snanomaly.models.sncandidate.bands import BandEnum, Bandset
from snanomaly.models.sncandidate.sncandidate import SNCandidate
from snanomaly.visualization.interpolation import PlotInterpolation


@click.group()
def interpolate():
    pass

def _plot_predictions(sn: SNCandidate, bandset: Bandset, predictions: dict | tuple[dict, dict],
                      interpolator: BaseInterpolator):
    stds = None
    if isinstance(predictions, tuple):
        predictions, stds = predictions[0], predictions[1]
    plotter = PlotInterpolation(
        original_bands=sn.photometry.bands,
        int_result=InterpolationResult(
            sn_name=sn.name,
                bandset=bandset,
                peak_time=interpolator.predicted_peak_time,
                days_pre_peak=20,
                days_post_peak=100,
                preds=list(predictions.values()),
                stds=list(stds.values()) if stds else None,
            ),
    )
    plotter.set_title(sn.name)
    plotter.set_subtitle(get_display_name(interpolator.kind))
    plotter.show(600, 600)

@interpolate.command()
@click.argument("sn_name", required=True)
@click.option("--linear", is_flag=True, default=False, help="Only run linear interpolation.")
@click.option("--bspline", is_flag=True, default=False, help="Only run B-spline interpolation.")
@click.option("--gauss-uni", is_flag=True, default=False, help="Only run univariate (one kernel) Gaussian Process Regression for interpolation.")
@click.option("--gauss-multi", is_flag=True, default=False, help="Only run multivariate (multiple kernels) Gaussian Process Regression for interpolation.")
@click.option("--kernel-ridge", is_flag=True, default=False, help="Only run Kernel Ridge Regression for interpolation.")
@click.option("--grad-boost", is_flag=True, default=False, help="Only run Gradient Boosting Regression for interpolation.")
@click.option("--bspline-k", default=3, type=int, help="B-spline degree.")
@click.option("--stop-after-first", is_flag=True, default=False, help="Stop after the first interpolation is run.")
def one(
    sn_name: str,
    linear: bool,
    bspline: bool,
    gauss_uni: bool,
    gauss_multi: bool,
    kernel_ridge: bool,
    grad_boost: bool,
    bspline_k: int,
    stop_after_first: bool,
):
    method_flags = {
        Method.LINEAR.value: linear,
        Method.BSPLINE.value: bspline,
        Method.GAUSS_UNI.value: gauss_uni,
        Method.GAUSS_MULTI.value: gauss_multi,
        Method.KERNEL_RIDGE.value: kernel_ridge,
        Method.GRADIENT_BOOST.value: grad_boost,
    }

    if not any(method_flags.values()):
        click.echo("Warning: No interpolation methods selected. Specify at least one of the flags.")
        return

    interpolators_to_iterate = (
        (SimpleInterpolator, InterpolatorFactory.TYPES),
        (RegressionInterpolator, RegressorFactory.TYPES),
    )

    os.environ["BSPLINE_K"] = str(bspline_k)

    try:
        sn_obj = OSCFactory.OSC2018June().load_datapoint(name=sn_name)
        if not sn_obj.photometry:
            click.echo("Warning: Object has no photometry.")
            return

        click.echo(f"> Interpolating `{sn_obj.name}`")
        click.echo(f"Available band sets: {sn_obj.photometry.bands.available_bandsets}")

        for bs in sn_obj.photometry.bands.available_bandsets:
            peak_band = BandEnum[bs.value[1]]
            for interpolator_class, kinds in interpolators_to_iterate:
                for kind in kinds:
                    if not method_flags[kind]:
                        continue

                    click.echo(f"bandset={bs.__str__()} interpolator_class={interpolator_class.__name__} method={kind}")
                    interpolator: BaseInterpolator = interpolator_class(
                        sn_name=sn_name, bandset=bs, bands=sn_obj.photometry.bands, peak_band=peak_band, kind=kind,
                    )
                    preds = interpolator.predict_from_peak((-20, 100))
                    _plot_predictions(sn=sn_obj, bandset=bs, predictions=preds, interpolator=interpolator)

                    if stop_after_first:
                        click.echo("Stopping after the first interpolation.")
                        return

    except DataPointNotFoundError:
        click.echo(message=f"Could not find supernova candidate with name `{sn_name}`", err=True)

@interpolate.command()
def batch():
    pass
