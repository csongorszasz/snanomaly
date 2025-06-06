from __future__ import annotations

import os

import click

from snanomaly.dataset.exception import DataPointNotFoundError
from snanomaly.dataset.factory import OSCFactory
from snanomaly.interpolation.baseinterpolator import BaseInterpolator
from snanomaly.interpolation.interpolatorfactory import InterpolatorFactory
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
    if isinstance(interpolator, SimpleInterpolator):
        title = f"{sn.name} - {bandset.__str__()} - {interpolator.kind} interpolation"
    elif isinstance(interpolator, RegressionInterpolator):
        title = f"{sn.name} - {bandset.__str__()} - {interpolator.kind} regression"
    else:
        raise ValueError(f"Unknown interpolator type: {type(interpolator)}")
    plotter.set_title(title)
    plotter.show(600, 600)

@interpolate.command()
@click.argument("sn_name", required=True)
@click.option("-r", "--regression-only", is_flag=True, default=False, help="Only run regression-based interpolators.")
@click.option("-s", "--simple-interpolation-only", is_flag=True, default=False, help="Only run simple interpolators.")
@click.option("--stop-after-first", is_flag=True, default=False, help="Stop after the first interpolation is run.")
@click.option("--bspline-k", default=3, type=int, help="B-spline degree for simple interpolation.")
def one(sn_name: str, regression_only: bool, simple_interpolation_only: bool, stop_after_first: bool,
        bspline_k: int):
    if regression_only and simple_interpolation_only:
        click.echo("Error: Cannot use both --regression-only and --simple-interpolation-only at the same time.")
        return

    interpolators_to_iterate = (
        (SimpleInterpolator, InterpolatorFactory.TYPES) if not regression_only else None,
        (RegressionInterpolator, RegressorFactory.TYPES) if not simple_interpolation_only else None,
    )
    interpolators_to_iterate = list(filter(lambda x: x is not None, interpolators_to_iterate))

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
