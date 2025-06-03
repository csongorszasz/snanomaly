import click

from snanomaly.dataset.exception import DataPointNotFoundError
from snanomaly.dataset.factory import OSCFactory
from snanomaly.interpolation.simpleinterpolator import SimpleInterpolator
from snanomaly.models.results.interpolation_result import InterpolationResult
from snanomaly.models.sncandidate.bands import BandEnum
from snanomaly.visualization.interpolation import PlotInterpolation


@click.group()
def interpolate():
    pass

@interpolate.command()
@click.argument("sn_name", required=True)
def one(sn_name: str):
    try:
        sn_obj = OSCFactory.OSC2018June().load_datapoint(name=sn_name)
        if not sn_obj.photometry:
            click.echo("Warning: Object has no photometry.")
            return

        click.echo(f"> Interpolating `{sn_obj.name}`")
        interpolators_kinds = ("linear", "cubic", "nearest", "zero", "slinear", "quadratic")

        for int_kind in interpolators_kinds:
            for bs in sn_obj.photometry.bands.available_bandsets:
                click.echo(f"bandset={bs.__str__()} interpolation={int_kind}")
                interpolator = SimpleInterpolator(sn_name=sn_name, bandset=bs, bands=sn_obj.photometry.bands,
                                   peak_band=BandEnum[bs.value[1]], kind=int_kind)
                preds: dict = interpolator.predict_from_peak((-20, 100))

                plotter = PlotInterpolation(
                    original_bands=sn_obj.photometry.bands,
                    int_result=InterpolationResult(sn_name=sn_name, bandset=bs, peak_time=interpolator.predicted_peak_time,
                                                   days_pre_peak=20, days_post_peak=100, preds=list(preds.values())),
                )
                plotter.set_title(f"{sn_name} - {bs.__str__()} - {int_kind}")
                plotter.show(600,600)

    except DataPointNotFoundError:
        click.echo(message=f"Could not find supernova candidate with name `{sn_name}`", err=True)

@interpolate.command()
def batch():
    pass
