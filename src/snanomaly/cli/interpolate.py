from __future__ import annotations

import os
import pathlib
from datetime import datetime

import click
import polars as pl
from tqdm import tqdm

from snanomaly import dirs
from snanomaly.dataset.exception import DataPointNotFoundError
from snanomaly.dataset.factory import OSCFactory
from snanomaly.interpolation.baseinterpolator import BaseInterpolator
from snanomaly.interpolation.interpolatorfactory import InterpolatorFactory
from snanomaly.interpolation.names import Method, get_display_name
from snanomaly.interpolation.regressioninterpolator import RegressionInterpolator
from snanomaly.interpolation.regressorfactory import RegressorFactory
from snanomaly.interpolation.simpleinterpolator import SimpleInterpolator
from snanomaly.models.results.interpolation_result import InterpolationResult
from snanomaly.models.results.validation_result import ValidationResult
from snanomaly.models.sncandidate.bands import BandEnum, Bandset
from snanomaly.models.sncandidate.sncandidate import SNCandidate
from snanomaly.visualization.interpolation import PlotInterpolation


@click.group()
def interpolate():
    pass

def _create_predictions_plot(sn: SNCandidate, bandset: Bandset, predictions: dict | tuple[dict, dict],
                             interpolator: BaseInterpolator) -> PlotInterpolation:
    stds = None
    if isinstance(predictions, tuple):
        predictions, stds = predictions[0], predictions[1]
    plotter = PlotInterpolation(
        original_bands=sn.photometry.bands,
        preprocessed_bands=interpolator.bands_binned,
        int_result=InterpolationResult(
            sn_name=sn.name,
                bandset=bandset,
                peak_time=interpolator.predicted_peak_time,
                days_pre_peak=20,
                days_post_peak=100,
                pred_means=list(predictions.values()),
                pred_stds=list(stds.values()) if stds else [],
            ),
    )
    model_name = get_display_name(interpolator.kind)
    plotter.set_title(sn.name, model_name)
    return plotter


@interpolate.command()
@click.option("-s", "--sn_name", type=str, help="Name of the supernova candidate to interpolate.", required=True)
@click.option("-d", "--dataset", type=click.Choice(["osc2018_june", "osc2022"]), required=True,
              help="Dataset to get objects from.")
@click.option("--linear", is_flag=True, default=False, help="Only run linear interpolation.")
@click.option("--bspline", is_flag=True, default=False, help="Only run B-spline interpolation.")
@click.option("--gauss-uni", is_flag=True, default=False, help="Only run univariate (one kernel) Gaussian Process Regression for interpolation.")
@click.option("--gauss-multi", is_flag=True, default=False, help="Only run multivariate (multiple kernels) Gaussian Process Regression for interpolation.")
@click.option("--kernel-ridge", is_flag=True, default=False, help="Only run Kernel Ridge Regression for interpolation.")
@click.option("--grad-boost", is_flag=True, default=False, help="Only run Gradient Boosting Regression for interpolation.")
@click.option("--bspline-k", default=3, type=int, help="B-spline degree.", show_default=True)
@click.option("--stop-after-first", is_flag=True, default=False, help="Stop after the first interpolation is run.")
@click.option("--plot-rows", default=-1, type=int, help="Number of rows in the plot grid (applies when creating plot with subplots). If `-1` is given, then `plot-cols` is dynamically calculated.", show_default=True)
@click.option("--plot-cols", default=-1, type=int, help="Number of columns in the plot grid (applies when creating plot with subplots). If `-1` is given, then `plot-rows` is", show_default=True)
@click.option("--plot-width", default=600, type=int, help="Width of the plot in pixels (applies to one individual supernova plot).", show_default=True)
@click.option("--plot-height", default=600, type=int, help="Height of the plot in pixels (applies to one individual supernova plot).", show_default=True)
@click.option("--silence-warnings", is_flag=True, default=False, help="Silence warnings during interpolation.")
@click.option("-v", "--verbosity-level", default=2, type=int, help="Verbosity level for logging (0: none, 1: basic, 2: detailed).", show_default=True)
def one(
    sn_name: str,
    dataset: str,
    linear: bool,
    bspline: bool,
    gauss_uni: bool,
    gauss_multi: bool,
    kernel_ridge: bool,
    grad_boost: bool,
    bspline_k: int,
    stop_after_first: bool,
    plot_rows: int,
    plot_cols: int,
    plot_width: int,
    plot_height: int,
    silence_warnings: bool,
    verbosity_level: int,
):
    ds = OSCFactory.get(dataset)

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
        sn_obj = ds.load_datapoint(name=sn_name)
        if not sn_obj.photometry:
            click.echo("Warning: Object has no photometry.")
            return

        click.echo(f"> Interpolating `{sn_obj.name}`")
        click.echo(f"Available band sets: {sn_obj.photometry.bands.available_bandsets}")

        plotters = []
        combined_figures = (plot_rows != -1 or plot_cols != -1)
        stop_iterating = False
        for bs in sn_obj.photometry.bands.available_bandsets:
            if stop_iterating:
                break
            peak_band = BandEnum[bs.value[1]]
            for interpolator_class, kinds in interpolators_to_iterate:
                if stop_iterating:
                    break
                for kind in kinds:
                    if not method_flags[kind]:
                        continue

                    click.echo(f"bandset={bs.__str__()} interpolator_class={interpolator_class.__name__} method={kind}")
                    interpolator: BaseInterpolator = interpolator_class(
                        sn_name=sn_name, bandset=bs, bands=sn_obj.photometry.bands, peak_band=peak_band, kind=kind,
                        interpolator_arguments={"verbose": verbosity_level}, silence_warnings=silence_warnings,
                    )
                    preds = interpolator.predict_from_peak((-20, 100))
                    plotter = _create_predictions_plot(sn=sn_obj, bandset=bs, predictions=preds, interpolator=interpolator)
                    plotters.append(plotter)

                    if stop_after_first:
                        click.echo("Stopping after the first interpolation.")
                        stop_iterating = True
                        break

        if not combined_figures:
            for p in plotters:
                p.show(plot_width, plot_height)
        else:
            if plot_rows == -1:
                plot_rows = len(plotters) // plot_cols
            elif plot_cols:
                plot_cols = len(plotters) // plot_rows
            PlotInterpolation.show_grid(
                plotters=plotters,
                nrows=plot_rows,
                ncols=plot_cols,
                width_per_subfig=plot_width,
                height_per_subfig=plot_height,
            )

    except DataPointNotFoundError:
        click.echo(message=f"Could not find supernova candidate with name `{sn_name}`", err=True)

@interpolate.command()
@click.option("-i", "--inpath", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path), required=True,
              help="Path to the input data file (Parquet format) that contains names of validated supernova candidates along with available band sets.")
@click.option("-d", "--dataset", type=click.Choice(["osc2018_june", "osc2022"]), required=True,
              help="Dataset to get objects from.")
@click.option("--method", type=click.Choice(["optimal", "gauss-uni", "kernel-ridge"]), default="optimal",
              show_default=True, help="Method to use for interpolation. (`optimal` primarily uses `gauss-uni` and where it fails, it falls back to `kernel-ridge`)")
@click.option("--stop-after", default=None, type=int, help="Stop after processing this many candidates")
@click.option("--silence-warnings", is_flag=True, default=False, help="Silence warnings during interpolation.")
def batch(
    inpath: pathlib.Path,
    dataset: str,
    method: str,
    stop_after: int,
    silence_warnings: bool,
):
    # prepare the output directory
    run_id = f"{dataset}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    out_dir_root = dirs.INTERPOLATED / run_id
    out_data_path = out_dir_root / f"{run_id}.parquet"
    out_dir_plots = out_dir_root / "plots"
    out_dir_root.mkdir(parents=True, exist_ok=True)
    out_dir_plots.mkdir(parents=True, exist_ok=True)

    click.echo("Loading dataframe...")
    df = pl.read_parquet(inpath, n_rows=stop_after)
    click.echo("OK")

    ds = OSCFactory.get(dataset)

    methods = [Method.GAUSS_UNI, Method.KERNEL_RIDGE] if method == "optimal" else [Method(method)]

    skipped = []
    for row in tqdm(df.iter_rows(named=True), desc="Interpolating candidates", total=len(df)):
        val_res = ValidationResult.from_dict(row)
        try:
            sn_obj = ds.load_datapoint(name=val_res.sn_name)
            if not sn_obj.photometry:
                skipped.append((val_res.sn_name, "no photometry"))
                continue

            for bs in val_res.available_bandsets:
                peak_band = BandEnum[bs.value[1]]
                success = False
                for method in methods:
                    if success:
                        break
                    try:
                        interpolator: BaseInterpolator = RegressionInterpolator(
                            sn_name=sn_obj.name, bandset=bs, bands=sn_obj.photometry.bands,
                            peak_band=peak_band, kind=method.value, interpolator_arguments={"verbose": 0},
                            silence_warnings=silence_warnings,
                        )
                        preds = interpolator.predict_from_peak((-20, 100))

                        plotter = _create_predictions_plot(sn=sn_obj, bandset=bs, predictions=preds, interpolator=interpolator)
                        plotter.write_image(path=out_dir_plots / f"{val_res.sn_name}_{bs.value}.png")

                        res_df = plotter.int_result.to_dataframe()
                        if out_data_path.exists():
                            old_df = pl.read_parquet(out_data_path)
                            old_df.vstack(res_df, in_place=True)
                            old_df.write_parquet(out_data_path)
                        else:
                            res_df.write_parquet(out_data_path)

                        success = True
                    except ValueError as ex:
                        print(f"Error: sn={val_res.sn_name} bs={bs} msg={ex}")
                        pass
                if not success:
                    skipped.append((val_res.sn_name, f"failed to interpolate (bandset={bs})"))
        except DataPointNotFoundError:
            skipped.append((val_res.sn_name, "sn not found in dataset"))

    click.echo(f"Skipped {len(skipped)} candidates:")
    for sn_name, reason in skipped:
        click.echo(f"- {sn_name}: {reason}")

    click.echo(f"Results saved to `{out_dir_root}`")
