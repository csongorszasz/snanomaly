from datetime import datetime

import click
import polars as pl
from tqdm import tqdm

from snanomaly import dirs
from snanomaly.dataset.exception import DataPointNotFoundError
from snanomaly.dataset.factory import OSCFactory
from snanomaly.models.results.validation_result import ValidationResult
from snanomaly.models.sncandidate import Bandset
from snanomaly.preprocessing.cleaning.checks.photometry import MinimumObservationsPerBand
from snanomaly.preprocessing.cleaning.validation_pipeline import ValidationPipeline


@click.command()
@click.option("-s", "--sn_name", default=None, type=str)
@click.option("-d", "--dataset", default=None, type=click.Choice(["osc2018_june", "osc2022"]), required=True,
              help="Dataset to get objects from.")
@click.option("--min-obs", type=int, default=3, show_default=True, help="Validation attribute: minimum number of observations per band with a 3-day binning.")
@click.option("--stop-after", default=None, type=int, help="Stop after processing this many candidates")
def validate(sn_name: str, dataset: str, min_obs: int, stop_after: int):
    pipeline = ValidationPipeline(
        checks=[
            MinimumObservationsPerBand(min_observations=min_obs, bandsets=[Bandset.BRI, Bandset.gri, Bandset.gri_primed]),
        ],
        fail_fast=True,
    )

    ds = OSCFactory.get(dataset)
    if sn_name:
        try:
            sn_obj = ds.load_datapoint(name=sn_name)
            pipeline.validate(sn_obj)
            pipeline.print_results(only_errors=False, printer_func=click.echo)
        except DataPointNotFoundError:
            click.echo(message=f"Could not find supernova candidate with name `{sn_name}`", err=True)
    else:
        run_id = f"{dataset}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        out_path = dirs.VALIDATED / f"{run_id}.parquet"

        batch_size = 12
        candidates = ds.load_dataset(batch_size=batch_size, stop_after=stop_after)
        valid_candidates = pipeline.filter_valid(candidates)

        # Write batches one by one to avoid memory bloating
        cnt_valid = 0
        for i, batch in enumerate(
            tqdm(valid_candidates, desc=f"Validating batches of {batch_size}", total=ds.nr_datapoints // batch_size),
        ):
            cnt_valid += len(batch)

            batch_data = [
                ValidationResult(
                    candidate.name, candidate.photometry.bands.available_bandsets,
                ).to_dict_dataframe_ready()
                for candidate in batch
            ]

            if batch_data:
                batch_df = pl.DataFrame(batch_data)
                if out_path.exists():
                    old_df = pl.read_parquet(out_path)
                    (old_df.vstack(batch_df, in_place=True).write_parquet(out_path))
                else:
                    batch_df.write_parquet(out_path)

        upper_bound = stop_after if stop_after is not None else ds.nr_datapoints
        print(
            f"No. valid candidates: {cnt_valid}/{upper_bound} ({cnt_valid / upper_bound * 100:.2f}%)",
        )
