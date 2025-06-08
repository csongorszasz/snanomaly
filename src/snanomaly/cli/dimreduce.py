import pathlib

import click
import polars as pl
from sklearn.manifold import TSNE

from snanomaly import dirs
from snanomaly.dimreduction.util import dimreduced_to_df, square_distance
from snanomaly.models.results.util import prepare_df_for_learning
from snanomaly.visualization.dimreduction import PlotDimreduction


@click.group()
def dimreduce():
    pass


@dimreduce.command()
@click.option("-i", "--inpath", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path), required=True,
              help="Path to the input data file (Parquet format) that contains interpolated light curves.")
@click.option("--dims", type=int, default=2, show_default=True, help="Target number of dimensions.")
@click.option("--perplexity", type=float, default=30.0, show_default=True, help="Perplexity.")
@click.option("--early-exaggeration", type=float, default=12.0, show_default=True, help="Early exaggeration.")
@click.option("--max-iter", type=int, default=1000, show_default=True, help="Maximum number of iterations.")
@click.option("--init", type=click.Choice(["random", "pca"]), default="pca",
              show_default=True, help="Initialization of embedding.")
@click.option("--method", type=click.Choice(["exact", "barnes_hut"]), default="exact",
              show_default=True, help="Method to use for gradient calculation.")
@click.option("--random-state", type=int, default=42, show_default=True, help="Random state for reproducibility.")
@click.option("--plot", is_flag=True, default=False, help="Create a plot of the reduced data.")
@click.option("-v", "--verbosity-level", default=2, type=int, show_default=True,
              help="Verbosity for the learning phase: 0 - no logs, 1 - some logs, 2 - detailed logs")
def tsne(inpath: str, dims: int, perplexity: float, early_exaggeration: float, max_iter: int, init: str, method: str,
         random_state: int, plot: bool, verbosity_level: int):
    click.echo("Loading dataframe...")
    df = pl.read_parquet(inpath)
    click.echo("OK")
    click.echo("Transforming data...")
    X = prepare_df_for_learning(df)
    click.echo("OK")
    click.echo(f"Prepared {X.shape[0]} objects with {X.shape[1]} dimensions")

    click.echo(f"Embedding into {dims} dimensions...")
    reducer = TSNE(
        n_components=dims,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        max_iter=max_iter,
        init=init,
        method=method,
        metric=square_distance,
        random_state=random_state,
        verbose=verbosity_level,
    )
    X_reduced = reducer.fit_transform(X)

    out_dir = dirs.DIMREDUCED / "TSNE"
    pathlib.Path.mkdir(out_dir, parents=True, exist_ok=True)
    out_path = out_dir / f"{inpath.stem}_TSNE_{dims}D.parquet"

    df_reduced = dimreduced_to_df(df, X_reduced)
    df_reduced.write_parquet(out_path)
    click.echo(f"Reduced data saved to {out_path}")

    if plot:
        if dims == 2:
            plotter = PlotDimreduction(X_reduced)
            plotter.show()
        else:
            click.echo("Plotting is only supported for 2D embeddings. Skipping plot generation.")

