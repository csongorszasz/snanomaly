import pathlib

import click
import polars as pl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from umap import UMAP

from snanomaly import dirs
from snanomaly.dimreduction.util import dimreduced_to_df, square_distance
from snanomaly.models.results.util import prepare_df_for_learning
from snanomaly.visualization.dimreduction import PlotDimreduction


@click.group()
def dimreduce():
    pass


def do_dim_reduction(inpath: pathlib.Path, reducer, dims: int, method_label: str, plot: bool):
    data_df, features = read_prepare(inpath)

    run_id = f"{inpath.stem}_{method_label}_{dims}D"
    out_dir_root = dirs.DIMREDUCED / run_id
    pathlib.Path.mkdir(out_dir_root, parents=True, exist_ok=True)
    out_img_path = out_dir_root / f"{run_id}.png"
    out_data_path = out_dir_root / f"{run_id}.parquet"

    features_reduced = embed_save(
        inpath=inpath,
        features=features,
        reducer=reducer,
        dims=dims,
        out_path=out_data_path,
    )
    if plot:
        plot_save(
            features_reduced=features_reduced,
            labels=data_df.get_column("sn_name").to_list(),
            title=method_label,
            out_path=out_img_path,
        )


def read_prepare(inpath: pathlib.Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Read and prepare the input data for dimensionality reduction.
    """
    click.echo("Loading dataframe...")
    data_df = pl.read_parquet(inpath)
    click.echo("OK")
    click.echo("Transforming data...")
    features = prepare_df_for_learning(data_df)
    click.echo("OK")
    click.echo(f"Prepared {features.shape[0]} objects with {features.shape[1]} dimensions")
    return data_df, features


def embed_save(
    inpath: pathlib.Path,
    features: pl.DataFrame,
    reducer,
    dims: int,
    out_path: pathlib.Path,
) -> pl.DataFrame:
    """
    Embed the features into lower dimensions and save the result.
    """
    click.echo(f"Embedding into {dims} dimensions...")
    features_reduced = reducer.fit_transform(features)

    df_reduced = dimreduced_to_df(pl.read_parquet(inpath), features_reduced)
    df_reduced.write_parquet(out_path)

    click.echo(f"Reduced data saved to `{out_path}`")

    return features_reduced

def plot_save(
    features_reduced: pl.DataFrame,
    labels: list[str],
    title: str,
    out_path: pathlib.Path,
):
    """
    Create a plot of the reduced data and show it.
    """
    if features_reduced.shape[1] == 2:
        plotter = PlotDimreduction(
            features_reduced,
            labels=labels,
        )
        plotter.set_title(title)
        plotter.write_image(path=out_path)
        click.echo(f"Plot saved to `{out_path}`")
    else:
        click.echo("Plotting is only supported for 2D embeddings. Skipping plot generation.")


@dimreduce.command()
@click.option(
    "-i",
    "--inpath",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    required=True,
    help="Path to the input data file (Parquet format) that contains interpolated light curves.",
)
@click.option("--dims", type=int, default=2, show_default=True, help="Target number of dimensions.")
@click.option("--random-state", type=int, default=42, show_default=True, help="Random state for reproducibility.")
@click.option("--plot", is_flag=True, default=False, help="Create a plot of the reduced data.")
def pca(inpath: pathlib.Path, dims: int, random_state: int, plot: bool):
    do_dim_reduction(
        inpath=inpath,
        reducer=PCA(
            n_components=dims,
            random_state=random_state,
        ),
        dims=dims,
        method_label="PCA",
        plot=plot,
    )

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
@click.option(
    "-v",
    "--verbosity-level",
    default=2,
    type=int,
    show_default=True,
    help="Verbosity for the learning phase: 0 - no logs, 1 - some logs, 2 - detailed logs",
)
def tsne(
    inpath: pathlib.Path,
    dims: int,
    perplexity: float,
    early_exaggeration: float,
    max_iter: int,
    init: str,
    method: str,
    random_state: int,
    plot: bool,
    verbosity_level: int,
):
    do_dim_reduction(
        inpath=inpath,
        reducer=TSNE(
            n_components=dims,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            max_iter=max_iter,
            init=init,
            method=method,
            metric=square_distance,
            random_state=random_state,
            verbose=verbosity_level,
        ),
        dims=dims,
        method_label="TSNE",
        plot=plot,
    )

@dimreduce.command()
@click.option(
    "-i",
    "--inpath",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    required=True,
    help="Path to the input data file (Parquet format) that contains interpolated light curves.",
)
@click.option("--dims", type=int, default=2, show_default=True, help="Target number of dimensions.")
@click.option("--n-neighbors", type=int, default=15, show_default=True, help="Number of neighbors.")
@click.option(
    "--min-dist", type=float, default=0.1, show_default=True, help="Minimum distance between embedded points.",
)
@click.option(
    "--metric", type=str, default="euclidean", show_default=True, help="Metric to use for distance computation.",
)
@click.option("--random-state", type=int, default=42, show_default=True, help="Random state for reproducibility.")
@click.option("--plot", is_flag=True, default=False, help="Create a plot of the reduced data.")
@click.option(
    "-v", "--verbosity-level", default=False, type=bool, show_default=True, help="Verbosity for the learning phase.",
)
def umap(
    inpath: pathlib.Path,
    dims: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
    plot: bool,
    verbosity_level: bool,
):
    do_dim_reduction(
        inpath=inpath,
        reducer=UMAP(
            n_components=dims,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            verbose=verbosity_level,
        ),
        dims=dims,
        method_label="UMAP",
        plot=plot,
    )


@dimreduce.command()
@click.option(
    "-i",
    "--inpath",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    required=True,
    help="Path to the input data file (Parquet format) that contains interpolated light curves.",
)
@click.option("--dims", type=int, default=2, show_default=True, help="Target number of dimensions.")
@click.option(
    "--n-neighbors", type=int, default=5, show_default=True, help="Number of neighbors to consider for each point.",
)
@click.option("--plot", is_flag=True, default=False, help="Create a plot of the reduced data.")
def isomap(inpath: pathlib.Path, dims: int, n_neighbors: int, plot: bool):
    do_dim_reduction(
        inpath=inpath,
        reducer=Isomap(
            n_components=dims,
            n_neighbors=n_neighbors,
        ),
        dims=dims,
        method_label="ISOMAP",
        plot=plot,
    )


@dimreduce.command()
@click.option(
    "-i",
    "--inpath",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    required=True,
    help="Path to the input data file (Parquet format) that contains interpolated light curves.",
)
@click.option("--dims", type=int, default=2, show_default=True, help="Target number of dimensions.")
@click.option(
    "--n-neighbors", type=int, default=5, show_default=True, help="Number of neighbors to consider for each point.",
)
@click.option("--random-state", type=int, default=42, show_default=True, help="Random state for reproducibility.")
@click.option("--plot", is_flag=True, default=False, help="Create a plot of the reduced data.")
@click.option(
    "--method",
    type=click.Choice(["standard", "hessian", "modified", "ltsa"]),
    default="standard",
    show_default=True,
    help="Method to use for LLE.",
)
def lle(inpath: pathlib.Path, dims: int, n_neighbors: int, random_state: int, plot: bool, method: str):
    do_dim_reduction(
        inpath=inpath,
        reducer=LocallyLinearEmbedding(
            n_components=dims,
            n_neighbors=n_neighbors,
            random_state=random_state,
            method=method,
        ),
        dims=dims,
        method_label="LLE",
        plot=plot,
    )
