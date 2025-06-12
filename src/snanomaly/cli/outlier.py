import pathlib

import click
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from snanomaly import dirs
from snanomaly.visualization.outlier import PlotOutlier


@click.group()
def outlier():
    pass


def read_prepare_data(inpath: pathlib.Path) -> tuple[pl.DataFrame, pl.DataFrame, str, str, str]:
    """
    Read and prepare data from a Parquet file (expected output from dimreduce).
    Extracts features from the 'values' column and parses metadata from the filename.
    """
    click.echo(f"Loading dataframe from {inpath}...")
    input_df = pl.read_parquet(inpath)
    click.echo("OK")

    stem_parts = inpath.stem.split("_")
    dataset_name = "UnknownDataset"
    dim_red_method = "UnknownMethod"
    dims_str = "UnknownDims"

    # Heuristic parsing: e.g., originalfile_PCA_2D -> dataset=originalfile, method=PCA, dims=2D
    if len(stem_parts) >= 3:
        if stem_parts[-1][:-1].isdigit() and stem_parts[-1].endswith("D"):
            dims_str = stem_parts[-1]
            dim_red_method = stem_parts[-2].upper()
            dataset_name = "_".join(stem_parts[:-2])
        elif (
            len(stem_parts) >= 4
            and stem_parts[-2][:-1].isdigit()
            and stem_parts[-2].endswith("D")
        ):  # e.g. ..._PCA_2D_results
            dims_str = stem_parts[-2]
            dim_red_method = stem_parts[-3].upper()
            dataset_name = "_".join(stem_parts[:-3])
        else:
            dataset_name = inpath.stem
    else:
        dataset_name = inpath.stem

    if not dataset_name:  # Handle cases where parsing might lead to empty dataset_name
        dataset_name = inpath.stem

    click.echo("Extracting features for outlier detection...")
    if "values" not in input_df.columns:
        click.echo("Error: 'values' column not found. Input must be Parquet from dimreduce step.")
        raise click.Abort

    features_list = input_df["values"].to_list()
    if not features_list or not isinstance(features_list[0], list):
        click.echo("Error: 'values' column is not a list of lists.")
        raise click.Abort

    num_dims = len(features_list[0])
    feature_column_names = [f"dim_{i+1}" for i in range(num_dims)]
    features_for_model = pl.DataFrame(features_list, schema=feature_column_names)

    click.echo(f"OK. Prepared {features_for_model.shape[0]} objects, {features_for_model.shape[1]} dims.")
    return input_df, features_for_model, dataset_name, dim_red_method, dims_str


def do_outlier_detection(
    inpath: pathlib.Path,
    model: IsolationForest | OneClassSVM,
    model_name: str,
    plot: bool,
    model_params_for_plot: dict,
):
    """Core logic for outlier detection: load data, fit model, save results & plot."""
    input_df, features_for_model, dataset_name, dim_red_method, dims_str = read_prepare_data(inpath)

    base_name = inpath.stem
    run_id = f"{base_name}_{model_name}"
    out_dir_root = dirs.ANOMALIES / run_id
    pathlib.Path.mkdir(out_dir_root, parents=True, exist_ok=True)

    out_img_path = out_dir_root / f"{run_id}_plot.png"
    out_data_path = out_dir_root / f"{run_id}_results.parquet"

    click.echo(f"Running {model_name}...")
    features_np = features_for_model.to_numpy()

    pred_np = model.fit_predict(features_np)
    score_np = model.decision_function(features_np)

    # Ensure sn_name is present for plotting, even if it's just generated indices
    if "sn_name" not in input_df.columns:
        sn_names_for_plot = [f"P{i}" for i in range(features_np.shape[0])]
    else:
        sn_names_for_plot = input_df["sn_name"].to_list()


    results_df = input_df.with_columns([
        pl.Series(name="outlier_pred", values=pred_np),
        pl.Series(name="outlier_score", values=score_np),
    ])
    results_df.write_parquet(out_data_path)
    click.echo(f"Outlier detection results saved to `{out_data_path}`")

    if plot:
        if features_for_model.shape[1] == 2:
            plotter = PlotOutlier(
                features_2d_np=features_np,
                pred_np=pred_np,
                score_np=score_np,
                sn_names=sn_names_for_plot,
                model_name=model_name,
                model_object=model,
                dataset_name=dataset_name,
                dim_red_method=dim_red_method,
                dims_str=dims_str,
                model_params_display=model_params_for_plot,
            )
            plotter.write_image(out_img_path)
            # click.echo is now handled by plotter.write_image
        else:
            click.echo(f"Plotting only for 2D features (found {features_for_model.shape[1]}). Skipping.")


@outlier.command()
@click.option("-i", "--inpath", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path), required=True,
              help="Path to Parquet file from dimensionality reduction (must contain 'values' column).")
@click.option("--n-estimators", type=int, default=1000, show_default=True,
              help="Number of base estimators for IsolationForest.")
@click.option("--contamination", default="auto", show_default=True,
              help="Contamination factor for IsolationForest (float or 'auto').")
@click.option("--max-samples", default="auto", show_default=True,
              help="Number/fraction of samples for individual estimators in IsolationForest.")
@click.option("--max-features", type=float, default=1.0, show_default=True,
              help="Number/fraction of features for individual estimators.")
@click.option("--random-state", type=int, default=42, show_default=True, help="Random state for reproducibility.")
@click.option("--plot", is_flag=True, default=False, help="Generate a plot of the results (only for 2D features).")
def isoforest(
    inpath: pathlib.Path,
    n_estimators: int,
    contamination: str,
    max_samples: str,
    max_features: float,
    random_state: int,
    plot: bool,
):
    """Detect outliers using Isolation Forest."""
    try:
        contam_float = float(contamination)
    except ValueError:
        if contamination != "auto":
            raise click.BadParameter("Contamination must be a float or 'auto'.")
        contam_float = contamination # Keep as 'auto' string for sklearn

    try: # max_samples can be int or float or 'auto'
        if max_samples != "auto":
            max_samples_parsed = int(max_samples) if float(max_samples) >= 1.0 else float(max_samples)
        else:
            max_samples_parsed = "auto"
    except ValueError:
        raise click.BadParameter("max_samples must be an int, float, or 'auto'.")


    model_params = {
        "n_estimators": n_estimators, "contamination": contam_float,
        "max_samples": max_samples_parsed, "max_features": max_features,
        "random_state": random_state, "bootstrap": False, # Default in sklearn
    }
    model = IsolationForest(**model_params)

    # For plotting, pass the user-provided string for contamination if it was 'auto'
    plot_params = model_params.copy()
    plot_params["contamination"] = contamination
    plot_params["max_samples"] = max_samples

    do_outlier_detection(inpath, model, "IsolationForest", plot, plot_params)


@outlier.command()
@click.option("-i", "--inpath", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path), required=True,
              help="Path to Parquet file from dimensionality reduction (must contain 'values' column).")
@click.option("--kernel", type=click.Choice(["linear", "poly", "rbf", "sigmoid"]), default="rbf", show_default=True,
              help="Kernel type for OneClassSVM.")
@click.option("--nu", type=float, default=0.05, show_default=True,
              help="An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors (0 < nu <= 1).")
@click.option("--gamma", default="scale", show_default=True,
              help="Kernel coefficient for 'rbf', 'poly', 'sigmoid' ('scale', 'auto', or float).")
@click.option("--degree", type=int, default=3, show_default=True, help="Degree for 'poly' kernel.")
@click.option("--coef0", type=float, default=0.0, show_default=True,
              help="Independent term in kernel function (for 'poly', 'sigmoid').")
@click.option("--plot", is_flag=True, default=False, help="Generate a plot of the results (only for 2D features).")
def oneclasssvm(inpath: pathlib.Path, kernel: str, nu: float, gamma: str, degree: int, coef0: float, plot: bool):
    """Detect outliers using One-Class SVM."""
    if not (0 < nu <= 1):
        raise click.BadParameter("nu must be between 0 (exclusive) and 1 (inclusive).")

    try:
        gamma_parsed = float(gamma)
    except ValueError:
        if gamma not in ["scale", "auto"]:
            raise click.BadParameter("gamma must be a float, 'scale', or 'auto'.")
        gamma_parsed = gamma # Keep as string for sklearn

    model_params = {
        "kernel": kernel, "nu": nu, "gamma": gamma_parsed,
        "degree": degree, "coef0": coef0,
    }
    model = OneClassSVM(**model_params)

    # For plotting, pass the user-provided string for gamma if it was 'scale' or 'auto'
    plot_params = model_params.copy()
    plot_params["gamma"] = gamma

    do_outlier_detection(inpath, model, "OneClassSVM", plot, plot_params)

