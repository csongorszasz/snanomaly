import json
import pathlib
from collections import Counter

import click
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from snanomaly import dirs
from snanomaly.visualization.outlier import PlotOutlier


@click.group()
def outlier():
    pass


def read_prepare_data(inpath: pathlib.Path) -> tuple[pl.DataFrame, pl.DataFrame, str]:
    """
    Read and prepare data from a Parquet file (expected output from dimreduce).
    Extracts features from the 'values' column and parses metadata from the filename.
    """
    click.echo(f"Loading dataframe from {inpath}...")
    input_df = pl.read_parquet(inpath)
    click.echo("OK")

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
    return input_df, features_for_model, dataset_name


def do_outlier_detection(
    inpath: pathlib.Path,
    model: IsolationForest | OneClassSVM,
    model_name: str,
    plot: bool,
    model_params_for_plot: dict,
    important_params: dict,
):
    """Core logic for outlier detection: load data, fit model, save results & plot."""
    if inpath.is_dir():
        # Find a parquet file in the directory
        parquet_files = list(inpath.glob("*.parquet"))
        if not parquet_files:
            click.echo(f"Error: No Parquet files found in directory {inpath}.")
            raise click.Abort
        inpath = parquet_files[0]
    elif inpath.suffix == ".txt":  # File with a list of directory names
        # Go through the list of directory names, find the belonging Parquet file and run the outlier detection on each of them
        with inpath.open() as f:
            dir_names = [line.strip() for line in f if line.strip()]
        for dir_name in dir_names:
            dir_path = dirs.DIMREDUCED / dir_name
            if not dir_path.is_dir():
                click.echo(f"Error: {dir_name} is not a directory.")
                raise click.Abort
            parquet_files = list(dir_path.glob("*.parquet"))
            if not parquet_files:
                click.echo(f"Error: No Parquet files found in directory {dir_name}.")
                raise click.Abort
            do_outlier_detection(parquet_files[0], model, model_name, plot, model_params_for_plot, important_params)
        return

    input_df, features_for_model, dataset_name = read_prepare_data(inpath)

    method_description = model_name
    for k, v in important_params.items():
        method_description += f"_{k}{v}"

    base_name = inpath.stem
    run_id = f"{base_name}_{method_description}"
    out_dir_root = dirs.ANOMALIES / run_id
    pathlib.Path.mkdir(out_dir_root, parents=True, exist_ok=True)

    out_img_path = out_dir_root / f"{run_id}.png"
    out_data_path = out_dir_root / f"{run_id}.parquet"

    click.echo(f"Running {method_description}...")
    features_np = features_for_model.to_numpy()

    pred_np = model.fit_predict(features_np)
    score_np = model.decision_function(features_np)

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
                model_params_display=model_params_for_plot,
            )
            plotter.write_image(out_img_path)
            plotter.show()
        else:
            click.echo(f"Plotting supported only for 2D features (found {features_for_model.shape[1]}). Skipping.")


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
    model = IsolationForest(**model_params, verbose=2)

    # For plotting, pass the user-provided string for contamination if it was 'auto'
    plot_params = model_params.copy()
    plot_params["contamination"] = contamination
    plot_params["max_samples"] = max_samples

    do_outlier_detection(inpath, model, "IsolationForest", plot, plot_params, important_params={"contam": contamination, "estimators": n_estimators, "maxsamples": max_samples})


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
    model = OneClassSVM(**model_params, verbose=True)

    # For plotting, pass the user-provided string for gamma if it was 'scale' or 'auto'
    plot_params = model_params.copy()
    plot_params["gamma"] = gamma

    do_outlier_detection(inpath, model, "OneClassSVM", plot, plot_params, important_params={"nu": nu, "gamma": gamma})


@outlier.command(help="List common outliers across detections.")
@click.option("-i", "--inpath", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path), required=True,
              help="Path to a TXT file containing a list of directories that all have a Parquet file with lists of detected outliers.")
@click.option("-m", "--min-threshold", type=int, default=2, show_default=True, help=
              "Minimum number of files in which an outlier must be detected to be considered common.")
def common_outliers(inpath: pathlib.Path, min_threshold: int):
    """List common outliers between multiple outlier detection runs."""
    with inpath.open() as f:
        dir_names = [line.strip() for line in f if line.strip()]

    if not dir_names:
        click.echo("Error: No directories found in the input file.")
        raise click.Abort

    all_outliers_flat_list = []
    num_files_processed = 0
    for dir_name in dir_names:
        dir_path = dirs.ANOMALIES / dir_name
        if not dir_path.is_dir():
            click.echo(f"Error: {dir_name} is not a directory.")
            raise click.Abort
        parquet_files = list(dir_path.glob("*.parquet"))
        if not parquet_files:
            click.echo(f"Error: No Parquet files found in directory {dir_name}.")
            raise click.Abort

        df = pl.read_parquet(parquet_files[0])
        outliers_in_file_df = df.filter(pl.col("outlier_pred") == -1)
        raw_rows = outliers_in_file_df.select(["sn_name", "bandset"]).rows()
        outliers_in_file = []
        for sn_name_val, bandset_val in raw_rows:
            hashable_bandset = "".join(bandset_val).replace("_pr", "'")
            outliers_in_file.append((sn_name_val, hashable_bandset))

        click.echo(
            f"Found {len(outliers_in_file)} outliers in `{dir_name}`: "
            f"{', '.join([f'({name}, {bandset})' for name, bandset in outliers_in_file[:3]])}"
            f"{'...' if len(outliers_in_file) > 3 else ''} "
            f"(total {len(outliers_in_file)})",
        )
        all_outliers_flat_list.extend(outliers_in_file)
        num_files_processed += 1

    if not all_outliers_flat_list:
        click.echo("No outliers found in any of the processed files.")
        return

    if num_files_processed == 0:
        click.echo("No valid outlier files processed.")
        return

    click.echo(f"Inspected {num_files_processed} files.")

    outlier_counts = Counter(all_outliers_flat_list)

    json_results = {}
    output_json_filename = dirs.COMMON_OUTLIERS / f"{inpath.stem}_common_outliers.json"
    pathlib.Path(output_json_filename.parent).mkdir(parents=True, exist_ok=True)

    if num_files_processed < min_threshold:
        if num_files_processed > 0:
            common_for_all = sorted(
                [outlier_tuple for outlier_tuple, count in outlier_counts.items() if count == num_files_processed],
            )
            click.echo(f"Common outliers across all {num_files_processed} files: {len(common_for_all)}")
            if common_for_all:
                for sn_name, bandset in common_for_all:
                    click.echo(f"- {sn_name} (bandset: {bandset})")
            json_results[f"common_in_all_{num_files_processed}_files"] = [list(o) for o in common_for_all]
        else:
            click.echo("No files were processed to find common outliers.")
    else:
        for k_threshold in range(min_threshold, num_files_processed + 1):
            current_common_outliers = sorted(
                [outlier_tuple for outlier_tuple, count in outlier_counts.items() if count >= k_threshold],
            )
            result_key = f"common_in_at_least_{k_threshold}_files"
            json_results[result_key] = [list(o) for o in current_common_outliers]
            click.echo(f"Common outliers across at least {k_threshold} files: {len(current_common_outliers)}")
            if current_common_outliers:
                for sn_name, bandset in current_common_outliers:
                    click.echo(f"- {sn_name} (bandset: {bandset})")
            click.echo("---")

    if json_results:
        with open(output_json_filename, "w") as f:
            json.dump(json_results, f, indent=4)
        click.echo(f"Common outlier results saved to `{output_json_filename}`")
    elif num_files_processed > 0:
        click.echo("No common outliers found meeting the min_threshold criteria.")

