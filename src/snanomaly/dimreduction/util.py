import numpy as np
import polars as pl


def square_distance(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    return np.sum((X1 - X2) ** 2)


def dimreduced_to_df(original_df: pl.DataFrame, X_reduced: np.ndarray) -> pl.DataFrame:
    X_df = pl.DataFrame(X_reduced).select(values=pl.concat_list(pl.all())).with_row_index()
    return original_df.select(["sn_name", "bandset"]).with_row_index().join(X_df, on="index").drop("index")
