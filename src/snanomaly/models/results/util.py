import polars as pl

from snanomaly.models.results.exception import EmptyDataFrameError, UnexpectedDataFrameColumnError


def rename_numbered_columns(df: pl.DataFrame, new_col_prefix: str) -> pl.DataFrame:
    numeric_cols = [col for col in df.columns if col.isdigit()]
    rename_dict = {col: f"{new_col_prefix}_{col}" for col in numeric_cols}
    return df.rename(rename_dict)

def explode_lists_to_numbered_columns(df: pl.DataFrame, target_col: str) -> pl.DataFrame:
    """Flattens lists of up to 2D lists into columns."""
    # get list dimensions
    first_row = df[target_col][0]
    if isinstance(first_row, pl.Series):
        if isinstance(first_row[0], float):
            num_lists = 1
            list_length = len(first_row)
        elif isinstance(first_row[0], pl.Series):
            num_lists = len(first_row)
            list_length = len(first_row[0])
        else:
            raise UnexpectedDataFrameColumnError(f"Column `{target_col}` must be of type 1D list or 2D list.")
    else:
        raise EmptyDataFrameError(f"No rows in data frame with columns: `{df.columns}`")
    # explode lists until there are no more lists
    nr_dimensions = 1 if num_lists == 1 else 2
    for i in range(nr_dimensions):
        df = df.explode(columns=[target_col])
    df = df.with_row_index(name="col_nr").with_columns(pl.col("col_nr").mod(num_lists * list_length)).pivot(on="col_nr", values=[target_col])
    return rename_numbered_columns(df, target_col)
