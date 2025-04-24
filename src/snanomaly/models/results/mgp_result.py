from __future__ import annotations

import cattrs
import polars as pl
from attrs import define, field

from snanomaly.models.sncandidate import Bandset


@define
class MGPResult:
    """
    Model for a Multivariate Gaussian Process interpolation result.
    The means and standard deviations of predictions are stored in `pred_means` and `pred_stds`.
    """

    sn_name: str = field()
    bandset: Bandset = field()
    days_pre_peak: int = field()
    days_post_peak: int = field()
    log_likelihood: float = field()
    thetas: list[float] = field()
    pred_means: dict = field()
    pred_stds: dict = field()

    def to_dataframe(self) -> pl.DataFrame:
        """Converting to Polars DataFrame with only one row being created."""
        data: dict = {}
        for key, value in cattrs.unstructure(self).items():
            data[key] = [value]
        return pl.DataFrame(data)

def write():
    print("Writing")
    res = MGPResult("a",
                    Bandset.gri, 20, 100,
                    -1.234,
                    [0.2, 0.5, 4.5],
                    {"g": [1.0, 2.0, 3.0], "r": [0.2, 1., 2.], "i": [1.,2.,3.]},
                    {"g": [0.2, 0.1, 0.15], "r": [0.2, 1., 2.], "i": [1.,0.11,0.04]},
                    )
    df = res.to_dataframe()
    df.write_parquet("test.parquet")
    print(df)

def read():
    print("Reading")
    lf = pl.scan_parquet("test.parquet")
    print(lf.collect(streaming=True))

# write()
# read()
