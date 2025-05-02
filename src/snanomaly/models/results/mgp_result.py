from __future__ import annotations

import polars as pl
from attrs import define, field

from snanomaly.models.results.result import Result
from snanomaly.models.sncandidate import Bandset


@define
class MGPResult(Result):
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
    df = lf.collect(streaming=True)
    mgp_res = MGPResult.from_dataframe(df)
    print(mgp_res)
    # df_dict = df.to_dict()
    # print(json.dumps(df_dict, indent=4))
    # mgp_res = cattrs.structure(df.to_dict(), MGPResult)
    # print(mgp_res)


# write()
# read()
