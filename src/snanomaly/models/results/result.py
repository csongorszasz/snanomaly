from __future__ import annotations

import cattrs
import polars as pl
from attrs import define


@define
class Result:
    def to_dict_dataframe_ready(self) -> dict:
        """Serialize to dict such that it is ready to be passed to a dataframe constructor."""
        data: dict = {}
        for key, value in cattrs.unstructure(self).items():
            if isinstance(value, set):
                data[key] = list(value)
            else:
                data[key] = value
        return data

    def to_dataframe(self) -> pl.DataFrame:
        """Converting to Polars DataFrame with only one row being created."""
        return pl.DataFrame(self.to_dict_dataframe_ready())

    @classmethod
    def from_dict(cls, row: dict) -> Result:
        """Converting dict to Result object."""
        return cattrs.structure(row, cls)

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame) -> list[Result]:
        """Converting Polars DataFrame to a list of Result objects."""
        results = []
        for row in df.iter_rows(named=True):
            result = cls.from_dict(row)
            results.append(result)
        return results
