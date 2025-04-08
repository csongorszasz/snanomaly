from __future__ import annotations

from pathlib import Path
from typing import Optional

from snanomaly import dirs
from snanomaly.dataset.osc import OSC


class OSCFactory:
    @classmethod
    def OSC2018June(cls, path: Optional[Path] = None):
        return OSC(
            path=dirs.DATASETS / "osc2018_june" if path is None else path,
            name="OSC-2018-June",
            description="The `Open Supernova Catalog` with data up to June, 2018."
                        "This exact dataset was used in the paper at https://arxiv.org/abs/1905.11516.",
            nr_datapoints=45162,
        )

    @classmethod
    def OSC2022(cls, path: Optional[Path] = None):
        return OSC(
            path=dirs.DATASETS / "osc2022" if path is None else path,
            name="OSC-2022",
            description="The `Open Supernova Catalog` with data up to 2022. "
                        "A merge between the `OSC-2018-June` dataset and all available data from the AstroCats "
                        "supernova repositories at: https://github.com/astrocatalogs/supernovae.",
            nr_datapoints=108921,
        )
