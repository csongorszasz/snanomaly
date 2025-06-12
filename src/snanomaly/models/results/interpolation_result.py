from __future__ import annotations

import numpy as np
from attrs import define, field

from snanomaly.models.results.result import Result
from snanomaly.models.sncandidate import Bandset


@define
class InterpolationResult(Result):
    sn_name: str = field()
    bandset: Bandset = field()
    peak_time: float = field()
    days_pre_peak: int = field()
    days_post_peak: int = field()
    pred_means: list[np.ndarray] = field()
    pred_stds: list[np.ndarray] = field()
