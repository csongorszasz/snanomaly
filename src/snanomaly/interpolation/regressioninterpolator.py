import numpy as np
import sklearn
from attrs import define, field

from snanomaly.interpolation.baseinterpolator import BaseInterpolator
from snanomaly.models.sncandidate.bands import BandEnum


@define
class RegressionInterpolator(BaseInterpolator):
    regressor: sklearn.base.BaseEstimator = field(default=None)

    def _find_predicted_peak_time(self) -> float:
        pass

    def train(self):
        pass

    def _predict_explicit(self, x: np.ndarray, band: BandEnum = None, **kwargs) -> np.ndarray:
        pass

    def _predict_from_peak(self, prediction_interval_from_peak: tuple[int, int], **kwargs):
        pass

    def y_negative_to_zero_until_infinity(self, peak_idx: int, y: np.ndarray, **kwargs) -> np.ndarray:
        pass



