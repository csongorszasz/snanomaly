import numpy as np
from attrs import define, field
from sklearn.base import RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor

from snanomaly.interpolation.baseinterpolator import BaseInterpolator
from snanomaly.interpolation.regressorfactory import RegressorFactory
from snanomaly.models.sncandidate.bands import BandEnum


@define
class RegressionInterpolator(BaseInterpolator):
    regressors: dict[tuple[str, RegressorMixin]] = field(init=False, factory=dict)
    norm_factors: dict[tuple[str, float]] = field(init=False, default=None)

    def _find_predicted_peak_time(self) -> float:
        min_day = self.bands_binned[self.get_band_index(self.peak_band, self.bandset)].time.min()
        max_day = self.bands_binned[self.get_band_index(self.peak_band, self.bandset)].time.max()
        days_range = max_day - min_day + 1
        x = np.linspace(min_day, max_day, int(days_range)).reshape(-1, 1)
        y = self.regressors[self.peak_band.value].predict(x)
        peak_idx = np.argmax(y)
        return x[peak_idx]

    def train(self):
        for band in self.bands_binned:
            self.regressors[band.name] = RegressorFactory(regressor_type=self.kind, band=band, **self.interpolator_arguments)
            band.normalize()
            band.time = band.time.reshape(-1, 1)
            self.regressors[band.name].fit(band.time, band.flux)

    def _predict_explicit(self, x: np.ndarray, band: BandEnum = None, **kwargs) -> np.ndarray:
        pass

    def _predict_from_peak(self, prediction_interval_from_peak: tuple[int, int], **kwargs):
        x = np.linspace(
            self.predicted_peak_time + prediction_interval_from_peak[0],
            self.predicted_peak_time + prediction_interval_from_peak[1],
            prediction_interval_from_peak[1] - prediction_interval_from_peak[0] + 1,
        )
        preds = {}
        stds = {}
        for band in self.bands_binned:
            regressor = self.regressors[band.name]
            if isinstance(regressor, GaussianProcessRegressor):
                pred, std = regressor.predict(x, return_std=True)
                std = std * band.norm_factor if band.norm_factor else std  # denormalize
                stds[band.name] = std
            else:
                pred = regressor.predict(x)
            pred = self.y_negative_to_zero_until_infinity(peak_idx=-prediction_interval_from_peak[0], y=pred)
            pred = pred * band.norm_factor if band.norm_factor else pred  # denormalize
            preds[band.name] = pred
        return preds, stds
