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

    def train(self):
        for band in self.bands_binned:
            band.normalize()
            self.regressors[band.name] = RegressorFactory(regressor_type=self.kind, band=band, **self.interpolator_arguments)
            band.time = band.time.reshape(-1, 1)
            self.regressors[band.name].fit(band.time, band.flux)

    def _get_time_array_for_peak_finding(self) -> np.ndarray:
        band_idx = self.get_band_index(self.peak_band, self.bandset)

        observed_max_idx = self.bands_binned[band_idx].flux.argmax()
        observed_max_time = self.bands_binned[band_idx].time[observed_max_idx]

        low = int(observed_max_time - 120)
        high = int(observed_max_time + 120)

        return np.linspace(low, high, high - low + 1).reshape(-1, 1)

    def predict_explicit(self, x: np.ndarray, band: BandEnum = None, **kwargs) -> np.ndarray:
        return self.regressors[band.value].predict(x)

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
