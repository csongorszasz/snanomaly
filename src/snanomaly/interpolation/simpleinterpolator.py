import numpy as np
import scipy
from attrs import define, field

from snanomaly.interpolation.baseinterpolator import BaseInterpolator
from snanomaly.interpolation.interpolatorfactory import InterpolatorFactory
from snanomaly.models.sncandidate.bands import BandEnum


@define
class SimpleInterpolator(BaseInterpolator):
    interpolators: dict[tuple[str, scipy.interpolate]] = field(factory=dict)

    def train(self):
        for band in self.bands_binned:
            self.interpolators[band.name] = InterpolatorFactory(x=band.time, y=band.flux, interpolator_type=self.kind,
                                                                **self.interpolator_arguments)

    def _get_time_array_for_peak_finding(self) -> np.ndarray:
        band_idx = self.get_band_index(self.peak_band, self.bandset)
        low = int(self.bands_binned[band_idx].time.min())
        high = int(self.bands_binned[band_idx].time.max())
        return np.linspace(low, high, high - low + 1)

    def predict_explicit(self, x: np.ndarray, band: BandEnum = None, **kwargs) -> np.ndarray:
        return self.interpolators[band.value](x)

    def _predict_from_peak(self, prediction_interval_from_peak: tuple[int, int], **kwargs):
        x = np.linspace(self.predicted_peak_time + prediction_interval_from_peak[0],
                        self.predicted_peak_time + prediction_interval_from_peak[1],
                        prediction_interval_from_peak[1] - prediction_interval_from_peak[0] + 1)
        preds = {}
        for band in self.bands_binned:
            pred = self.interpolators[band.name](x)
            pred = self.y_negative_to_zero_until_infinity(peak_idx=-prediction_interval_from_peak[0], y=pred)
            preds[band.name] = pred
        return preds
