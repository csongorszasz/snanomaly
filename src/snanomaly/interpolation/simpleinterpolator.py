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

    def _find_predicted_peak_time(self) -> float:
        min_day = self.bands_binned[self.get_band_index(self.peak_band, self.bandset)].time.min()
        max_day = self.bands_binned[self.get_band_index(self.peak_band, self.bandset)].time.max()
        days_range = max_day - min_day + 1
        x = np.linspace(min_day, max_day, int(days_range))
        y = self.interpolators[self.peak_band.value](x)
        peak_idx = np.argmax(y)
        return x[peak_idx]

    def _predict_explicit(self, x: np.ndarray, band: BandEnum = None, **kwargs) -> np.ndarray:
        pass

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
