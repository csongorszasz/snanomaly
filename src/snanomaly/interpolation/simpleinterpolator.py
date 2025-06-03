import numpy as np
import scipy
from attrs import define, field

from snanomaly.interpolation.baseinterpolator import BaseInterpolator
from snanomaly.models.sncandidate.bands import BandEnum


@define
class SimpleInterpolator(BaseInterpolator):
    kind: str = field(default=None)
    interpolators: dict[tuple[str, scipy.interpolate]] = field(factory=dict)

    def train(self):
        for band in self.bands_binned:
            self.interpolators[band.name] = scipy.interpolate.interp1d(x=band.time, y=band.flux, kind=self.kind,
                                                                       fill_value="extrapolate")

    def _find_predicted_peak_time(self) -> float:
        min_day = self.bands_binned[0].time.min()
        max_day = self.bands_binned[0].time.max()
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

    def y_negative_to_zero_until_infinity(self, peak_idx: int, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Relative to the peak point, finds the two closest points (1 to the left, 1 to the right) that are zero or
        negative and zeroes all subsequents values until infinity.
        """
        # Find first negative or zero value to the left of peak
        for j in range(peak_idx - 1, -1, -1):
            if y[j] <= 0:
                # Zero out all values to the left including this point
                y[:j+1] = 0
                break

        # Find first negative or zero value to the right of peak
        for j in range(peak_idx + 1, len(y)):
            if y[j] <= 0:
                # Zero out all values to the right including this point
                y[j:] = 0
                break

        return y
