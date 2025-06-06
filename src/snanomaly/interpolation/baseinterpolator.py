from abc import ABC, abstractmethod

import numpy as np
from attrs import define, field

from snanomaly.interpolation.exception import BandNotFoundError, NegativePeakIndexError, PeakTimeNotSetError
from snanomaly.models.sncandidate import Bandset
from snanomaly.models.sncandidate.band import Band
from snanomaly.models.sncandidate.bands import BandEnum, Bands


@define
class BaseInterpolator(ABC):
    sn_name: str = field()
    bandset: Bandset = field()
    bands: Bands = field()
    peak_band: BandEnum = field()
    kind: str = field()
    interpolator_arguments: dict = field(factory=dict)
    random_state: int = field(default=42)
    bands_binned: list[Band] = field(init=False)
    _predicted_peak_time: float = field(init=False, default=None)

    def __attrs_post_init__(self):
        self.bands_binned = [
            band.binned(bin_width=1) if not band.is_binned else band for band in self.bands.get_bands(self.bandset)
        ]
        self.train()

    @property
    def nr_bands(self):
        return len(self.bandset.value)

    @property
    def predicted_peak_time(self) -> float:
        if self._predicted_peak_time:
            return self._predicted_peak_time
        return self._find_predicted_peak_time()

    @classmethod
    def get_interval_relative_to_peak(cls, peak_time: float, days_pre: int, days_post: int) -> np.array:
        return np.linspace(peak_time - days_pre, peak_time + days_post, days_pre + days_post + 1)

    @classmethod
    def get_band_index(cls, band: BandEnum, bandset: Bandset) -> int:
        try:
            return bandset.value.index(band.value)
        except ValueError:
            raise BandNotFoundError(f"Band `{band}` not found in bandset `{bandset}`")

    @abstractmethod
    def _find_predicted_peak_time(self) -> float:
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def _predict_explicit(self, x: np.ndarray, band: BandEnum = None, **kwargs) -> np.ndarray:
        pass

    def predict_explicit(self, x_low: float, x_high: float, band: BandEnum = None, **kwargs) -> np.ndarray:
        """
        Predict y values independently of peak point.
        If specified, only return predictions to one band.
        """
        x = np.linspace(x_low, x_high, int(x_high - x_low) + 1)
        return self._predict_explicit(x=x, band=band, kwargs=kwargs)

    @abstractmethod
    def _predict_from_peak(self, prediction_interval_from_peak: tuple[int, int], **kwargs):
        pass

    def predict_from_peak(self, prediction_interval_from_peak: tuple[int, int], **kwargs) -> dict | tuple[dict, dict]:
        """Returns the predicted light curve and optionally the standard deviation of the prediction if applicable."""
        if self.predicted_peak_time is None:
            raise PeakTimeNotSetError("Set time of peak brightness before making predictions")

        return self._predict_from_peak(prediction_interval_from_peak, kwargs=kwargs)

    def y_negative_to_zero_until_infinity(self, peak_idx: int, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Relative to the peak point, finds the two closest points (1 to the left, 1 to the right) that are zero or
        negative and zeroes all subsequents values until infinity.
        """
        if peak_idx < 0:
            raise NegativePeakIndexError(f"Invalid peak_idx: `{peak_idx}`")

        # Find first negative or zero value to the left of peak
        for j in range(peak_idx - 1, -1, -1):
            if y[j] <= 0:
                # Zero out all values to the left including this point
                y[: j + 1] = 0
                break

        # Find first negative or zero value to the right of peak
        for j in range(peak_idx + 1, len(y)):
            if y[j] <= 0:
                # Zero out all values to the right including this point
                y[j:] = 0
                break

        return y
