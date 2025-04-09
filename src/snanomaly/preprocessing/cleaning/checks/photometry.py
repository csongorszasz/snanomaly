from __future__ import annotations

from attrs import define, field

from snanomaly.models.sncandidate import Bandset
from snanomaly.models.sncandidate.sncandidate import SNCandidate
from snanomaly.preprocessing.cleaning.checks.check import Check
from snanomaly.preprocessing.cleaning.validation_result import ValidationResult


@define
class MinimumObservationsPerBand(Check):
    """
    If band sets are provided, then this check is successful already if the criterion is fulfilled by at least
    one band set.

    Example:
    -------
    bandsets = [["B","R","I"], ["g","r","i"]]
        -> if the check fails for `BRI` but succeeds for `gri`, then it's valid

        -> if both `BRI` and `gri` fail, then it's invalid

    """

    min_observations: int = field(default=3)
    bandsets: list[Bandset] = field(factory=list)  # empty list means all bands will be checked

    def validate(self, data: SNCandidate) -> ValidationResult:
        if not data.photometry:
            return ValidationResult.invalid("No photometry data available", self.name)

        error_msgs = []
        for bandset in self.bandsets:
            ok, msg = self._validate_bandset(data, bandset)
            if not ok:
                error_msgs.append(msg)
            else:
                data.photometry.bands.add_to_available_bandsets(bandset)
        if len(error_msgs) >= len(self.bandsets):
            return ValidationResult.invalid("\n".join(error_msgs), self.name)
        return ValidationResult.valid(self.name) # TODO: establish logic for when to consider a candidate invalid

    def _validate_bandset(self, data: SNCandidate, bandset: Bandset) -> tuple[bool, str]:
        for band in data.photometry.bands.get_bands(bandset=bandset):
            binned = band.binned(bin_width=3, discrete_time=True)
            if binned.nr_observations < self.min_observations:
                return False, (f"Band [{band.name}] has only {band.nr_observations} observation(s) "
                               f"(minimum: {self.min_observations})")
        return True, ""
