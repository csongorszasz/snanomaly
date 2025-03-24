from attrs import define, field

from snanomaly.models.sncandidate.sncandidate import SNCandidate
from snanomaly.preprocessing.cleaning.checks.check import Check
from snanomaly.preprocessing.cleaning.validation_result import ValidationResult


@define
class MinimumObservationsPerBand(Check):
    min_observations: int = field(default=3)

    def validate(self, data: SNCandidate) -> ValidationResult:
        if not data.photometry:
            return ValidationResult.invalid("No photometry data available", self.name)

        # Group observations by band
        bands = {}
        for phot in data.photometry:
            if phot.band not in bands:
                bands[phot.band] = 0
            bands[phot.band] += 1

        # Check each band has minimum observations
        for band, count in bands.items():
            if count < self.min_observations:
                return ValidationResult.invalid(
                    f"Band [{band}] has only {count} observations (minimum: {self.min_observations})",
                    self.name,
                )

        return ValidationResult.valid(self.name)
