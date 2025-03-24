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

        if data.photometry.bands.nr_observations >= self.min_observations * 3:  # 3 bands in a bandset (e.g.: BRI)
            for band in data.photometry.bands.get_bands():
                if 0 < band.nr_observations < self.min_observations:
                    return ValidationResult.invalid(
                        f"Band [{band.name}] has only {band.nr_observations} observation(s) "
                        f"(minimum: {self.min_observations})",
                        self.name,
                    )

        return ValidationResult.valid(self.name)
