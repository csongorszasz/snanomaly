from __future__ import annotations

from attrs import define, field

from snanomaly.models.sncandidate.sncandidate import SNCandidate
from snanomaly.preprocessing.cleaning.checks.check import Check
from snanomaly.preprocessing.cleaning.validation_result import ValidationResult


@define
class ValidationPipeline:
    """
    Manages a series of validation checks to be run on data.
    """

    checks: list[Check] = field(factory=list)
    fail_fast: bool = field(default=False)

    def add_check(self, check: Check) -> ValidationPipeline:
        """Add a check to the pipeline and return self for method chaining."""
        self.checks.append(check)
        return self

    def validate(self, data: SNCandidate) -> list[ValidationResult]:
        """
        Run all checks on the data and return results.
        If fail_fast is True, stop on the first failure.
        """
        results = []
        for check in self.checks:
            result = check.validate(data)
            results.append(result)
            if self.fail_fast and not result.is_valid:
                break
        return results

    def is_valid(self, data: SNCandidate) -> bool:
        """Return True if all checks pass."""
        return all(result.is_valid for result in self.validate(data))

    def filter_valid(self, candidates: list[SNCandidate]) -> list[SNCandidate]:
        """Filter a list of candidates to only those that pass all checks."""
        return [sn for sn in candidates if self.is_valid(sn)]
