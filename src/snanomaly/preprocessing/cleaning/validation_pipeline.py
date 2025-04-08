from __future__ import annotations

import multiprocessing
from collections.abc import Generator

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
    results: list[ValidationResult] = field(factory=list, init=False)
    subject_name: str = field(factory=str, init=False)

    def add_check(self, check: Check) -> ValidationPipeline:
        """Add a check to the pipeline and return self for method chaining."""
        self.checks.append(check)
        return self

    def validate(self, data: SNCandidate) -> list[ValidationResult]:
        """
        Run all checks on the data and return results.
        If fail_fast is True, stop on the first failure.
        """
        self.subject_name = data.name
        self.results = []
        for check in self.checks:
            result = check.validate(data)
            self.results.append(result)
            if self.fail_fast and not result.is_valid:
                break
        return self.results

    def print_results(self, only_errors: bool = True):
        """Output the results of a `validate` run."""
        if self.results == []:
            raise ValueError("No results to print. Run `validate` first.")
        error_msgs = []
        for r in self.results:
            if not r.is_valid:
                error_msgs.append(f"Check failed: [check={r.check_name}; message={r.message}]")
        if error_msgs:
            print(f"### {self.subject_name} ###")
            print("\n".join(error_msgs))
        elif not only_errors:
            print(f"### {self.subject_name} ###")
            print("Passed.")

    def is_valid(self, data: SNCandidate) -> bool:
        """Return True if all checks pass."""
        return all(result.is_valid for result in self.validate(data))

    def filter_valid(self, candidates: Generator[list]) -> Generator[list]:
        """Filter a collection of candidates to only those that pass all checks."""
        for batch in candidates:
            batch_size = len(batch)
            with multiprocessing.Pool(batch_size) as p:
                valid_flags = p.map(self.is_valid, batch)
            valid_candidates = [cand for cand, is_valid in zip(batch, valid_flags, strict=False) if is_valid]
            yield valid_candidates
