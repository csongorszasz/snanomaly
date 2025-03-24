from __future__ import annotations

from abc import ABC, abstractmethod

from attrs import define

from snanomaly.models.sncandidate.sncandidate import SNCandidate
from snanomaly.preprocessing.cleaning.validation_result import ValidationResult


@define
class Check(ABC):
    """
    Abstract base class for data validation checks.
    """

    @property
    def name(self) -> str:
        """Return the name of the check for reporting purposes."""
        return self.__class__.__name__

    @abstractmethod
    def validate(self, data: SNCandidate) -> ValidationResult:
        """
        Validate the data and return a ValidationResult.
        """
        pass
