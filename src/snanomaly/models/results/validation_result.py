from __future__ import annotations

from attrs import define, field

from snanomaly.models.results.result import Result
from snanomaly.models.sncandidate import Bandset


@define
class ValidationResult(Result):
    """
    Model for a SNCandidate validation result.
    """

    sn_name: str = field()
    available_bandsets: set[Bandset] = field(factory=set)
