from __future__ import annotations

from typing import Optional

from attrs import define, field

from snanomaly.models.sncandidate import Bandset


@define
class ValidationResult:
    is_valid: bool = field(default=True)
    message: Optional[str] = field(default=None)
    check_name: str = field(default="")
    available_bandsets: list[Bandset] = field(factory=list)

    @classmethod
    def valid(cls, check_name: str, available_bandsets: list[Bandset]) -> ValidationResult:
        return cls(True, None, check_name, available_bandsets)

    @classmethod
    def invalid(cls, message: str, check_name: str) -> ValidationResult:
        return cls(False, message, check_name)
