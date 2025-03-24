from __future__ import annotations

from typing import Optional

from attrs import define, field


@define
class ValidationResult:
    is_valid: bool = field(default=True)
    message: Optional[str] = field(default=None)
    check_name: str = field(default="")

    @classmethod
    def valid(cls, check_name: str) -> ValidationResult:
        return cls(True, None, check_name)

    @classmethod
    def invalid(cls, message: str, check_name: str) -> ValidationResult:
        return cls(False, message, check_name)
