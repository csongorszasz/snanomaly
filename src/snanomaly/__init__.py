import os
import sys
from pathlib import Path

from attrs import define, field
from loguru import logger

### Predefined paths ###

@define
class Directories:
    """Project directory structure as an enum-like interface."""

    PROJECT: Path = field(default=Path(__file__).resolve().parent.parent.parent)

    @property
    def DATASETS(self) -> Path:
        return self.PROJECT / "datasets"

    @property
    def LOGS(self) -> Path:
        return self.PROJECT / "logs"

    def create_dirs(self) -> None:
        """Create all subdirectories."""
        # Get all property methods that represent directories (except PROJECT)
        for name in dir(self):
            if name.isupper() and name != "PROJECT":
                path = getattr(self, name)
                if isinstance(path, Path):
                    path.mkdir(parents=True, exist_ok=True)


# Instantiate singleton
dirs = Directories()
dirs.create_dirs()

### Configure logger ###

# Remove default handler
logger.remove()

# Add handler for logging errors (including exceptions) to a file
logger.add(
    dirs.LOGS / "errors.log",
    format="{time} | {level} | {module}:{function}:{line} - {message}",
    level="ERROR",
    rotation="1 MB",
    retention="7 days",
    backtrace=True,
    diagnose=True,
    serialize=True,
)

# Add custom handler for stdout that excludes exception logs
logger.add(
    sys.stdout,
    format="{time} | {level} | {module}:{function}:{line} - {message}",
    level=os.environ.get("LOG_LEVEL", "INFO"),
    filter=lambda record: record["exception"] is None,
)

__all__ = [
    "dirs",
    "logger",
]
