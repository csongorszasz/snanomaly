from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from pathlib import Path
from typing import Any

from attrs import define, field


@define
class Dataset(ABC):
    """
    Abstract base class for datasets (e.g.: OSC-2022).
    """

    path: Path = field()
    name: str = field()
    description: str = field()
    nr_datapoints: int = field()

    @abstractmethod
    def list_datapoints(self) -> None:
        pass

    @abstractmethod
    def load_dataset(self, batch_size: int) -> Generator[list]:
        pass

    @abstractmethod
    def load_datapoint(self, file: Path) -> Any:
        pass

    def __str__(self):
        return (f"{'#' * 30}\n"
                f"Dataset: {self.name}\n"
                f"Description: {self.description}\n"
                f"No. data points: {self.nr_datapoints}\n"
                f"Path: {self.path}\n"
                f"{'#' * 30}")
