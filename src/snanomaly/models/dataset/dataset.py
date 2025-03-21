from abc import ABC, abstractmethod
from pathlib import Path

from attrs import define, field


@define()
class Dataset(ABC):
    """
    Abstract base class for datasets (e.g.: OSC-2022).
    """

    path: Path = field(default=None)
    name: str = field(default=None, init=False)
    description: str = field(default=None, init=False)
    size: int = field(default=None, init=False)
    objects: list = field(factory=list, init=False)

    @abstractmethod
    def _load_data(self):
        pass

    def __attrs_post_init__(self):
        self._load_data()

    def __str__(self):
        return (f"{'#' * 30}\n"
                f"Dataset: {self.name}\n"
                f"Description: {self.description}\n"
                f"Size: {self.size} objects\n"
                f"Path: {self.path}\n"
                f"{'#' * 30}")
