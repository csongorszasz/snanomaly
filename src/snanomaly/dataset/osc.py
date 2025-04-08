from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import cattrs
from attrs import define
from cattrs import structure
from loguru import logger

from snanomaly.dataset.dataset import Dataset
from snanomaly.dataset.exception import DatasetError, InvalidDataPointError, InvalidDataPointSchemaError
from snanomaly.models.sncandidate.sncandidate import SNCandidate


@define
class OSC(Dataset):
    """
    An implementation of the OSC (Open Supernova Catalog) dataset.
    """

    def files(self) -> Generator[Path]:
        yield from self.path.glob("*.json")

    def list_datapoints(self) -> None:
        for file in self.files():
            print(file.name)  # noqa: T201

    def load_dataset(self, batch_size: int) -> Generator[list]:
        logger.debug(f"Loading OSC dataset at {self.path}")

        data = []
        for file in self.files():
            try:
                data.append(self.load_datapoint(file))
                if len(data) == batch_size:
                    yield data
                    data = []
            except DatasetError as ex:
                logger.warning(f"Failed to load `{file}`: {ex}")

        if data:  # yield any remaining data
            yield data

    def load_datapoint(self, file: Path) -> SNCandidate:
        logger.debug(f"Loading data point at: {file}")
        with file.open() as f:
            event_name = file.stem
            try:
                datapoint = json.load(f).get(event_name)
                if datapoint is None:
                    raise InvalidDataPointSchemaError
                return structure(datapoint, SNCandidate)
            except (json.JSONDecodeError, ValueError, TypeError, cattrs.errors.ExceptionGroup) as ex:
                logger.exception(ex)
                raise InvalidDataPointError
