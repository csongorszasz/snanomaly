from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Optional

import cattrs
from attrs import define
from cattrs import structure, transform_error
from loguru import logger

from snanomaly.dataset.dataset import Dataset
from snanomaly.dataset.exception import (
    DataPointNotFoundError,
    DatasetError,
    InvalidDataPointSchemaError,
)
from snanomaly.models.sncandidate.sncandidate import SNCandidate


@define
class OSC(Dataset):
    """
    An implementation of the OSC (Open Supernova Catalog) dataset.
    """

    DATA_EXTENSION = ".json"

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
                logger.warning(f"Skipped invalid file `{file}`: {ex}")

        if data:  # yield any remaining data
            yield data

    def load_datapoint(self, path: Optional[Path] = None, name: Optional[str] = None) -> SNCandidate:
        if not path and not name:
            raise DatasetError("No `path` or `name` provided.")
        if not path and name:
            path = self.path / f"{name}{self.DATA_EXTENSION}"
        if not path.exists():
            raise DataPointNotFoundError(f"Path: {path}")

        logger.debug(f"Loading data point at: {path}")
        with path.open() as f:
            event_name = path.stem
            try:
                datapoint = json.load(f).get(event_name)
                if datapoint is None:
                    raise InvalidDataPointSchemaError
                return structure(datapoint, SNCandidate)
            except cattrs.errors.ExceptionGroup as ex:
                err_msg = transform_error(ex)
                logger.exception(err_msg)
                raise InvalidDataPointSchemaError(err_msg)
            except (json.JSONDecodeError, ValueError, TypeError) as ex:
                logger.exception(ex)
                raise InvalidDataPointSchemaError
