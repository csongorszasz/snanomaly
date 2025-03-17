import json

from attrs import define
from cattrs import structure
from loguru import logger

from snanomaly.models.dataset.dataset import Dataset
from snanomaly.models.sncandidate.sncandidate import SNCandidate


@define
class OSC(Dataset):
    """
    Represents the OSC-2019 dataset.
    """

    def _load_data(self):
        logger.info("Loading OSC dataset...")
        files = self.path.glob("*.json")

        for i, file in enumerate(files):
            logger.trace(f"Loading file: {file.name}")
            with file.open() as f:
                try:
                    data = json.load(f).get(file.stem)
                    if data is None:
                        logger.warning(f"Warning: Invalid format for `{file.name}`. Skipping.")
                        continue
                    sn_candidate = structure(data, SNCandidate)
                    self.objects.append(sn_candidate)
                except Exception as ex:
                    logger.error(f"{i+1}. Error parsing file: `{file.name}`.")
                    logger.debug(f"Error: {ex}")
                    continue

        logger.info(f"Loaded {len(self.objects)} objects.")

