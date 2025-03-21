import json

import cattrs.errors
from attrs import define
from cattrs import structure
from tqdm import tqdm

from snanomaly import logger
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

        cnt_failed = 0
        cnt_success = 0
        for i, file in tqdm(enumerate(files), total=self.size):
            logger.trace(f"Loading file: {file.name}")
            with file.open() as f:
                try:
                    data = json.load(f).get(file.stem)
                    if data is None:
                        logger.warning(f"Warning: Invalid format for `{file.name}`. Skipping.")
                        continue
                    sn_candidate = structure(data, SNCandidate)
                    self.objects.append(sn_candidate)
                    cnt_success += 1
                except (json.JSONDecodeError, ValueError, TypeError, cattrs.errors.ExceptionGroup) as ex:
                    cnt_failed += 1
                    logger.warning(f"Error parsing file nr. {i+1}: `{file.name}`.")
                    logger.exception(ex)

        logger.info(f"{cnt_failed} object(s) were omitted. Successfully loaded {cnt_success}/{self.size}.")

