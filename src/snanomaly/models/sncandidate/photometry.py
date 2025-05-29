from __future__ import annotations

from typing import Any

import cattrs
from attrs import define, field
from loguru import logger

from snanomaly.models.sncandidate.bands import Bands
from snanomaly.models.sncandidate.photometryobs import PhotometryObs


@define
class Photometry:
    raw_observations: list[PhotometryObs] = field(default=None)
    bin_width: float = field(default=None)
    bands: Bands = field(default=None)


@cattrs.register_structure_hook
def observation_hook(val: Any, _: Any) -> Photometry:
    if isinstance(val, dict):
        filtered_obs_list = []
        logger.info("Parsing Photometry")
        for item in val.get("raw_observations", []):
            try:
                obs = cattrs.structure(item, PhotometryObs)
                filtered_obs_list.append(obs)
            except (KeyError, ValueError, TypeError):
                logger.debug(f"Failed to structure item: {item}")
            except cattrs.errors.ClassValidationError as e:
                logger.debug(f"Failed to structure item: {item} with error: {e}")
        return Photometry(raw_observations=filtered_obs_list, bin_width=val.get("bin_width"), bands=val.get("bands"))
    raise ValueError(f"Cannot convert {val} to ObsUser")


# cattrs.structure({
#     "raw_observations": [{"source": "a", "time": 1, "magnitude": 1}],
# }, Photometry)
