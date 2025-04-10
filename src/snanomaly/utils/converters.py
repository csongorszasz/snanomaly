from __future__ import annotations

import numpy as np
from cattrs import structure
from loguru import logger

from snanomaly.models.sncandidate.band import Band
from snanomaly.models.sncandidate.bands import Bands
from snanomaly.models.sncandidate.photometry import Photometry
from snanomaly.models.sncandidate.photometryobs import PhotometryObs


class NumpyArrayConverter:
    """Handles serialization and deserialization of NumPy arrays."""

    @classmethod
    def structure(cls, data: list, _target_cls: type):
        if data is None:
            return None
        return np.array(data)

    @classmethod
    def unstructure(cls, arr: np.ndarray):
        if arr is None:
            return None
        return arr.tolist()

class PhotometryConverter:
    """
    Handles serialization of photometry data.

    Builds additional numpy arrays for each band with binning.
    """

    @classmethod
    def structure(cls, data: Photometry, _target_cls: type):
        if data is None:
            return None
        raw_observations = structure(data, list[PhotometryObs])
        bin_width = 3
        bands = cls._bands_from_raw_observations(raw_observations)
        return Photometry(
            raw_observations=raw_observations,
            bin_width=bin_width,
            bands=bands,
        )

    @classmethod
    def _bands_from_raw_observations(cls, raw_observations: list[PhotometryObs]) -> Bands:
        bands = Bands()
        bands_lists: dict[str, dict[str, list]] = {}
        band_attribs = Band.get_public_field_names()
        band_names = Bands.get_public_field_names()

        for obs in raw_observations:
            if cls.is_valid(obs):
                band_name = obs.band.replace("'", "_pr")
                if band_name not in band_names:
                    logger.debug(f"Skipping photometric observation: Unsupported band: {band_name}")
                    continue

                if band_name not in bands_lists:
                    bands_lists[band_name] = {}
                    for attr in band_attribs:
                        bands_lists[band_name][attr] = []

                for attr in band_attribs:
                    bands_lists[band_name][attr].append(getattr(obs, attr))

        for band_name, band_data in bands_lists.items():
            band = getattr(bands, band_name)
            for attr in band_attribs:
                setattr(band, attr, np.array(band_data[attr]))

        return bands

    @classmethod
    def is_valid(cls, obs: PhotometryObs) -> bool:
        """TODO: Implement a more exhaustive check."""
        # Check if the observation is valid
        return obs.time is not None and obs.flux is not None and obs.band is not None
