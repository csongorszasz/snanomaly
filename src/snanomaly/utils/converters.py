from __future__ import annotations

import cattrs
import numpy as np
from loguru import logger

from snanomaly.models.sncandidate.band import Band
from snanomaly.models.sncandidate.bands import Bands
from snanomaly.models.sncandidate.photometry import Photometry
from snanomaly.models.sncandidate.photometryobs import PhotometryObs
from snanomaly.models.sncandidate.spectra import Spectra
from snanomaly.models.sncandidate.spectraobs import SpectraObs


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
    """

    @classmethod
    def structure(cls, data: list, _target_cls: type):
        if data is None:
            return None

        filtered_obs_list = []
        logger.debug("Parsing Photometry")
        for raw_obs in data:
            try:
                obs = cattrs.structure(raw_obs, PhotometryObs)
                filtered_obs_list.append(obs)
            except (KeyError, ValueError, TypeError):
                logger.debug(f"Failed to structure item: {raw_obs}")
            except cattrs.errors.ClassValidationError as e:
                logger.debug(f"Failed to structure item: {raw_obs} with error: {e}")

        bands = cls._bands_from_raw_observations(filtered_obs_list)
        return Photometry(
            raw_observations=filtered_obs_list,
            bands=bands,
        )

    @classmethod
    def _bands_from_raw_observations(cls, raw_observations: list[PhotometryObs]) -> Bands:
        bands = Bands()
        bands_lists: dict[str, dict[str, list]] = {}
        band_attribs = Band.get_public_field_names()
        band_names = Bands.get_public_field_names()

        for obs in raw_observations:
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


class SpectraConverter:
    """
    Handles serialization of spectra data.
    """

    @classmethod
    def structure(cls, data: list, _target_cls: type):
        if data is None:
            return None

        filtered_obs_list = []
        logger.debug("Parsing Spectra")
        for raw_obs in data:
            try:
                obs = cattrs.structure(raw_obs, SpectraObs)
                filtered_obs_list.append(obs)
            except (KeyError, ValueError, TypeError):
                logger.debug(f"Failed to structure item: {raw_obs}")
            except cattrs.errors.ClassValidationError as e:
                logger.debug(f"Failed to structure item: {raw_obs} with error: {e}")

        return Spectra(raw_observations=filtered_obs_list)
