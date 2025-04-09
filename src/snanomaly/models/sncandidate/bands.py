from enum import Enum
from typing import Optional

import attrs
from attrs import define, field

from snanomaly.models.sncandidate.band import Band


class Bandset(Enum):
    BRI = ("B", "R", "I")
    gri = ("g", "r", "i")
    gri_primed = ("g_pr", "r_pr", "i_pr")


@define
class Bands:
    """
    Represents a collection of photometry bands.

    The list of bands is not exhaustive and can be extended as needed. Any band unlisted here will be omitted when
    loading an SNCandidate object.
    """

    B: Band = field(factory=Band)
    R: Band = field(factory=Band)
    I: Band = field(factory=Band)
    g: Band = field(factory=Band)
    r: Band = field(factory=Band)
    i: Band = field(factory=Band)
    g_pr: Band = field(factory=Band)
    r_pr: Band = field(factory=Band)
    i_pr: Band = field(factory=Band)

    _available_bandsets: set[Bandset] = field(factory=set, init=False)

    def get_bands(self, bandset: Optional[Bandset] = None) -> list[Band]:
        """
        Returns a list of all bands in the collection.
        """
        if bandset is not None:
            return [getattr(self, band_name) for band_name in bandset.value]
        return [getattr(self, band_name) for band_name in self.get_public_field_names()]

    def __attrs_post_init__(self):
        """
        Initialize the bands' names.
        """
        for band_name in self.get_public_field_names():
            band = getattr(self, band_name)
            band.name = band_name

    @property
    def nr_observations(self) -> int:
        """
        Returns the total number of observations across all bands.
        """
        return sum(band.nr_observations for band in self.get_bands())

    @classmethod
    def get_public_field_names(cls) -> list[str]:
        """
        Returns a list of all public field names in the Bands class.
        """
        return [field.name for field in attrs.fields(cls) if not field.name.startswith("_")]

    @property
    def available_bandsets(self) -> set[Bandset]:
        return self._available_bandsets

    def add_to_available_bandsets(self, bandset: Bandset):
        self._available_bandsets.add(bandset)
