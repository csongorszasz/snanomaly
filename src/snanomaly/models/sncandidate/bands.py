from attrs import define, field

from snanomaly.models.sncandidate.band import Band


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
