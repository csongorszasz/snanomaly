from attrs import define, field

from snanomaly.models.sncandidate.alias import Alias
from snanomaly.models.sncandidate.floatdatum import FloatDatum
from snanomaly.models.sncandidate.photometry import Photometry
from snanomaly.models.sncandidate.source import Source
from snanomaly.models.sncandidate.spectraobs import SpectraObs
from snanomaly.models.sncandidate.stringdatum import StringDatum


@define(repr=False)
class SNCandidate:
    """
    Represents a supernova candidate.

    Follows the schema defined in the Open Astronomy Catalog Schema v1.0 but leaves out non-supernova-related fields:
        https://github.com/astrocatalogs/schema/blob/a9619d360ba330cc9d7534a9dcc4342e3b0c8085/README.md
    """

    schema: str = field()
    name: str = field()
    sources: list[Source] = field()
    alias: list[Alias] = field(factory=list)
    distinctfrom: list[StringDatum] = field(factory=list)
    error: list[StringDatum] = field(factory=list)
    ra: list[StringDatum] = field(factory=list)
    dec: list[StringDatum] = field(factory=list)
    discoverdate: list[StringDatum] = field(factory=list)
    maxdate: list[StringDatum] = field(factory=list)
    maxabsmag: list[FloatDatum] = field(factory=list)
    maxappmag: list[FloatDatum] = field(factory=list)
    maxband: list[StringDatum] = field(factory=list)
    maxvisualabsmag: list[FloatDatum] = field(factory=list)
    maxvisualappmag: list[FloatDatum] = field(factory=list)
    maxvisualband: list[StringDatum] = field(factory=list)
    maxvisualdate: list[StringDatum] = field(factory=list)
    redshift: list[FloatDatum] = field(factory=list)
    lumdist: list[FloatDatum] = field(factory=list)
    comovingdist: list[FloatDatum] = field(factory=list)
    velocity: list[FloatDatum] = field(factory=list)
    claimedtype: list[StringDatum] = field(factory=list)
    discoverer: list[StringDatum] = field(factory=list)
    ebv: list[FloatDatum] = field(factory=list)
    host: list[StringDatum] = field(factory=list)
    hostra: list[StringDatum] = field(factory=list)
    hostdec: list[StringDatum] = field(factory=list)
    hostoffsetang: list[FloatDatum] = field(factory=list)
    hostoffsetdist: list[FloatDatum] = field(factory=list)
    photometry: Photometry = field(factory=list, converter=None) # TODO
    spectra: list[SpectraObs] = field(factory=list)

    def __repr__(self):
        return f"SNCandidate(name={self.name}, alias={self.alias})"
