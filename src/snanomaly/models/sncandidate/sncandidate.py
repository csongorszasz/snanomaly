from attrs import define, field

from snanomaly.models.sncandidate.date import Date
from snanomaly.models.sncandidate.dec import Dec
from snanomaly.models.sncandidate.floatdatum import FloatDatum
from snanomaly.models.sncandidate.ra import RA
from snanomaly.models.sncandidate.source import Source
from snanomaly.models.sncandidate.stringdatum import StringDatum


@define
class SNCandidate:
    """
    Represents a supernova candidate.

    Follows the schema defined in the Open Astronomy Catalog Schema v1.0:
        https://github.com/astrocatalogs/schema/blob/a9619d360ba330cc9d7534a9dcc4342e3b0c8085/README.md
    """

    schema: str = field()
    name: str = field()
    sources: list[Source] = field()
    alias: list[StringDatum] = field(factory=list)
    distinctfrom: list[StringDatum] = field(factory=list)
    error: list[StringDatum] = field(factory=list)
    ra: list[RA] = field(factory=list)
    dec: list[Dec] = field(factory=list)
    discoverdate: list[Date] = field(factory=list)
    maxdate: list[Date] = field(factory=list)
    maxabsmag: list[FloatDatum] = field(factory=list)
    maxappmag: list[FloatDatum] = field(factory=list)
    maxband: list[StringDatum] = field(factory=list)
    maxvisualabsmag: list[FloatDatum] = field(factory=list)
    maxvisualappmag: list[FloatDatum] = field(factory=list)
    maxvisualband: list[StringDatum] = field(factory=list)
    maxvisualdate: list[Date] = field(factory=list)
    redshift: list[FloatDatum] = field(factory=list)
    lumdist: list[FloatDatum] = field(factory=list)
    comovingdist: list[FloatDatum] = field(factory=list)
    velocity: list[FloatDatum] = field(factory=list)
    claimedtype: list[StringDatum] = field(factory=list)
    discoverer: list[StringDatum] = field(factory=list)
    ebv: list[FloatDatum] = field(factory=list)
    host: list[StringDatum] = field(factory=list)
    hostra: list[RA] = field(factory=list)
    hostdec: list[Dec] = field(factory=list)
    hostoffsetang: list[FloatDatum] = field(factory=list)
    hostoffsetdist: list[FloatDatum] = field(factory=list)
    # photometry
    # spectra
