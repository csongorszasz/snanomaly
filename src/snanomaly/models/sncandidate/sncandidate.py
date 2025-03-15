from attrs import define, field

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

    schema: str
    name: str
    sources: list[Source]
    alias: list[StringDatum] = field(factory=list)
    distinctfrom: list[StringDatum] = field(factory=list)
    error: list[StringDatum] = field(factory=list)  # TODO: Find an example for this
    ra: RA = field(factory=RA)
    # dec
    # discoverdate
    # maxdate
    # maxabsmag
    # maxappmag
    # maxband
    # maxvisualabsmag
    # maxvisualappmag
    # maxvisualband
    # maxvisualdate
    # redshift
    # lumdist
    # comovingdist
    # velocity
    # claimedtype
    # discoverer
    # ebv
    # host
    # hostra
    # hostdec
    # hostoffsetang
    # hostoffsetdist
    # photometry
    # spectra
    #
    #
    #
