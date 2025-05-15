from attrs import define, field

from snanomaly.models.results.result import Result
from snanomaly.models.sncandidate import Bandset


@define
class IsoForestResult(Result):
    sn_name: str = field()
    bandset: Bandset = field()
    anomaly_score: float = field()
