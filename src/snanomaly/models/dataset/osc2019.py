from attrs import define

from snanomaly import DATASETS_DIR
from snanomaly.models.dataset.osc import OSC


@define
class OSC2019(OSC):
    """
    Represents the OSC-2019 dataset.
    """

    def __attrs_post_init__(self):
        self.path = DATASETS_DIR / "osc2019" if self.path is None else self.path
        self.name = "OSC-2019"
        self.description = "The `Open Supernova Catalog` with data up to 2019."
        super().__attrs_post_init__()  # data loading happens in the parent class
