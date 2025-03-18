from attrs import define

from snanomaly import DATASETS_DIR
from snanomaly.models.dataset.osc import OSC


@define
class OSC2019Mini(OSC):
    """
    Represents a mini version of the OSC-2019 dataset with fewer than 100 selected objects.
    """

    def __attrs_post_init__(self):
        self.path = DATASETS_DIR / "osc2019_mini" if self.path is None else self.path
        self.name = "OSC-2019 Mini"
        self.description = (
            "The `Open Supernova Catalog` with data up to 2019. Contains a small subset of the "
            "OSC-2019 dataset (<100 objects). Objects were selected from various catalogues to "
            "ensure a diversity."
        )
        super().__attrs_post_init__()  # data loading happens in the parent class
