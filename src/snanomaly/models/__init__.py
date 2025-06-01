import cattrs
import numpy as np

from snanomaly.models.sncandidate.photometry import Photometry
from snanomaly.models.sncandidate.spectra import Spectra
from snanomaly.utils.converters import NumpyArrayConverter, PhotometryConverter, SpectraConverter

### numpy ndarray serialization/deserialization
cattrs.register_structure_hook(np.ndarray, NumpyArrayConverter.structure)
cattrs.register_unstructure_hook(np.ndarray, NumpyArrayConverter.unstructure)

### photometry serialization
cattrs.register_structure_hook(Photometry, PhotometryConverter.structure)

### spectra serialization
cattrs.register_structure_hook(Spectra, SpectraConverter.structure)
