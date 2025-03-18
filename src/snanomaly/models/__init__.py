import cattrs
import numpy as np

from snanomaly.utils.converters import NumpyArrayConverter

### numpy ndarray serialization/deserialization
cattrs.register_structure_hook(np.ndarray, NumpyArrayConverter.structure)
cattrs.register_unstructure_hook(np.ndarray, NumpyArrayConverter.unstructure)
