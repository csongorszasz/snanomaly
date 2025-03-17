import cattrs
import numpy as np


# Register a structure hook for numpy.ndarray
def structure_ndarray(data, _):
    if data is None:
        return None
    return np.array(data)

# Add this to your initialization code (e.g., in __init__.py or before loading data)
cattrs.register_structure_hook(np.ndarray, structure_ndarray)
