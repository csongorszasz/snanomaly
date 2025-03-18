import numpy as np


class NumpyArrayConverter:
    """Handles serialization and deserialization of NumPy arrays."""

    @classmethod
    def structure(cls, data: list, _target_cls: type):
        if data is None:
            return None
        return np.array(data)

    @classmethod
    def unstructure(cls, arr: np.ndarray):
        if arr is None:
            return None
        return arr.tolist()
