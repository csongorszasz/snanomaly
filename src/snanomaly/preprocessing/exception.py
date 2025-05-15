class PreprocessingError(Exception):
    pass

class ColumnNotFoundError(PreprocessingError):
    pass

class InvalidTransformationTableError(PreprocessingError):
    pass
