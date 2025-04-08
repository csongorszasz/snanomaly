class DatasetError(Exception):
    pass

class InvalidDataPointSchemaError(DatasetError):
    pass

class InvalidDataPointError(DatasetError):
    pass
