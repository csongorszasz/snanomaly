class ResultError(Exception):
    pass

class UnexpectedDataFrameColumnError(ResultError):
    pass

class EmptyDataFrameError(ResultError):
    pass
