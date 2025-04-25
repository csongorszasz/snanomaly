class RegressionError(Exception):
    pass

class BandNotFoundError(RegressionError):
    pass

class PredictionIntervalOutOfBoundsError(RegressionError):
    pass

class PeakTimeNotSetError(RegressionError):
    pass
