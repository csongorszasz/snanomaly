class InterpolationError(Exception):
    pass

class PeakTimeNotSetError(InterpolationError):
    pass

class BandNotFoundError(InterpolationError):
    pass
