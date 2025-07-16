"""Custom exceptions for pyTransC."""


class PyTransCError(Exception):
    """Base class for other exceptions."""

    pass


class InputError(PyTransCError):
    """Raised when necessary inputs are missing."""

    def __init__(self, msg=""):
        super().__init__(msg)
