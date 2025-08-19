"""Custom exceptions for pyTransC.

This module defines the exception hierarchy for the pyTransC package,
providing specific error types for different failure modes.
"""


class PyTransCError(Exception):
    """Base exception class for all pyTransC-specific errors.

    This is the root exception class from which all other pyTransC
    exceptions inherit. It can be used to catch any pyTransC-related
    error in a general exception handler.
    """

    pass


class InputError(PyTransCError):
    """Raised when required inputs are missing or invalid.

    This exception is raised when:
    - Required function arguments are not provided
    - Input arrays have incompatible shapes
    - Parameter values are outside acceptable ranges
    - Input data types are incorrect

    Parameters
    ----------
    msg : str, optional
        Human-readable error message describing the input problem.
    """

    def __init__(self, msg="Invalid or missing input parameters"):
        super().__init__(msg)
