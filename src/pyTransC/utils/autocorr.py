"""Autocorrelation time estimation utilities.

This module implements autocorrelation time calculation routines following
Goodman & Weare (2010) and improved methods suggested in the emcee documentation.
These functions are used for assessing MCMC chain convergence and determining
appropriate thinning intervals.
"""

import numpy as np
import numpy.typing as npt

from .types import FloatArray


def autocorr_gw2010(y: FloatArray, c: float = 5.0) -> float:
    """Calculate autocorrelation time using Goodman & Weare (2010) method.

    Parameters
    ----------
    y : FloatArray
        MCMC chain data with shape (n_walkers, n_steps) or (n_steps,).
    c : float, optional
        Window size factor for automatic windowing. Default is 5.0.

    Returns
    -------
    float
        Autocorrelation time estimate.

    References
    ----------
    Goodman, J. & Weare, J. (2010). Ensemble samplers with affine invariance.
    Communications in Applied Mathematics and Computational Science, 5(1), 65-80.
    """
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_fardal(y: FloatArray, c: float = 5.0) -> float:
    """Calculate autocorrelation time using improved estimator.

    This function implements an improved autocorrelation time estimator
    that averages the autocorrelation function across all walkers before
    calculating the integrated time, as recommended in the emcee documentation.

    Parameters
    ----------
    y : FloatArray
        MCMC chain data.
    c : float, optional
        Window size factor for automatic windowing. Default is 5.0.

    Returns
    -------
    float
        Autocorrelation time estimate.

    References
    ----------
    https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
    """
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def next_pow_two(n: int) -> int:
    """Get the next power of two greater than or equal to n.

    Parameters
    ----------
    n : int
        Input number.

    Returns
    -------
    int
        Next power of two >= n.
    """
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x: npt.ArrayLike, norm: bool = True) -> FloatArray:
    """Calculate 1D autocorrelation function using FFT.

    Parameters
    ----------
    x : array_like
        1D input sequence.
    norm : bool, optional
        Whether to normalize by the zero-lag value. Default is True.

    Returns
    -------
    FloatArray
        Autocorrelation function values.

    Raises
    ------
    ValueError
        If input is not 1-dimensional.
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


def auto_window(taus: FloatArray, c: float) -> int:
    """Determine optimal windowing for autocorrelation time calculation.

    This function implements the automatic windowing procedure following
    Sokal (1989) to determine the appropriate cutoff for integrated
    autocorrelation time calculations.

    Parameters
    ----------
    taus : FloatArray
        Array of cumulative autocorrelation times.
    c : float
        Window size factor. The window is determined as the first index
        where c * tau[i] < i.

    Returns
    -------
    int
        Optimal window index for autocorrelation time calculation.

    References
    ----------
    Sokal, A. (1989). Monte Carlo methods in statistical mechanics: foundations
    and new algorithms. NATO Advanced Science Institutes Series E, 188, 131-192.
    """
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return int(np.argmin(m))
    return len(taus) - 1
