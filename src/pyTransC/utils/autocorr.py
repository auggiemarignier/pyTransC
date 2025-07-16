"""Following the suggestion from Goodman & Weare (2010) we implement routines for auto_correlation calculations."""

import numpy as np


def autocorr_gw2010(y, c=5.0):
    """Auto correlation utility routine following Goodman & Weare (2010)."""
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_fardal(y, c=5.0):
    """Auto correlation utility routine for improved auto correlation time estimate as per emcee notes.

    see https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
    """
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def next_pow_two(n):
    """Get the next power of two greater than or equal to n."""
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    """Auto correlation utility routine following Goodman & Weare (2010)."""
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


def auto_window(taus, c):
    """Auto correlation utility routine for Automated windowing procedure following Sokal (1989)."""
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1
