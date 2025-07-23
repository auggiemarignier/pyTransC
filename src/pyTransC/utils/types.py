"""Custom types for pytransc."""

from typing import Protocol

import numpy as np


class MultiStateDensity(Protocol):
    """Protocol for multi-state density function.  Not necessarily normalised."""

    def __call__(self, x: np.ndarray, state: int) -> float:
        """
        Evaluate the density at point x in state.

        Parameters
        ----------
        x : array-like
            Input point(s) where the density is evaluated.
        state : int
            The state index for which the density is evaluated.

        Returns
        -------
        log_density : float
            Log density value at x.
        """
        ...


class SampleableMultiStateDensity(MultiStateDensity, Protocol):
    """Protocol for multi-state density function that can sample from the distribution.

    This is useful mainly for the state jump sampler, where we need to draw deviates from the pseudo-prior distribution.
    """

    def draw_deviate(self, state: int) -> np.ndarray:
        """
        Draw a random deviate from the distribution for a given state.

        Parameters
        ----------
        state : int
            The state index for which the deviate is drawn.

        Returns
        -------
        random_deviate : np.ndarray
            A random deviate sampled from the pseudo-prior distribution.
        """
        ...


class ProposableMultiStateDensity(MultiStateDensity, Protocol):
    """Protocol for multi-state density function that can propose model in a different state.

    This is useful for the state jump sampler.
    """

    def propose(self, x: np.ndarray, state: int) -> np.ndarray:
        """
        Propose a new model based on the input point x.

        Parameters
        ----------
        x : array-like
            Input point(s) based on which the new model is proposed.

        state : int
            The state index for which the new model is proposed.

        Returns
        -------
        new_model : np.ndarray
            The proposed new model index.
        """
        ...
