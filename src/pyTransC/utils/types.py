"""Custom types for pytransc."""

from typing import Annotated, Protocol, TypeAlias

import numpy as np

# See https://medium.com/data-science-collective/do-more-with-numpy-array-type-hints-annotate-validate-shape-dtype-09f81c496746
# for guidance on numpy type annotations.
# These types are not actually supported by type checkers, so this is more for documentation purposes.
# Current numpy type annotations only specify the dtype, not the shape.
Int1DArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.integer]]
Int2DArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.integer]]
Float1DArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.floating]]
FloatNDArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.floating]]
MultiWalkerStateChain: TypeAlias = Annotated[Int2DArray, "(n_walkers, n_steps)"]
MultiWalkerModelChain: TypeAlias = Annotated[
    list[list[Float1DArray]],
    "(n_walkers, n_steps, n_dims[state_i])",
]
StateOrderedEnsemble: TypeAlias = list[FloatNDArray]


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


class MultiStateMultiWalkerResult(Protocol):
    """Protocol for the result of a multi-state multi-walker sampler."""

    @property
    def state_chain(self) -> MultiWalkerStateChain:
        """State chain for each walker.

        Expected shape is (n_walkers, n_steps).
        """
        ...

    @property
    def model_chain(self) -> MultiWalkerModelChain:
        """Model chain for each walker.

        Expected shape is (n_walkers, n_steps, n_dims).
        """
        ...

    @property
    def state_chain_tot(self) -> np.ndarray:
        """Running totals of states visited along the Markov chains."""
        ...

    @property
    def n_walkers(self) -> int:
        """Number of walkers in the sampler."""
        ...

    @property
    def n_steps(self) -> int:
        """Number of steps in the sampler."""
        ...

    @property
    def n_states(self) -> int:
        """Number of states in the sampler."""
        ...
