"""Common functions for samplers."""

import numpy as np

from ..utils.types import IntArray


def count_visits_to_states(
    state_chain: IntArray,
    n_states: int,
) -> IntArray:
    """
    Count the running total number of visits to each state in the state chain for a single walker.

    Parameters
    ----------
    state_chain : IntArray
        The state chain for a single walker (shape: [n_steps]).

    n_states : int
        The number of states in the sampler.

    Returns
    -------
    counts : IntArray
        An array of shape (n_steps, n_states) containing the counts of visits to each state at each step.
    """
    n_steps = state_chain.shape[0]
    counts = np.zeros((n_steps, n_states), dtype=int)
    for state in range(n_states):
        counts[:, state] = (state_chain == state).astype(int).cumsum()
    return counts
