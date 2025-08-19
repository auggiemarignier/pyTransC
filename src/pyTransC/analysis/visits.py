"""Module to get the number of visits in a state."""

from functools import partial
from typing import Annotated

import numpy as np

from ..utils.autocorr import autocorr_fardal
from ..utils.types import (
    FloatArray,
    IntArray,
    MultiStateMultiWalkerResult,
    MultiWalkerStateChain,
)


def get_visits_to_states(  # calculate evolution of relative visits to each state along chain
    transc_sampler: MultiStateMultiWalkerResult,
    discard=0,
    thin=1,
    normalize=False,
):
    """Calculate state visit statistics from trans-dimensional MCMC chains.

    This function analyzes the state visitation patterns in trans-dimensional MCMC
    chains to estimate relative marginal likelihoods and assess convergence.

    Parameters
    ----------
    transc_sampler : MultiStateMultiWalkerResult
        Results from a trans-dimensional MCMC sampler containing state chains
        and cumulative visit counts.
    discard : int, optional
        Number of initial samples to discard as burn-in. Default is 0.
    thin : int, optional
        Thinning factor - use every `thin`-th sample. Default is 1 (no thinning).
    normalize : bool, optional
        Whether to normalize visit counts to get proportions. If False, returns
        raw visit counts. Default is False.

    Returns
    -------
    IntArray
        State visit statistics as a function of chain step. Shape depends on
        the input sampler structure and normalize parameter:
        - If normalize=True: proportions of visits to each state
        - If normalize=False: cumulative visit counts to each state

    Notes
    -----
    The relative visit proportions provide estimates of the relative marginal
    likelihoods (Bayes factors) between states in the trans-dimensional model.
    These estimates become more accurate as the chain length increases and the
    sampler converges.

    For convergence assessment, monitor whether the visit proportions stabilize
    across different walkers and chain segments.

    Examples
    --------
    >>> visits = get_visits_to_states(
    ...     transc_sampler=results,
    ...     discard=1000,
    ...     thin=10,
    ...     normalize=True
    ... )
    >>> # visits now contains proportional visits to each state
    """

    samples = transc_sampler.state_chain[:, discard::thin]
    visits = transc_sampler.state_chain_tot[:, discard::thin, :].astype("float")
    if normalize:
        visits /= np.sum(visits, axis=2)[:, :, np.newaxis]

    return (
        visits,  # fraction of visits to each state along chain for all walkers (n_walkers, n_steps, n_states)
        samples,  # actual indices of states visited along each Markov chain (n_walkers, n_steps)
    )


def count_state_changes(
    state_chain: MultiWalkerStateChain,
    discard: int = 0,
    thin: int = 1,
) -> IntArray:
    """
    Count the number of state changes in the state chain.

    Parameters:
    state_chain - list of lists of ints : the state chain to analyze (n_walkers, n_steps)
    discard - int                         : number of initial samples to discard (default = 0)
    thin - int                            : thinning factor for samples (default = 1)

    Returns:
    -------
    IntArray : array of counts of state changes for each walker
    """
    _state_chain = state_chain[:, discard::thin]
    n_walkers = _state_chain.shape[0]
    changes = np.zeros(n_walkers, dtype=int)
    for i in range(n_walkers):
        changes[i] = np.count_nonzero(_state_chain[i, 1:] - _state_chain[i, :-1])
    return changes


def count_total_state_changes(
    state_chain: MultiWalkerStateChain,
    discard: int = 0,
    thin: int = 1,
) -> int:
    """
    Count the total number of state changes across all walkers.

    Parameters:
    state_chain - list of lists of ints : the state chain to analyze (n_walkers, n_steps)
    discard - int                         : number of initial samples to discard (default = 0)
    thin - int                            : thinning factor for samples (default = 1)

    Returns:
    int : total number of state changes
    """
    return int(np.sum(count_state_changes(state_chain, discard, thin)))


def get_acceptance_rate_between_states(
    transc_sampler: MultiStateMultiWalkerResult,
    discard: int = 0,
    thin: int = 1,
) -> float:
    """
    Calculate the acceptance rate between states.

    Parameters:
    transc_sampler - MultiStateMultiWalkerResult : sampler object containing state chain information
    discard - int                                 : number of initial samples to discard (default = 0)
    thin - int                                    : thinning factor for samples (default = 1)

    Returns:
    float : acceptance rate between states as a percentage
    """
    total_state_changes = count_total_state_changes(
        transc_sampler.state_chain, discard, thin
    )
    return (
        100
        * total_state_changes
        * thin
        / (transc_sampler.n_walkers * transc_sampler.n_steps)
    )


def get_autocorr_between_state_jumps(state_chain: IntArray) -> float:
    """
    Calculate the autocorrelation time for between state jumps.

    Parameters:
    state_chain - list of lists of ints : the state chain to analyze (n_walkers, n_steps)

    Returns:
    float : autocorrelation time for between state jumps
    """
    return autocorr_fardal(state_chain)


walker_average_functions = {
    "mean": partial(np.mean, axis=0),
    "median": partial(np.median, axis=0),
}


def get_relative_marginal_likelihoods(
    visits_to_states: Annotated[IntArray, "n_walkers, n_states"],
    walker_average: str = "mean",
) -> FloatArray:
    """Calculate relative marginal likelihoods from state visit counts.

    This function estimates the relative marginal likelihoods
    between different states based on the number of times each state was
    visited during trans-dimensional MCMC sampling.

    Parameters
    ----------
    visits_to_states : IntArray
        Final visit counts for each walker to each state. Shape (n_walkers, n_states)
        where element [i, j] is the total number of times walker i visited state j.
    walker_average : str, optional
        Type of average to compute across walkers. Options are:
        - 'mean': arithmetic mean (default)
        - 'median': median (more robust to outliers)

    Returns
    -------
    FloatArray
        Relative marginal likelihoods for each state. Shape (n_states,).

    Notes
    -----
    The relative marginal likelihood for state i is proportional to the expected
    number of visits to that state. Under certain conditions (detailed balance,
    proper pseudo-priors), this provides an unbiased estimator of the ratio of
    marginal likelihoods.

    For reliable estimates:
    - Ensure sufficient sampling (many chain steps)
    - Check convergence across different walkers
    - Verify that all states have been adequately explored

    Examples
    --------
    >>> # visits_to_states has shape (32, 3) for 32 walkers, 3 states
    >>> rel_ml = get_relative_marginal_likelihoods(visits_to_states)
    >>> print(f"Relative marginal likelihoods: {rel_ml}")
    """
    visits = walker_average_functions[walker_average](visits_to_states)
    return visits / np.sum(visits)
