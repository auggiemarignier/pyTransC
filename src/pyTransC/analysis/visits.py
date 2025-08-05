"""Module to get the number of visits in a state."""

from functools import partial
from typing import Annotated

import numpy as np

from ..utils.autocorr import autocorr_fardal
from ..utils.types import MultiStateMultiWalkerResult, MultiWalkerStateChain


def get_visits_to_states(  # calculate evolution of relative visits to each state along chain
    transc_sampler: MultiStateMultiWalkerResult,
    discard=0,
    thin=1,
    normalize=False,
):
    """
    Utility routine to retrieve proportion of visits to each state as a function of chain step, i.e. calculates the relative evidence/marginal Liklihoods of states.

    Collects information from previously run sampler. Can be used to diagnose performance and convergence.

    Inputs:
    discard - int               : number of output samples to discard (default = 0). (Also known as `burnin'.)
    thin - int                  : frequency of output samples in output chains to accept (default = 1, i.e. all)
    normalize - bool            : switch to calculate normalize relative evidence (True) or total visits to each state (False).
    flat - bool                 : switch to flatten walkers to a single chain (True) or calculate properties per chain (False).
                                    if false then information per chain can indicate whether some chains have not converged.
    walker_average - string     : indicates type of average pf visit statistics to calculate per chain step. Options are: 'median' (default) or 'mean'.
                                    'median' provides more diagnostics if a subset of chains have not converged and statistics are outliers.
    return_samples - bool       : switch to (optionally) return a record of visits to states for each step of each chain.

    Returns:
    visits - list int           : distribution of states visited as a function of chain step.
                                    either per chain (flat=False), or overall (flat=True).
                                    either normalized (normalize=True) or raw numbers (normalize=False).
                                    size equal to number of Markov chain steps retained (depends on discard and thin values).
    samples - list              : actual indices of states visited along each Markov chain (return_samples=True).
                                    used to view details of chain movement between states, largely for convergence checks.

    Attributes defined/updated:
    state_chain - ints                   : list of states visited along markov chains
    relative_marginal_likelihoods        : ratio of evidences/marginal Likelihods of each state
    state_changes_perwalker - array ints : number of times the walker changed state along the markov chain
    total_state_changes - int            : total number of state changes for all walkers

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
) -> np.ndarray:
    """
    Count the number of state changes in the state chain.

    Parameters:
    state_chain - list of lists of ints : the state chain to analyze (n_walkers, n_steps)
    discard - int                         : number of initial samples to discard (default = 0)
    thin - int                            : thinning factor for samples (default = 1)

    Returns:
    np.ndarray : array of counts of state changes for each walker
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


def get_autocorr_between_state_jumps(state_chain: MultiWalkerStateChain) -> float:
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
    visits_to_states: Annotated[
        np.ndarray[tuple[int, int], np.dtype[np.integer]], "n_walkers, n_states"
    ],
    walker_average: str = "mean",
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """
    Get the relative marginal likelihoods from the state chain.

    This is simply the expected number of visits to each state.

    Parameters:
    visits_to_states - list[float] : final count of visits of each walker to each state i.e. shape is (n_walkers, n_states)
    walker_average - str            : type of average over walkers ('mean' or 'median', default = 'mean')

    Returns:
    np.ndarray : array of relative marginal likelihoods
    """
    visits = walker_average_functions[walker_average](visits_to_states)
    return visits / np.sum(visits)
