"""Module to get the number of visits in a state."""

import emcee
import numpy as np

from ..utils.autocorr import autocorr_fardal


def get_visits_to_states(  # calculate evolution of relative visits to each state along chain
    alg: str,
    n_states: int,
    n_walkers: int,
    n_steps: int,
    state_chain_tot: np.ndarray | None = None,  # total state chain for all walkers
    state_chain: np.ndarray | None = None,  # state chain for each walker
    discard=0,
    thin=1,
    normalize=False,
    flat=False,
    walker_average="median",
    product_space_sampler: emcee.EnsembleSampler | None = None,
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
    if alg == "TransC-integration":
        raise ValueError(
            "In function get_visits_to_states: alg='TransC-integration' is not supported."
        )
    elif alg == "TransC-product-space":
        # calculate fraction of visits to each state along chain averaged over walkers
        if product_space_sampler is None:
            raise ValueError(
                "In function get_visits_to_states: product_space_sampler must be provided for alg='TransC-product-space'."
            )
        (
            out,
            samples,
            relative_marginal_likelihoods,
            acceptance_rate_per_walker,
            acceptance_rate,
            mean_autocorr_time,
            max_autocorr_time,
            autocorr_time_for_between_state_jumps,
        ) = _get_visits_to_states_product_space(
            product_space_sampler,
            n_states,
            discard=discard,
            thin=thin,
            normalize=normalize,
            flat=flat,
            walker_average=walker_average,
        )
        _additional_outs = (
            acceptance_rate_per_walker,
            acceptance_rate,
            mean_autocorr_time,
            max_autocorr_time,
        )
    else:
        (
            out,
            samples,
            relative_marginal_likelihoods,
            autocorr_time_for_between_state_jumps,
        ) = _get_visits_to_states_default(
            state_chain_tot,
            state_chain,
            discard=discard,
            thin=thin,
            normalize=normalize,
            flat=flat,
            walker_average=walker_average,
        )
        _additional_outs = ()

    changes = np.zeros(n_walkers, dtype=int)
    for i in range(n_walkers):
        changes[i] = np.count_nonzero(samples.T[i][1:] - samples.T[i][:-1])

    state_changes_per_walker = changes
    total_state_changes = np.sum(changes)
    acceptance_rate_between_states = (
        100 * total_state_changes * thin / (n_walkers * n_steps)
    )

    return (
        out,  # fraction of visits to each state along chain for all walkers
        samples,  # actual indices of states visited along each Markov chain
        relative_marginal_likelihoods,  # ratio of evidences/marginal Likelihoods of each state
        state_changes_per_walker,  # number of times the walker changed state along the markov chain
        total_state_changes,  # total number of state changes for all walkers
        acceptance_rate_between_states,  # acceptance rate between states
        autocorr_time_for_between_state_jumps,  # autocorrelation time for between state jumps
        *(_additional_outs),  # additional outputs if available
    )


def _get_visits_to_states_product_space(
    product_space_sampler: emcee.EnsembleSampler,
    n_states,
    discard=0,
    thin=1,
    normalize=False,
    flat=False,
    walker_average="median",
):
    samples = product_space_sampler.get_chain(
        discard=discard, thin=thin
    )  # collect model ensemble
    state_chain = np.rint(samples[:, :, 0]).astype("int")
    visits = np.zeros((np.shape(samples)[0], np.shape(samples)[1], n_states))
    for i in range(n_states):
        visits[:, :, i] = np.cumsum(state_chain == i, axis=0)
    if normalize:
        for _ in range(n_states):
            visits /= np.sum(visits, axis=2)[:, :, np.newaxis]
    out = visits
    if flat and walker_average == "mean":
        out = np.mean(
            visits, axis=1
        )  # fraction of visits to each state along chain averaged over walkers
    if flat and walker_average == "median":
        out = np.median(
            visits, axis=1
        )  # fraction of visits to each state along chain averaged over walkers
    if flat:
        relative_marginal_likelihoods = out[-1]
    else:
        relative_marginal_likelihoods = np.mean(out[-1], axis=0)
    samples = state_chain

    acceptance_rate_per_walker = product_space_sampler.acceptance_fraction
    acceptance_rate = 100 * np.mean(acceptance_rate_per_walker)

    mean_autocorr_time = np.mean(
        product_space_sampler.get_autocorr_time(tol=0)
    )  # mean autocorrelation time in steps for all parameters
    max_autocorr_time = np.max(
        product_space_sampler.get_autocorr_time(tol=0)
    )  # max autocorrelation time in steps for all parameters
    autocorr_time_for_between_state_jumps = product_space_sampler.get_autocorr_time(
        tol=0
    )[0]

    return (
        out,
        samples,
        relative_marginal_likelihoods,
        acceptance_rate_per_walker,
        acceptance_rate,
        mean_autocorr_time,
        max_autocorr_time,
        autocorr_time_for_between_state_jumps,
    )


def _get_visits_to_states_default(
    state_chain_tot,
    state_chain,
    discard=0,
    thin=1,
    normalize=False,
    flat=False,
    walker_average="median",
):
    visits = state_chain_tot[discard::thin, :, :].astype("float")
    if normalize:
        visits /= np.sum(visits, axis=2)[:, :, np.newaxis]
    out = visits
    samples = state_chain[discard::thin, :]
    if flat and walker_average == "mean":
        out = np.mean(
            visits, axis=1
        )  # fraction of visits to each state along chain averaged over walkers
    if flat and walker_average == "median":
        out = np.median(
            visits, axis=1
        )  # fraction of visits to each state along chain averaged over walkers
    if flat:
        rml = out[-1]
    else:
        rml = np.mean(out[-1], axis=0)
    if normalize:
        rml /= np.sum(rml)
    relative_marginal_likelihoods = rml
    autocorr_time_for_between_state_jumps = autocorr_fardal(state_chain.T)

    return (
        out,
        samples,
        relative_marginal_likelihoods,
        autocorr_time_for_between_state_jumps,
    )
