"""Per-State MCMC Sampling."""

import random
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from ..utils.types import FloatArray, MultiStateDensity
from ._emcee import perform_sampling_with_emcee


def run_mcmc_per_state(
    n_states: int,
    n_dims: list[int],
    n_walkers: int | list[int],
    n_steps: int | list[int],
    pos: list[FloatArray],
    log_posterior: MultiStateDensity,
    discard: int | list[int] = 0,
    thin: int | list[int] = 1,
    auto_thin: bool = False,
    seed: int = 61254557,
    parallel: bool | list[bool] = False,
    n_processors: int = 1,
    skip_initial_state_check: bool = False,
    verbose: bool = True,
    **kwargs,
) -> tuple[list[FloatArray], list[FloatArray]]:
    """Run independent MCMC sampling within each state.

    This utility function runs the emcee sampler independently within each state
    to generate posterior ensembles. These ensembles can then be used as input
    for ensemble resampling or for constructing pseudo-priors.

    Parameters
    ----------
    n_states : int
        Number of independent states in the problem.
    n_dims : list of int
        List of parameter dimensions for each state.
    n_walkers : int or list of int
        Number of random walkers for the emcee sampler. If int, same number
        is used for all states. If list, specifies walkers per state.
    n_steps : int or list of int
        Number of MCMC steps per walker. If int, same number is used for all
        states. If list, specifies steps per state.
    pos : list of FloatArray
        Starting positions for each state. Each array should have shape
        (n_walkers[state], n_dims[state]).
    log_posterior : MultiStateDensity
        Function to evaluate the log-posterior density at location x in state i.
        Must have signature log_posterior(x, state) -> float.
    discard : int or list of int, optional
        Number of samples to discard as burn-in. If int, same value used for
        all states. Default is 0.
    thin : int or list of int, optional
        Thinning factor for chains. If int, same value used for all states.
        Default is 1 (no thinning).
    auto_thin : bool, optional
        If True, automatically thin chains based on autocorrelation time,
        ignoring the `thin` parameter. Default is False.
    seed : int, optional
        Random number seed for reproducible results. Default is 61254557.
    parallel : bool or list of bool, optional
        Whether to use multiprocessing for parallel sampling. If bool, same
        setting used for all states. Default is False.
    n_processors : int, optional
        Number of processors to use if parallel=True. Default is 1.
    skip_initial_state_check : bool, optional
        Whether to skip emcee's initial state check. Default is False.
    verbose : bool, optional
        Whether to print progress information. Default is True.
    **kwargs
        Additional keyword arguments passed to emcee.EnsembleSampler.

    Returns
    -------
    -------
    ensemble_per_state : list of FloatArray
        Posterior samples for each state. Each array has shape
        (n_samples, n_dims[state]).
    log_posterior_ens : list of FloatArray
        Log posterior values for each ensemble. Each array has shape (n_samples,)

    Notes
    -----
    This function is primarily a convenience wrapper around emcee for generating
    posterior ensembles within each state independently. The resulting ensembles
    can be used with:

    - `run_ensemble_resampler()` for ensemble-based trans-dimensional sampling
    - Automatic pseudo-prior construction functions
    - Direct analysis of within-state posterior distributions

    If `auto_thin=True`, the function will automatically determine appropriate
    burn-in and thinning based on the autocorrelation time, following emcee
    best practices.

    Examples
    --------
    >>> ensembles, log_probs = run_mcmc_per_state(
    ...     n_states=2,
    ...     n_dims=[3, 2],
    ...     n_walkers=32,
    ...     n_steps=1000,
    ...     pos=[np.random.randn(32, 3), np.random.randn(32, 2)],
    ...     log_posterior=my_log_posterior,
    ...     auto_thin=True
    ... )
    """

    random.seed(seed)
    if not isinstance(n_walkers, list):
        n_walkers = [n_walkers] * n_states
    if not isinstance(discard, list):
        discard = [discard] * n_states
    if not isinstance(thin, list):
        thin = [thin] * n_states
    if not isinstance(n_steps, list):
        n_steps = [n_steps] * n_states
    if isinstance(parallel, bool):
        parallel = [parallel] * n_states

    if auto_thin:
        # ignore thinning factor because we are post thinning by the auto-correlation times
        # burn_in is also calculated from the auto-correlation time
        thin = [1] * n_states
        discard = [0] * n_states

    if verbose:
        print("\nRunning within-state sampler separately on each state")
        print("\nNumber of walkers               : ", n_walkers)
        print("\nNumber of states being sampled: ", n_states)
        print("Dimensions of each state: ", n_dims)

    samples: list[FloatArray] = []
    log_posterior_ens: list[FloatArray] = []
    auto_correlation: list[FloatArray] = []
    for i in range(n_states):  # loop over states
        _log_posterior = partial(log_posterior, state=i)
        _samples, _log_posterior_ens, _auto_corr = process_state(
            log_posterior=_log_posterior,
            n_walkers=n_walkers[i],
            pos=pos[i],
            n_steps=n_steps[i],
            discard=discard[i],
            thin=thin[i],
            parallel=parallel[i],
            n_processors=n_processors,
            skip_initial_state_check=skip_initial_state_check,
            verbose=verbose,
            **kwargs,
        )
        samples.append(_samples)
        log_posterior_ens.append(_log_posterior_ens)
        auto_correlation.append(_auto_corr)

    if auto_thin:
        samples, log_posterior_ens = _perform_auto_thinning(
            samples, log_posterior_ens, auto_correlation, verbose=verbose
        )

    return samples, log_posterior_ens


def process_state(
    log_posterior: Callable[[FloatArray], float],
    n_walkers: int,
    pos: FloatArray,
    n_steps: int,
    discard: int = 0,
    thin: int = 1,
    parallel: bool = False,
    n_processors: int = 1,
    verbose: bool = True,
    **kwargs: Any,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Get the posterior samples, log probabilities, and autocorrelation times for a single state."""
    sampler = perform_sampling_with_emcee(
        log_prob_func=log_posterior,
        n_walkers=n_walkers,
        n_steps=n_steps,
        initial_state=pos,
        parallel=parallel,
        n_processors=n_processors,
        progress=verbose,
        **kwargs,
    )
    samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    if samples is None:
        raise ValueError(
            "Sampler did not return a chain. Check the log_prob function and initial state."
        )

    log_posterior_ens = sampler.get_log_prob(discard=discard, thin=thin, flat=True)
    if log_posterior_ens is None:
        raise ValueError(
            "Sampler did not return log probabilities. Check the log_prob function and initial state."
        )

    autocorr_time = sampler.get_autocorr_time(tol=0)
    if autocorr_time is None:
        raise ValueError(
            "Sampler did not return autocorrelation times. Check the log_prob function and initial state."
        )

    return samples, log_posterior_ens, autocorr_time


def _perform_auto_thinning(
    samples: list[FloatArray],
    log_posterior_ens: list[FloatArray],
    auto_correlation: list[FloatArray],
    verbose: bool = False,
) -> tuple[list[FloatArray], list[FloatArray]]:
    """Perform auto thinning of samples based on autocorrelation times.

    Thin the chains using the maximum auto_correlation function for each state to get independent samples for fitting.
    emcee manual suggests tau = sampler.get_autocorr_time(); burn_in = int(2 * np.max(tau)); thin = int(0.5 * np.min(tau))
    """
    if verbose:
        print("Performing auto thinning of ensemble...")

    samples_auto = []
    log_posterior_ens_auto = []
    for i in range(len(samples)):
        thin = int(np.ceil(0.5 * auto_correlation[i].min()))
        burn_in = int(np.ceil(2.0 * auto_correlation[i].max()))
        samples_auto.append(samples[i][burn_in::thin])
        log_posterior_ens_auto.append(log_posterior_ens[i][burn_in::thin])

    return samples_auto, log_posterior_ens_auto
