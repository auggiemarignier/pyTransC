"""Per-State MCMC Sampling."""

import multiprocessing
import random
from collections.abc import Callable
from functools import partial
from typing import Any

import emcee
import numpy as np

from ..utils.types import MultiStateDensity


def run_mcmc_per_state(
    n_states: int,
    n_dims: list[int],
    n_walkers: int | list[int],
    n_steps: int | list[int],
    pos: list[np.ndarray],
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
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Utility routine to run an MCMC sampler independently within each state.

    Creates a set of ensembles of posterior samples for each state.
    Makes use of emcee sampler for posterior sampling.

    This function is for convenience only. Its creates an ensemble of posterior samples within each state which
        - can serve as the input to run_ensemble_resampler()
        - can be used to build an approximate normalized pseudo_prior with build_auto_pseudo_prior().
    Alternatively, the user could supply their own ensembles in each state for these purposes,
    or directly provide their on own log_pseudo_prior function as required.

    Inputs:
    n_walkers - int, or list     : number of random walkers used by emcee sampler.
    n_steps - int                : number of steps required per walker.
    pos - n_walkers*n_dims*float  : list of starting points of markov chains in each state.
    log_posterior - func        : user supplied function to evaluate the log-posterior density for the ith state at location x.
                                    calling sequence log_posterior(x,i)
    discard - int, or list      : number of output samples to discard (default = 0). (Parameter passed to emcee, also known as `burnin'.)
    thin - int, or list         : frequency of output samples in output chains to accept (default = 1, i.e. all) (Parameter passed to emcee.)
    auto_thin - bool             : if True, ignores input thin value and instead thins the chain by the maximum auto_correlation time estimated (default = False).
    seed - int                  : random number seed
    parallel - bool             : switch to make use of multiprocessing package to parallelize over walkers
    n_processors - int           : number of processors to distribute work across (if parallel=True, else ignored).
                                    Default = multiprocessing.cpu_count() if parallel = True, else 1 if False.
    progress - bool             : switch to report progress of emcee to standard out. (Parameter passed to emcee.)
    kwargs - dict               : dictionary of optional control parameters passed to emcee package to determine sampling behaviour.

    Returns:
    log_posterior_ens - floats. : list of log-posterior densities of samples in each state, format [i][j],(i=1,...,n_states;j=1,..., n_walkers*n_samples).
    ensemble_per_state - floats : list of posterior samples in each state, format [i][j][k],(i=1,...,n_states;j=1,..., n_walkers*n_samples;k=1,...,ndim[i]).


    Attributes defined:
    n_walkers - int              : number of random walkers per state
    ensemble_per_state - floats : list of posterior samples in each state, format [i][j][k],(i=1,...,n_states;j=1,..., n_walkers*n_samples;k=1,...,ndim[i]).
    n_samples - n_states*int      : list of number of samples in each state.
    log_posterior_ens - floats. : list of log-posterior densities of samples in each state, format [i][j],(i=1,...,n_states;j=1,..., n_walkers*n_samples).
    run_per_state - bool.       : bool to keep track of whether run_mcmc_per_state has been called.

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

    samples: list[np.ndarray] = []
    log_posterior_ens: list[np.ndarray] = []
    auto_correlation: list[np.ndarray] = []
    for i in range(n_states):  # loop over states
        _log_posterior = partial(log_posterior, state=i)
        _samples, _log_posterior_ens, _auto_corr = process_state(
            log_posterior=_log_posterior,
            n_walkers=n_walkers[i],
            n_dims=n_dims[i],
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
    log_posterior: Callable[[np.ndarray], float],
    n_walkers: int,
    n_dims: int,
    pos: np.ndarray,
    n_steps: int,
    discard: int = 0,
    thin: int = 1,
    parallel: bool = False,
    n_processors: int = 1,
    skip_initial_state_check: bool = False,
    verbose: bool = True,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the posterior samples, log probabilities, and autocorrelation times for a single state."""
    _sampling_func = partial(
        _perform_sampling,
        n_walkers=n_walkers,
        n_dim=n_dims,
        log_prob_func=log_posterior,
        initial_state=pos,
        n_steps=n_steps,
        progress=verbose,
        skip_initial_state_check=skip_initial_state_check,
    )
    if parallel:
        n_processors = (
            multiprocessing.cpu_count() if n_processors == 1 else n_processors
        )
        with multiprocessing.Pool(processes=n_processors) as pool:
            kwargs["pool"] = pool
            sampler = _sampling_func(**kwargs)
    else:
        sampler = _sampling_func(**kwargs)

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


def _perform_sampling(
    n_walkers: int,
    n_dim: int,
    log_prob_func: Callable[[np.ndarray], float],
    initial_state: np.ndarray,
    n_steps: int,
    **kwargs,
) -> emcee.EnsembleSampler:
    """Perform MCMC sampling using emcee."""

    run_kwargs: dict[str, bool] = {}
    if "progress" in kwargs:
        run_kwargs["progress"] = kwargs.pop("progress")
    if "skip_initial_state_check" in kwargs:
        run_kwargs["skip_initial_state_check"] = kwargs.pop("skip_initial_state_check")

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob_func, **kwargs)
    sampler.run_mcmc(initial_state, n_steps, **run_kwargs)
    return sampler


def _perform_auto_thinning(
    samples: list[np.ndarray],
    log_posterior_ens: list[np.ndarray],
    auto_correlation: list[np.ndarray],
    verbose: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
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
