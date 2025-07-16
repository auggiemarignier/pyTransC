"""Per-State MCMC Sampling."""

import multiprocessing
import random

import emcee
import numpy as np


def run_mcmc_per_state(
    n_states: int,
    n_dims: list[int],
    n_walkers,
    n_steps,
    pos,
    log_posterior,
    log_posterior_args=[],
    discard=0,
    thin=1,
    auto_thin=False,
    seed=61254557,
    parallel=False,
    n_processors=1,
    progress=True,
    skip_initial_state_check=False,
    io=False,
    verbose=False,
    **kwargs,
):
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
                                    calling sequence log_posterior(x,i,*log_posterior_args)
    log_posterior_args - list   : user defined list of additional arguments passed to log_posterior function (optional).
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
        n_walkers = [n_walkers for i in range(n_states)]
    if not isinstance(discard, list):
        discard = [discard for i in range(n_states)]
    if not isinstance(thin, list):
        thin = [thin for i in range(n_states)]
    if not isinstance(n_steps, list):
        n_steps = [n_steps for i in range(n_states)]
    if auto_thin:
        thin = [
            1 for i in range(n_states)
        ]  # ignore thinning factor because we are post thinning by the auto-correlation times
    if isinstance(parallel, bool):
        parallel = [parallel for i in range(n_states)]

    samples = []
    log_posterior_ens = []
    auto_correlation = []
    if progress:
        print("\nRunning within-state sampler separately on each state")
        print("\nNumber of walkers               : ", n_walkers)
        print("\nNumber of states being sampled: ", n_states)
        print("Dimensions of each state: ", n_dims)

    for i in range(n_states):  # loop over states
        log_func = lambda x, i=i: log_posterior(x, i, *log_posterior_args)
        if parallel[i]:
            if n_processors == 1:
                n_processors = multiprocessing.cpu_count()  # set number of processors

            with multiprocessing.Pool(processes=n_processors) as pool:
                sampler = emcee.EnsembleSampler(  # instantiate emcee class
                    n_walkers[i], n_dims[i], log_func, pool=pool, **kwargs
                )
                sampler.run_mcmc(
                    pos[i],
                    n_steps[i],
                    progress=progress,
                    skip_initial_state_check=skip_initial_state_check,
                )  # run sampler
        else:
            sampler = emcee.EnsembleSampler(n_walkers[i], n_dims[i], log_func, **kwargs)
            sampler.run_mcmc(
                pos[i],
                n_steps[i],
                progress=progress,
                skip_initial_state_check=skip_initial_state_check,
            )  # run sampler in current state
        samples.append(
            sampler.get_chain(discard=discard[i], thin=thin[i], flat=True)
        )  # collect state ensemble
        log_posterior_ens.append(
            sampler.get_log_prob(discard=discard[i], thin=thin[i], flat=True)
        )  # collect state log_posterior values

        if auto_thin:
            if verbose:
                print("Performing auto thinning of ensemble...")
            auto_correlation.append([sampler.get_autocorr_time(tol=0)])
            if verbose:
                print(
                    "Auto thinning factor calculated = ",
                    int(np.ceil(np.max(auto_correlation[i]))),
                )

    if auto_thin:
        # we now thin the chains using the maximum auto_correlation function for each state to get independent samples for fitting
        # emcee manual suggests tau = sampler.get_autocorr_time(); burn_in = int(2 * np.max(tau)); thin = int(0.5 * np.min(tau))
        samples_auto, log_posterior_ens_auto = [], []
        for i in range(n_states):
            # thin = int(np.ceil(np.max(auto_correlation[i])))
            thin = int(
                np.ceil(0.5 * np.min(auto_correlation[i]))
            )  # use emcee suggestion
            burn_in = int(
                np.ceil(2.0 * np.max(auto_correlation[i]))
            )  # use emcee suggestion
            samples_auto.append(samples[i][burn_in::thin])
            log_posterior_ens_auto.append(log_posterior_ens[i][burn_in::thin])
        samples = samples_auto
        log_posterior_ens = log_posterior_ens_auto

    return samples, log_posterior_ens
