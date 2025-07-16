"""Product-Space Sampling for TransC."""

import multiprocessing
import random
import warnings
from functools import partial

import emcee
import numpy as np


def run_product_space_sampler(  # Independent state Metropolis algorithm sampling across product space. This is algorithm 'TransC-product-space'
    n_walkers: int,
    n_steps: int,
    n_states: int,
    n_dims: list[int],
    pos,
    pos_state,
    log_posterior,
    log_pseudo_prior,
    log_posterior_args=[],
    log_pseudo_prior_args=[],
    seed=61254557,
    parallel=False,
    n_processors=1,
    progress=False,
    suppress_warnings=False,  # bool to suppress warnings
    my_pool=False,
    skip_initial_state_check=False,
    **kwargs,
) -> emcee.EnsembleSampler:
    """
    MCMC sampler over independent states using emcee fixed dimension sampler over trans-C product space.

    Inputs:
    n_walkers - int               : number of random walkers used by product_space sampler.
    n_steps - int                 : number of steps required per walker.
    pos - n_walkers*n_dims*float   : list of starting locations of markov chains in each state.
    pos_state - n_walkers*int     : list of starting states of markov chains in each state.
    log_posterior()              : user supplied function to evaluate the log-posterior density for the ith state at location x.
                                   calling sequence log_posterior(x,i,*log_posterior_args)
    log_pseudo_prior()           : user supplied function to evaluate the log-pseudo-prior density for the ith state at location x.
                                   calling sequence log_posterior(x,i,*log_posterior_args).
                                   NB: must be normalized over respective state spaces.
    log_posterior_args - list    : user defined (optional) list of additional arguments passed to log_posterior. See calling sequence above.
    log_pseudo_prior_args - list : user defined (optional) list of additional arguments passed to log_pseudo_prior. See calling sequence above.
    prob_state - float           : probability of proposal a state change per step of Markov chain (otherwise a parameter change within current state is proposed)
    seed - int                   : random number seed
    parallel - bool              : switch to make use of multiprocessing package to parallelize over walkers
    n_processors - int            : number of processors to distribute work across (if parallel=True, else ignored). Default = multiprocessing.cpu_count()/1 if parallel = True/False.
    progress - bool              : switch to report progress to standard out.
    suppress_warnings - bool      : switch to suppress warnings.
    my_pool - bool                : switch to use local multiprocessing pool for emcee (experimental feature not recommended)
    kwargs - dict                : dictionary of optional arguments passed to emcee.

    """

    random.seed(seed)

    if progress:
        print("\nRunning product space trans-C sampler")
        print("\nNumber of walkers               : ", n_walkers)
        print("Number of states being sampled  : ", n_states)
        print("Dimensions of each state        : ", n_dims)

    if parallel and not suppress_warnings:  # do some housekeeping checks
        if n_walkers == 1:
            warnings.warn(
                " Parallel mode used but only a single walker specified. Nothing to parallelize over?"
            )

    ndim_ps = np.sum(n_dims) + 1  # dimension of product space

    pos_ps = _model_vectors2product_space(
        pos, pos_state, n_walkers, sum(n_dims), n_states
    )  # convert initial walker positions to product space model vectors

    log_func = partial(
        _product_space_log_prob,
        n_states=n_states,
        n_dims=n_dims,
        log_posterior=log_posterior,
        log_pseudo_prior=log_pseudo_prior,
        log_posterior_args=log_posterior_args,
        log_pseudo_prior_args=log_pseudo_prior_args,
    )

    if parallel:
        if n_processors == 1:
            n_processors = (
                multiprocessing.cpu_count()
            )  # set number of processors equal to those available

        if my_pool:  # try to run emcee myself on separate cores (doesn't make sense for emcee to do this as n_walkers > 2*ndim for performance)
            chunksize = int(
                np.ceil(n_walkers / n_processors)
            )  # set work per0 processor
            jobs = [pos_ps[i] for i in range(n_walkers)]  # input data for parallel jobs
            print(" n_steps", n_steps)
            func = partial(
                _my_emcee,
                n_steps=n_steps,
                log_func=log_func,
                n_dim=ndim_ps,
                progress=progress,
                kwargs=kwargs,
            )
            # return func,jobs,n_processors,chunksize
            result = []
            pool = multiprocessing.Pool(processes=n_processors)
            res = pool.map(func, jobs, chunksize=chunksize)
            result.append(res)
            pool.close()
            pool.join()
            return result

        else:  # use emcee in parallel
            with multiprocessing.Pool() as pool:
                sampler = emcee.EnsembleSampler(  # instantiate emcee class
                    n_walkers, ndim_ps, log_func, pool=pool, **kwargs
                )

                sampler.run_mcmc(pos_ps, n_steps, progress=progress)  # run sampler

    else:
        sampler = emcee.EnsembleSampler(  # instantiate emcee class
            n_walkers, ndim_ps, log_func, **kwargs
        )

        sampler.run_mcmc(
            pos_ps,
            n_steps,
            progress=progress,
            skip_initial_state_check=skip_initial_state_check,
        )  # run sampler

    return sampler


def _my_emcee(pos, n_steps, log_func, n_dim, progress, kwargs):
    # instantiate emcee class with a single walker
    sampler = emcee.EnsembleSampler(1, n_dim, log_func, **kwargs)
    sampler.run_mcmc(pos, n_steps, progress=progress)  # run sampler

    return sampler


def _product_space_vector2model(
    x: np.ndarray, n_states: int, n_dims: list[int]
):  # convert a combined product space model space vector to model vector in each state
    """
    Internal utility routine to convert a single vector in product state format to a list of vectors of differing length one per state.

    This routine is the inverse operation to routine '_model_vectors2product_space()'

    Inputs:
    x - float array or list : trans-C vectors in product space format. (length sum ndim[i], i=1,...,n_states)

    Returns:
    m - list of floats      : list of trans-C vectors one per state with format
                                m[i][v[i]], (i=1,...,n_states) where i is the state and v[i] is a model vector in state i.

    """
    m = []
    kk = 1
    for k in range(n_states):
        m.append(x[kk : kk + n_dims[k]])
        kk += n_dims[k]
    return m


def _model_vectors2product_space(
    m, states, n_walkers, ps_ndim: int, n_states: int
):  # convert model space vectors in each state to product space vectors
    """
    Internal utility routine to convert a list of vectors of differing length one per state to a single vector in product state format.

    This routine is the inverse operation to routine '_product_space_vector2model()' but over multiple walkers.

    Inputs:
    m - list of floats arrays      : list of trans-C vectors one per state with format
                                        m[i][v[i]], (i=1,...,n_states) where i is the state and v[i] is a vector in state i.
    states - n_walkers*int          : list of states for each walker/chain.
    n_walkers - int                 : number of walkers.

    Returns:
    x - float array or list : trans-C vectors in product space format. (length = n_walkers*(1 + sum ndim[i], i=1,...,n_states))

    """
    x = np.zeros((n_walkers, ps_ndim + 1))
    for j in range(n_walkers):
        x[j, 0] = states[j]
        x[j, 1:] = np.concatenate([m[i][j] for i in range(n_states)])
    return x


def _product_space_log_prob(
    x,
    n_states: int,
    n_dims: list[int],
    log_posterior,
    log_pseudo_prior,
    log_posterior_args,
    log_pseudo_prior_args,
):  # Calculate product space target PDF from posterior and pseudo-priors in each state
    """
    Internal utility routine to calculate the combined target density for product space vector i.e. sum of log posterior + log pseudo prior density of all states.

    here input vector is in product space format.

    Inputs:
    x - float array or list : trans-C vectors in product space format. (length = n_walkers*(1 + sum n_dim[i], i=1,...,n_states))
    log_posterior()              : user supplied function to evaluate the log-posterior density for the ith state at location x.
                                    calling sequence log_posterior(x,i,*log_posterior_args)
    log_pseudo_prior()           : user supplied function to evaluate the log-pseudo-prior density for the ith state at location x.
                                    calling sequence log_posterior(x,i,*log_posterior_args).
                                    NB: must be normalized over respective state spaces.
    log_posterior_args - list    : user defined (optional) list of additional arguments passed to log_posterior. See calling sequence above.
    log_pseudo_prior_args - list : user defined (optional) list of additional arguments passed to log_pseudo_prior. See calling sequence above.


    Returns:
    x - float array or list : trans-C vectors in product space format. (length sum ndim[i], i=1,...,n_states)

    """
    if x[0] < -0.5 or x[0] >= n_states - 0.5:
        return -np.inf
    state = int(np.rint(x[0]))
    state = int(np.min((state, n_states - 1)))
    state = int(np.max((state, 0)))
    m = _product_space_vector2model(x, n_states, n_dims)
    log_prob = log_posterior(m[state], state, *log_posterior_args)
    for i in range(n_states):
        if i != state:
            new = log_pseudo_prior(m[i], i, *log_pseudo_prior_args)
            log_prob += new
    return log_prob
