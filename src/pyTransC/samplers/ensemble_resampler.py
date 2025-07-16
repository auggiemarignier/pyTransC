"""Ensemble resampler for TransC."""

import multiprocessing
import random
from functools import partial

import numpy as np
from tqdm import tqdm


def run_ensemble_resampler(  # Independent state Marginal Likelihoods from pre-computed posterior and pseudo prior ensembles
    n_walkers,
    n_steps,
    n_dims: list[int],
    log_posterior_ens,
    log_pseudo_prior_ens,
    seed=61254557,
    parallel=False,
    n_processors=1,
    state_proposal_weights=None,
    progress=False,
):
    """
    MCMC sampler over independent states using a Markov Chain.

    Calculates relative evidence of each state by sampling over previously computed posterior ensembles for each state.
    Requires only log density values for posterior and pseudo priors at the sample locations (not actual samples).
    This routine is an alternate to run_ens_mcint(), using the same inputs of log density values of posterior samples within each state.
    Here a single Markov chain is used.

    Inputs:
    n_walkers - int                                                       : number of random walkers used by ensemble resampler.
    n_steps - int                                                         : number of Markov chain steps to perform
    log_posterior_ens -  list of floats, [i,n[i]], (i=1,...,n_states)     : log-posterior of ensembles in each state, where n[i] is the number of samples in the ith state.
    log_pseudo_prior_ens -  list of floats, [i,n[i]], (i=1,...,n_states)  : log-pseudo prior of samples in each state, where n[i] is the number of samples in the ith state.
    seed - int                                                           : random number seed
    parallel - bool                                                      : switch to make use of multiprocessing package to parallelize over walkers
    n_processors - int                                                    : number of processors to distribute work across (if parallel=True, else ignored). Default = multiprocessing.cpu_count()/1 if parallel = True/False.
    progress - bool                                                      : option to write diagnostic info to standard out

    Attributes defined/updated:
    n_states - int                                 : number of independent states (calculated from input ensembles if provided).
    n_samples - int                                : list of number of samples in each state (calculated from input ensembles if provided).
    state_chain_tot - n_samples*int                : array of states visited along the trans-C chain.
    alg - string                                  : string defining the sampler method used.


    Notes:
    The input posterior samples and log posterior values in each state can be either be calculated using utility routine 'run_mcmc_per_state', or provided by the user.
    The input log values of pseudo prior samples in each state can be either be calculated using utility routine 'run_fitmixture', or provided by the user.

    """

    n_states = len(log_posterior_ens)
    n_samples = [len(log_post_ens) for log_post_ens in log_posterior_ens]

    print("\nRunning ensemble resampler")
    print("\nNumber of walkers               : ", n_walkers)
    print("Number of states being sampled  : ", n_states)
    print("Dimensions of each state        : ", n_dims)

    random.seed(seed)
    state_chain_tot = np.zeros((n_walkers, n_steps, n_states), dtype=int)
    state_chain = np.zeros((n_walkers, n_steps), dtype=int)
    accept_between = np.zeros(n_walkers, dtype=int)
    if parallel:
        if n_processors == 1:
            n_processors = (
                multiprocessing.cpu_count()
            )  # set number of processors equal to those available
        chunk_size = int(np.ceil(n_walkers / n_processors))  # set work per processor
        jobs = random.choices(
            range(n_states), k=n_walkers
        )  # input data for parallel jobs
        func = partial(
            _mcmc_walker_ens,  # create reduced one argument function for passing to pool.map())
            n_states=n_states,
            n_samples=n_samples,
            n_steps=n_steps,
            log_posterior_ens=log_posterior_ens,
            log_pseudo_prior_ens=log_pseudo_prior_ens,
            state_proposal_weights=state_proposal_weights,
        )
        result = []
        if progress:
            with multiprocessing.Pool(processes=n_processors) as pool:
                res = list(
                    tqdm(pool.imap(func, jobs, chunksize=chunk_size), total=len(jobs))
                )
        else:
            pool = multiprocessing.Pool(processes=n_processors)
            res = pool.map(func, jobs, chunksize=chunk_size)
        result.append(res)
        pool.close()
        pool.join()
        for i in range(n_walkers):  # decode the output
            state_chain_tot[i] = result[0][i][1]
            state_chain[i] = result[0][i][0]
            accept_between[i] = result[0][i][2]

        pass
    else:
        for walker in _my_range(progress, n_walkers):
            cstate = random.choice(
                range(n_states)
            )  # choose initial current state randomly
            state_chain[walker], state_chain_tot[walker], accept_between[walker] = (
                _mcmc_walker_ens(
                    n_states,
                    n_samples,
                    cstate,
                    n_steps,
                    log_posterior_ens,
                    log_pseudo_prior_ens,
                    state_proposal_weights=state_proposal_weights,
                )
            )  # carry out an mcmc walk between ensembles

    return state_chain, state_chain_tot, accept_between


def _mcmc_walker_ens(
    n_states: int,
    n_samples: list[int],
    current_state,
    n_steps,
    log_posterior_ens,
    log_pseudo_prior_ens,
    state_proposal_weights=None,
    verbose=False,
):
    """Internal one chain MCMC sampler used by run_ensemble_resampler()."""

    visits = np.zeros(n_states)
    state_chain_tot = np.zeros((n_steps, n_states), dtype=int)
    state_chain = np.zeros((n_steps), dtype=int)
    current_member = random.choice(
        range(n_samples[current_state])
    )  # randomly choose ensemble member from current state
    visits[current_state] += 1
    state_chain[0] = current_state  # record initial state for this step and walker
    state_chain_tot[0] = visits  # record initial current state visited by chain
    accept = 0
    if state_proposal_weights is None:
        state_proposal_weights = np.ones((n_states, n_states))
    else:
        np.fill_diagonal(state_proposal_weights, 0.0)  # ensure diagonal is zero
        state_proposal_weights = state_proposal_weights / state_proposal_weights.sum(
            axis=1, keepdims=1
        )  # set row sums to unity

    for chain_step in range(n_steps - 1):  # loop over markov chain steps
        states = list(range(n_states))  # list of all states
        states.remove(current_state)  # list of available states
        weights = state_proposal_weights[
            current_state, np.delete(np.arange(n_states), current_state)
        ]
        proposed_state = random.choices(states, weights=weights)[
            0
        ]  # choose proposed state
        proposed_member = random.choice(
            range(n_samples[proposed_state])
        )  # randomly select ensemble member from proposed state

        log_pseudo_prior_current = log_pseudo_prior_ens[current_state][current_member]
        log_pseudo_prior_proposed = log_pseudo_prior_ens[proposed_state][
            proposed_member
        ]
        log_posterior_current = log_posterior_ens[current_state][current_member]
        log_posterior_proposed = log_posterior_ens[proposed_state][proposed_member]

        log_proposal_prob = np.log(
            state_proposal_weights[proposed_state, current_state]
        ) - np.log(state_proposal_weights[current_state, proposed_state])

        log_proposal_ratio = (
            log_posterior_proposed
            + log_pseudo_prior_current
            - log_posterior_current
            - log_pseudo_prior_proposed
            + log_proposal_prob
        )  # Metropolis-Hastings acceptance criteria

        if log_proposal_ratio >= np.log(random.random()):  # Accept move between states
            visits[proposed_state] += 1
            current_state = np.copy(proposed_state)
            current_member = np.copy(proposed_member)
            accept += 1
        else:
            # Reject move between states
            visits[current_state] += 1

        state_chain[chain_step + 1] = (
            current_state  # record state for this step and walker
        )
        state_chain_tot[chain_step + 1] = (
            visits  # record current state visited by chain
        )

    return state_chain, state_chain_tot, accept


def _my_range(progress: bool, length: int):
    if progress:
        return tqdm(range(length))
    return range(length)
