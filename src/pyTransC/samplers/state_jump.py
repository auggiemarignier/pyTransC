"""State-Jump Sampling for TransC."""

import multiprocessing
import random
import warnings
from functools import partial

import numpy as np
from tqdm import tqdm

from pytransc.utils.auto_pseudo import PseudoPrior


def run_state_jump_sampler(  # Independent state MCMC sampler on product space with proposal equal to pseudo prior
    n_walkers,
    n_steps,
    n_states: int,
    n_dims: list[int],
    pos,
    pos_state,
    log_posterior,
    log_pseudo_prior,
    log_proposal,
    log_posterior_args=[],
    log_pseudo_prior_args=[],
    log_proposal_args=[],
    prob_state=0.1,
    seed=61254557,
    parallel=False,
    n_processors=1,
    progress=False,
    suppress_warnings=False,
    verbose=False,
):
    """
    MCMC sampler over independent states using a Metropolis-Hastings algorithm and proposal equal to the supplied pseudo-prior function.

    Calculates Markov chain across states for state jump sampler

    Inputs:
    n_walkers - int               : number of random walkers used by state jump sampler.
    n_steps - int                 : number of steps required per walker.
    pos - n_walkers*n_dims*float   : list of starting locations of markov chains in each state.
    pos_state - n_walkers*int     : list of starting states of markov chains in each state.
    log_posterior()              : user supplied function to evaluate the log-posterior density for the ith state at location x.
                                    calling sequence log_posterior(x,i,*log_posterior_args)
    log_pseudo_prior()           : user supplied function to evaluate the log-pseudo-prior density for the ith state at location x.
                                    calling sequence log_posterior(x,i,*log_posterior_args).
                                    NB: must be normalized over respective state spaces.
    log_proposal()               : user supplied function to generate random deviate for ith state
                                    calling sequence log_proposal(xc,i,*log_proposal_args), where xc is the current location of the chain (allows for relative proposals)
                                    This is only used for within state moves, and not for between state moves for which it is effectively replaced by the pseudo-prior.
    log_posterior_args - list    : user defined (optional) list of additional arguments passed to log_posterior. See calling sequence above.
    log_pseudo_prior_args - list : user defined (optional) list of additional arguments passed to log_pseudo_prior. See calling sequence above.
    log_proposal_args - list     : user defined (optional) list of additional arguments passed to log_proposal. See calling sequence above.
    prob_state - float           : probability of proposal a state change per step of Markov chain (otherwise a parameter change within current state is proposed)
    seed - int                   : random number seed
    parallel - bool              : switch to make use of multiprocessing package to parallelize over walkers
    n_processors - int            : number of processors to distribute work across (if parallel=True, else ignored). Default = multiprocessing.cpu_count()/1 if parallel = True/False.
    progress - bool              : switch to report progress to standard out.
    suppress_warnings - bool      : switch to report detailed workings to standard out.
    verbose - bool               : switch to report detailed workings to standard out.

    Attributes defined/updated:
    nsamples - int                                : list of number of samples in each state (calculated from input ensembles if provided).
    n_walkers - int                                : number of random walkers used by state jump sampler.
    state_chain - n_walkers*n_steps*int             : list of states visited along the trans-C chain.
    state_chain_tot - n_walkers*n_steps*int         : array of cumulative number of visits to each state along the chains.
    model_chain - floats                          : list of trans-C sample along chain.
    alg - string                                  : string defining the sampler method used.

    Notes:
    A simple Metropolis-Hastings MCMC algorithm is used and applied to the product space formulation. Here moves between states are assumed to only perturb the state variable, k-> k'.
    This means that one only needs to generate a new model in state k' from the pseudo-prior of k'. The M-H condition then only involves the current model in state k and the new model in state k',
    with the acceptance criterion then equal to the ratio of the posteriors multiplied by the ratio of the normalized pseudo-priors.
    For within state moves the algorithm becomes normal M-H using a user supplied proposal function to generate new deviates within state k. The user can define this as relative to current model,
    or according to a prescribed PDF within the respective state, e.g. the pseudo-prior again. An independent user supplied proposal function is provided for flexibility.

    """

    if progress:
        print("\nRunning state-jump trans-C sampler")
        print("\nNumber of walkers               : ", n_walkers)
        print("Number of states being sampled  : ", n_states)
        print("Dimensions of each state        : ", n_dims)

    if parallel and not suppress_warnings:  # do some housekeeping checks
        if n_walkers == 1:
            warnings.warn(
                " Parallel mode used but only a single walker specified. Nothing to parallelize over?"
            )

    random.seed(seed)
    state_chain_tot = np.zeros((n_walkers, n_steps, n_states), dtype=int)
    state_chain = np.zeros((n_walkers, n_steps), dtype=int)
    model_chain = []
    accept_within = np.zeros(n_walkers)
    accept_between = np.zeros(n_walkers)
    prop_within = np.zeros(n_walkers)
    prop_between = np.zeros(n_walkers)

    if parallel:  # put random walkers on different processors
        if n_processors == 1:
            n_processors = (
                multiprocessing.cpu_count()
            )  # set number of processors equal to those available
        chunksize = int(np.ceil(n_walkers / n_processors))  # set work per processor
        jobs = [
            (pos_state[i], pos[i]) for i in range(n_walkers)
        ]  # input data for parallel jobs
        func = partial(
            _mcmc_walker,
            n_states=n_states,
            log_posterior=log_posterior,
            log_pseudo_prior=log_pseudo_prior,
            log_proposal=log_proposal,
            log_posterior_args=log_posterior_args,
            log_pseudo_prior_args=log_pseudo_prior_args,
            log_proposal_args=log_proposal_args,
            n_steps=n_steps,
            prob_state=prob_state,
            verbose=verbose,
        )
        result = []
        if progress:
            with multiprocessing.Pool(processes=n_processors) as pool:
                res = list(
                    tqdm(
                        pool.imap_unordered(func, jobs, chunksize=chunksize),
                        total=len(jobs),
                    )
                )
        else:
            pool = multiprocessing.Pool(processes=n_processors)
            res = pool.map(func, jobs, chunksize=chunksize)
        result.append(res)
        pool.close()
        pool.join()
        for i in range(n_walkers):  # decode the output
            state_chain_tot[i] = result[0][i][2]
            state_chain[i] = result[0][i][1]
            model_chain.append(result[0][i][0])
            accept_within[i] = result[0][i][3]
            accept_between[i] = result[0][i][5]
            prop_within[i] = result[0][i][4]
            prop_between[i] = result[0][i][6]

    else:
        for walker in _my_range(progress, n_walkers):  # loop over walkers
            current_state = pos_state[walker]  # initial state
            current_model = pos[walker]
            out = _mcmc_walker(
                n_states,
                [current_state, current_model],
                log_posterior,
                log_pseudo_prior,
                log_posterior_args,
                log_proposal,
                log_proposal_args,
                n_steps,
                prob_state,
                verbose,
            )

            (
                chain,
                state_chainw,
                state_chain_totw,
                accept_within[walker],
                prop_within[walker],
                accept_between[walker],
                prop_between[walker],
            ) = out
            state_chain_tot[walker] = state_chain_totw
            state_chain[walker] = state_chainw
            model_chain.append(chain)  # record locations visited for this walker

    return (
        model_chain,
        state_chain,
        state_chain_tot,
        accept_within,
        prop_within,
        accept_between,
        prop_between,
    )


def _my_range(progress: bool, length: int):
    if progress:
        return tqdm(range(length))
    return range(length)


def _mcmc_walker(
    n_states: int,
    cstate_cmodel,
    log_posterior,
    log_pseudo_prior: PseudoPrior,
    log_posterior_args,
    log_proposal,
    log_proposal_args,
    n_steps,
    prob_state,
    verbose,
):
    current_state, current_model = cstate_cmodel
    visits = np.zeros(n_states, dtype=int)
    log_posterior_current = log_posterior(
        current_model, current_state, *log_posterior_args
    )  # initial log-posterior
    log_pseudo_prior_current = log_pseudo_prior(
        current_model, current_state
    )  # initial log-pseudo prior
    chain = []
    state_chain_tot = np.zeros((n_steps, n_states), dtype=int)
    state_chain = np.zeros((n_steps), dtype=int)
    prop_between = 0
    prop_within = 0
    accept_within = 0
    accept_between = 0

    for chain_step in range(n_steps):  # loop over markov chain steps
        if random.random() < prob_state:  # Choose to propose a new state
            states = list(range(n_states))  # list of all states
            states.remove(current_state)  # list of available states
            proposed_state = random.choice(states)  # choose proposed state
            if verbose:
                print("current state", current_state, " propose state", proposed_state)
            within = False
            prop_between += 1
            proposed_model = log_pseudo_prior.draw_deviate(proposed_state)
            log_pseudo_prior_proposed = log_pseudo_prior(proposed_model, proposed_state)

            log_proposal_prob = (
                log_pseudo_prior_current - log_pseudo_prior_proposed
            )  # log difference in pseduo-priors

        else:  # Choose to propose a new model within current state
            proposed_state = np.copy(current_state)  # retain current state
            if verbose:
                print("within state", current_state, " model change")
            within = True
            prop_within += 1
            log_proposal_prob, proposed_model = log_proposal(
                current_model, proposed_state, *log_proposal_args
            )  # generate proposed model in current state and calculate log density ratio

        log_posterior_proposed = log_posterior(
            proposed_model, proposed_state, *log_posterior_args
        )  # log posterior for proposed state

        log_proposal_ratio = (
            log_posterior_proposed - log_posterior_current + log_proposal_prob
        )  # Metropolis-Hastings acceptance criterion

        if log_proposal_ratio >= np.log(random.random()):  # Accept move
            if verbose:
                print(" Accept move")
                print(" current model", current_model, "proposed model", proposed_model)
            visits[proposed_state] += 1
            current_state = np.copy(proposed_state)
            current_model = np.copy(proposed_model)
            log_posterior_current = np.copy(log_posterior_proposed)
            if within:
                log_pseudo_prior_proposed = log_pseudo_prior(
                    proposed_model,
                    proposed_state,
                )  # record log pseudo-prior for new state
            log_pseudo_prior_current = np.copy(log_pseudo_prior_proposed)
            if within:
                accept_within += 1
            else:
                accept_between += 1
        else:  # Reject move
            if verbose:
                print(" Reject move")
                print(" current model", current_model, "proposed model", proposed_model)
            visits[current_state] += 1

        chain.append(current_model)
        state_chain[chain_step] = current_state  # record state for this step and walker
        state_chain_tot[chain_step] = (
            visits  # record cumulative tally of states visited for this step and walker
        )

    return (
        chain,
        state_chain,
        state_chain_tot,
        accept_within,
        prop_within,
        accept_between,
        prop_between,
    )
