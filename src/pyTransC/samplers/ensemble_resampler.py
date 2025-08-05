"""Ensemble resampler for TransC."""

import logging
import multiprocessing
import random
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from tqdm import tqdm

from ..utils.types import Int2DArray, StateOrderedEnsemble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """Convenience dataclass to hold a single sample from the ensemble resampler.

    This only deals with indices of the member in the ensemble and its state index.
    """

    member: int  # index of the model in the ensemble
    state: int


@dataclass
class EnsembleResamplerChain:
    """Data class to hold the results of the ensemble resampler."""

    n_states: int  # This could be inferred from state_chain assuming every state is visited at least once.  Requiring it at initialisation makes it more robust for downstream tasks.
    member_chain: list[int] = field(default_factory=list, init=False)
    state_chain: list[int] = field(default_factory=list, init=False)
    n_proposed: int = field(default=0, init=False)
    n_accepted: int = field(default=0, init=False)

    def __post_init__(self):
        """Post-initialization checks."""
        if not isinstance(self.n_states, int) or self.n_states <= 0:
            raise ValueError("n_states must be a positive integer.")

    @property
    def state_chain_tot(self) -> Int2DArray:
        """Running cumulative tally of states visited."""

        from ._utils import count_visits_to_states

        return count_visits_to_states(
            np.array(self.state_chain, dtype=int), self.n_states
        )

    @property
    def n_steps(self) -> int:
        """Number of steps in the chain, calculated as the total number of proposals."""
        return self.n_proposed


def update_chain(
    chain: EnsembleResamplerChain,
    sample: Sample,
    proposal_accepted: bool,
) -> None:
    """Update the chain with a new sample.

    Args:
        chain (EnsembleResamplerChain): The chain to update.
        sample (Sample): The new sample to add.  Note that this sample is the outcome of the acceptance/rejection step i.e. it will be the previous sample if the proposal was rejected.
        proposal_accepted (bool): Whether the proposal was accepted or not.
    """
    chain.member_chain.append(sample.member)
    chain.state_chain.append(sample.state)
    chain.n_accepted += int(proposal_accepted)
    chain.n_proposed += 1


@dataclass
class MultiWalkerEnsembleResamplerChain:
    """Class to hold and manage multiple ensemble resampler chains from different walkers."""

    chains: list[EnsembleResamplerChain] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization checks."""
        if not self.chains:
            # no chains, not a problem
            return

        if any(not isinstance(chain, EnsembleResamplerChain) for chain in self.chains):
            raise TypeError("All chains must be instances of EnsembleResamplerChain.")

        n_states = self.chains[0].n_states
        if any(chain.n_states != n_states for chain in self.chains[1:]):
            raise ValueError("All chains must have the same number of states.")

    @property
    def n_walkers(self) -> int:
        """Number of walkers in the multi-walker chain."""
        return len(self.chains)

    @property
    def n_states(self) -> int:
        """Number of states in the multi-walker chain."""
        if self.chains:
            return self.chains[0].n_states
        return 0

    @property
    def member_chain(self) -> list[list[int]]:
        """Concatenated member index chain from all walkers."""
        return [chain.member_chain for chain in self.chains]

    @property
    def state_chain(self) -> np.ndarray:
        """Concatenated state chain from all walkers."""
        return np.array([chain.state_chain for chain in self.chains])

    @property
    def state_chain_tot(self) -> np.ndarray:
        """Concatenated total state chain from all walkers."""
        return np.array([chain.state_chain_tot for chain in self.chains])

    @property
    def n_accepted(self) -> np.ndarray:
        """Number of accepted proposals for each walker."""
        return np.array([chain.n_accepted for chain in self.chains])

    @property
    def n_proposed(self) -> np.ndarray:
        """Number of proposals for each walker."""
        return np.array([chain.n_proposed for chain in self.chains])


def run_ensemble_resampler(  # Independent state Marginal Likelihoods from pre-computed posterior and pseudo prior ensembles
    n_walkers,
    n_steps,
    n_states: int,
    n_dims: list[int],
    log_posterior_ens: StateOrderedEnsemble,
    log_pseudo_prior_ens: StateOrderedEnsemble,
    seed=61254557,
    parallel=False,
    n_processors=1,
    state_proposal_weights: list[list[float]] | None = None,
    progress=False,
) -> MultiWalkerEnsembleResamplerChain:
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

    n_samples = [len(log_post_ens) for log_post_ens in log_posterior_ens]

    if state_proposal_weights is None:
        # uniform proposal weights
        _state_proposal_weights = [[1.0] * n_states] * n_states
    else:
        _state_proposal_weights = np.array(state_proposal_weights)
        np.fill_diagonal(_state_proposal_weights, 0.0)  # ensure diagonal is zero
        _state_proposal_weights = _state_proposal_weights / _state_proposal_weights.sum(
            axis=1, keepdims=True
        )  # set row sums to unity
        _state_proposal_weights = _state_proposal_weights.tolist()

    logger.info("\nRunning ensemble resampler")
    logger.info("\nNumber of walkers               : %d", n_walkers)
    logger.info("Number of states being sampled  : %d", n_states)
    logger.info("Dimensions of each state        : %s", n_dims)

    random.seed(seed)
    if parallel:
        chains = _run_mcmc_walker_parallel(
            n_walkers,
            n_states,
            n_samples,
            n_steps,
            log_posterior_ens,
            log_pseudo_prior_ens,
            state_proposal_weights=_state_proposal_weights,
            n_processors=n_processors,
            progress=progress,
        )

    else:
        chains = _run_mcmc_walker_serial(
            n_walkers,
            n_states,
            n_samples,
            n_steps,
            log_posterior_ens,
            log_pseudo_prior_ens,
            state_proposal_weights=_state_proposal_weights,
            progress=progress,
        )

    return MultiWalkerEnsembleResamplerChain(chains)


def _run_mcmc_walker_parallel(
    n_walkers: int,
    n_states: int,
    n_samples: list[int],
    n_steps: int,
    log_posterior_ens: StateOrderedEnsemble,
    log_pseudo_prior_ens: StateOrderedEnsemble,
    state_proposal_weights: list[list[float]],
    n_processors: int = 1,
    progress: bool = False,
) -> list[EnsembleResamplerChain]:
    """Run the ensemble resampler in parallel using multiprocessing.

    THIS HAS NOT BEEN TESTED YET.
    """
    if n_processors == 1:
        # set number of processors equal to those available
        n_processors = multiprocessing.cpu_count()

    # set work per processor
    chunk_size = int(np.ceil(n_walkers / n_processors))

    # input data for parallel jobs
    jobs = random.choices(range(n_states), k=n_walkers)

    # create reduced one argument function for passing to pool.map())
    func = partial(
        _mcmc_walker,
        n_states=n_states,
        n_samples=n_samples,
        n_steps=n_steps,
        log_posterior_ens=log_posterior_ens,
        log_pseudo_prior_ens=log_pseudo_prior_ens,
        state_proposal_weights=state_proposal_weights,
    )

    # run the parallel jobs
    if progress:
        with multiprocessing.Pool(processes=n_processors) as pool:
            chains: list[EnsembleResamplerChain] = list(
                tqdm(pool.imap(func, jobs, chunksize=chunk_size), total=len(jobs))
            )
    else:
        pool = multiprocessing.Pool(processes=n_processors)
        chains: list[EnsembleResamplerChain] = pool.map(
            func, jobs, chunksize=chunk_size
        )
        pool.close()
        pool.join()

    return chains


def _run_mcmc_walker_serial(
    n_walkers: int,
    n_states: int,
    n_samples: list[int],
    n_steps: int,
    log_posterior_ens: StateOrderedEnsemble,
    log_pseudo_prior_ens: StateOrderedEnsemble,
    state_proposal_weights: list[list[float]],
    progress: bool = False,
) -> list[EnsembleResamplerChain]:
    chains: list[EnsembleResamplerChain] = []
    for _ in tqdm(range(n_walkers), disable=not progress):
        # choose initial current state randomly
        current_state = random.choice(range(n_states))

        # carry out an mcmc walk between ensembles
        chains.append(
            _mcmc_walker(
                n_states,
                n_samples,
                current_state,
                n_steps,
                log_posterior_ens,
                log_pseudo_prior_ens,
                state_proposal_weights=state_proposal_weights,
            )
        )

    return chains


def _mcmc_walker(
    n_states: int,
    n_samples: list[int],
    current_state,
    n_steps,
    log_posterior_ens: StateOrderedEnsemble,
    log_pseudo_prior_ens: StateOrderedEnsemble,
    state_proposal_weights: list[list[float]],
):
    """Internal one chain MCMC sampler used by run_ensemble_resampler()."""

    current_member = random.choice(
        range(n_samples[current_state])
    )  # randomly choose ensemble member from current state
    sample = Sample(current_member, current_state)
    chain = EnsembleResamplerChain(n_states)
    for _ in range(n_steps - 1):  # loop over markov chain steps
        sample, accepted = _chain_step(
            n_states,
            n_samples,
            sample,
            log_posterior_ens,
            log_pseudo_prior_ens,
            state_proposal_weights,
        )
        update_chain(chain, sample, accepted)

    return chain


def _chain_step(
    n_states: int,
    n_samples: list[int],
    current: Sample,
    log_posterior_ens: StateOrderedEnsemble,
    log_pseudo_prior_ens: StateOrderedEnsemble,
    state_proposal_weights: list[list[float]],  # 2D q(k|k') matrix
) -> tuple[Sample, bool]:
    proposed = _propose_sample(
        current.state, n_states, n_samples, state_proposal_weights
    )

    log_prob_proposed = _log_prob_sample(
        proposed, log_posterior_ens, log_pseudo_prior_ens
    )
    log_prob_current = _log_prob_sample(
        current, log_posterior_ens, log_pseudo_prior_ens
    )

    log_proposal_prob = np.log(
        state_proposal_weights[proposed.state][current.state]
    ) - np.log(state_proposal_weights[current.state][proposed.state])

    # Metropolis-Hastings acceptance criteria
    log_proposal_ratio = log_prob_proposed - log_prob_current + log_proposal_prob
    accept = log_proposal_ratio >= np.log(random.random())

    next_ = proposed if accept else current

    return next_, accept


def _propose_sample(
    current_state: int,
    n_states: int,
    n_samples: list[int],
    weights: list[list[float]],
) -> Sample:
    """Propose a new sample based on the current state and weights."""
    proposed_state = _propose_state(current_state, n_states, weights)
    proposed_member = _propose_member_in_state(n_samples[proposed_state])
    return Sample(proposed_member, proposed_state)


def _propose_state(
    current_state: int, n_states: int, weights: list[list[float]]
) -> int:
    """Propose a new state based on the current state."""
    inactive_states = [i for i in range(n_states) if i != current_state]
    _weights = [weights[current_state][i] for i in inactive_states]
    return random.choices(inactive_states, weights=_weights)[0]


def _propose_member_in_state(n_samples: int) -> int:
    """Propose a new member within the current state.

    Args:
        n_samples (int): Number of samples in the current state.
    """
    return random.choice(range(n_samples))


def _log_prob_sample(
    sample: Sample,
    log_posterior_ens: StateOrderedEnsemble,
    log_pseudo_prior_ens: StateOrderedEnsemble,
) -> float:
    """Calculate the log probability of a sample."""
    return (
        log_posterior_ens[sample.state][sample.member]
        - log_pseudo_prior_ens[sample.state][sample.member]
    )
