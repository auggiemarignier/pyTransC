"""State-Jump Sampling for TransC."""

import logging
import multiprocessing
import random
import warnings
from dataclasses import dataclass, field
from enum import StrEnum, auto
from functools import partial

import numpy as np
from tqdm import tqdm

from ..utils.types import (
    FloatArray,
    IntArray,
    MultiStateDensity,
    MultiWalkerModelChain,
    MultiWalkerStateChain,
    ProposableMultiStateDensity,
    SampleableMultiStateDensity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """Convenience dataclass to hold a single sample from the state jump sampler."""

    model: FloatArray
    state: int


class ProposalType(StrEnum):
    """Enum for proposal types in state jump sampler."""

    WITHIN_STATE = auto()
    BETWEEN_STATE = auto()


@dataclass
class StateJumpChain:
    """Dataclass to hold the results of the state jump sampler."""

    n_states: int  # This could be inferred from state_chain assuming every state is visited at least once.  Requiring it at initialisation makes it more robust for downstream tasks.
    model_chain: list[FloatArray] = field(default_factory=list, init=False)
    state_chain: list[int] = field(default_factory=list, init=False)
    accept_within: int = field(default=0, init=False)
    prop_within: int = field(default=0, init=False)
    accept_between: int = field(default=0, init=False)
    prop_between: int = field(default=0, init=False)

    def __repr__(self):
        """String representation of the state jump chain."""
        return f"StateJumpChain(n_states={self.n_states}, n_steps={self.n_steps})"

    def __post_init__(self):
        """Post-initialization checks."""
        if not isinstance(self.n_states, int) or self.n_states <= 0:
            raise ValueError("n_states must be a positive integer.")

    @property
    def state_chain_tot(self) -> IntArray:
        """Running cumulative tally of states visited."""

        from ._utils import count_visits_to_states

        return count_visits_to_states(
            np.array(self.state_chain, dtype=int), self.n_states
        )

    @property
    def n_steps(self) -> int:
        """Number of steps in the chain, calculated as the total number of proposals."""
        return self.prop_between + self.prop_within


def update_chain(
    chain: StateJumpChain,
    sample: Sample,
    proposal_type: ProposalType,
    proposal_accepted: bool,
) -> None:
    """Update the chain with a new sample and proposal type.

    Args:
        chain (StateJumpChain): The chain to update.
        sample (Sample): The new sample to add.  Note that this sample is the outcome of the acceptance/rejection step i.e. it will be the previous sample if the proposal was rejected.
        proposal_type (ProposalType): The type of proposal made (within or between state).
        proposal_accepted (bool): Whether the proposal was accepted or not.
    """
    chain.model_chain.append(sample.model)
    chain.state_chain.append(sample.state)

    if proposal_type == ProposalType.WITHIN_STATE:
        chain.prop_within += 1
        chain.accept_within += int(proposal_accepted)
    elif proposal_type == ProposalType.BETWEEN_STATE:
        chain.prop_between += 1
        chain.accept_between += int(proposal_accepted)


@dataclass
class MultiWalkerStateJumpChain:
    """Class to hold and manage multiple state jump chains from different walkers."""

    chains: list[StateJumpChain] = field(default_factory=list)

    def __repr__(self):
        """String representation of the multi-walker state jump chain."""
        return (
            f"MultiWalkerStateJumpChain(n_walkers={self.n_walkers}, "
            f"n_states={self.n_states}, n_steps={self.n_steps})"
        )

    def __post_init__(self):
        """Post-initialization checks."""
        if not self.chains:
            # no chains, not a problem
            return

        if any(not isinstance(chain, StateJumpChain) for chain in self.chains):
            raise TypeError("All chains must be instances of StateJumpChain.")

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
    def n_steps(self) -> int:
        """Total number of steps across all walkers."""
        if self.chains:
            # assuming all chains have the same number of steps
            return self.chains[0].n_steps
        return 0

    @property
    def model_chain(self) -> MultiWalkerModelChain:
        """Concatenated model chain from all walkers."""
        return [chain.model_chain for chain in self.chains]

    @property
    def state_chain(self) -> MultiWalkerStateChain:
        """Concatenated state chain from all walkers."""
        return np.array([chain.state_chain for chain in self.chains])

    @property
    def state_chain_tot(self) -> IntArray:
        """Concatenated total state chain from all walkers."""
        return np.array([chain.state_chain_tot for chain in self.chains])

    @property
    def accept_within(self) -> IntArray:
        """Number of within-state acceptances for each state."""
        return np.array([chain.accept_within for chain in self.chains])

    @property
    def prop_within(self) -> IntArray:
        """Number of within-state proposals for each walkers."""
        return np.array([chain.prop_within for chain in self.chains])

    @property
    def accept_between(self) -> IntArray:
        """Number of between-state acceptances for each walkers."""
        return np.array([chain.accept_between for chain in self.chains])

    @property
    def prop_between(self) -> IntArray:
        """Number of between-state proposals for each walkers."""
        return np.array([chain.prop_between for chain in self.chains])


def run_state_jump_sampler(  # Independent state MCMC sampler on product space with proposal equal to pseudo prior
    n_walkers,
    n_steps,
    n_states: int,
    n_dims: list[int],
    start_positions: list[FloatArray],
    start_states: list[int],
    log_posterior: MultiStateDensity,
    log_pseudo_prior: SampleableMultiStateDensity,
    log_proposal: ProposableMultiStateDensity,
    prob_state=0.1,
    seed=61254557,
    parallel=False,
    n_processors=1,
    progress=False,
) -> MultiWalkerStateJumpChain:
    """Run MCMC sampler with direct jumps between states of different states.

    This function implements trans-conceptual MCMC using a Metropolis-Hastings
    algorithm that can propose jumps between states with different numbers of
    parameters. Between-state moves use the pseudo-prior as the proposal, while
    within-state moves use a user-defined proposal function.

    Parameters
    ----------
    n_walkers : int
        Number of random walkers used by the state jump sampler.
    n_steps : int
        Number of MCMC steps required per walker.
    n_states : int
        Number of independent states in the problem.
    n_dims : list of int
        List of parameter dimensions for each state.
    start_positions : list of FloatArray
        Starting parameter positions for each walker. Each array should contain
        the initial parameter values for the corresponding starting state.
    start_states : list of int
        Starting state indices for each walker.
    log_posterior : MultiStateDensity
        Function to evaluate the log-posterior density at location x in state i.
        Must have signature log_posterior(x, state) -> float.
    log_pseudo_prior : SampleableMultiStateDensity
        Object with methods:
        - __call__(x, state) -> float: evaluate log pseudo-prior at x for state
        - draw_deviate(state) -> FloatArray: sample from pseudo-prior for state
        Note: Must be normalized over respective state spaces.
    log_proposal : ProposableMultiStateDensity
        Object with methods:
        - propose(x_current, state) -> FloatArray: propose new x in state
        - __call__(x, state) -> float: log proposal probability (for MH ratio)
    prob_state : float, optional
        Probability of proposing a state change per MCMC step. Otherwise,
        a parameter change within the current state is proposed. Default is 0.1.
    seed : int, optional
        Random number seed for reproducible results. Default is 61254557.
    parallel : bool, optional
        Whether to use multiprocessing to parallelize over walkers. Default is False.
    n_processors : int, optional
        Number of processors to use if parallel=True. Default is 1.
    progress : bool, optional
        Whether to display progress information. Default is False.

    Returns
    -------
    MultiWalkerStateJumpChain
        Chain results containing state sequences, model parameters, proposal
        acceptance rates, and diagnostics for all walkers.

    Notes
    -----
    The algorithm uses a Metropolis-Hastings sampler with two types of moves:

    1. **Between-state moves** (probability `prob_state`):
       - Propose a new state uniformly at random
       - Generate new parameters from the pseudo-prior of the proposed state
       - Accept/reject based on posterior and pseudo-prior ratios

    2. **Within-state moves** (probability `1 - prob_state`):
       - Use the user-defined proposal function to generate new parameters
       - Accept/reject using standard Metropolis-Hastings criterion

    The pseudo-prior must be normalized for the between-state acceptance
    criterion to satisfy detailed balance.

    Examples
    --------
    >>> results = run_state_jump_sampler(
    ...     n_walkers=32,
    ...     n_steps=1000,
    ...     n_states=3,
    ...     n_dims=[2, 3, 1],
    ...     start_positions=[[0.5, 0.5], [1.0, 0.0, -1.0], [2.0]],
    ...     start_states=[0, 1, 2],
    ...     log_posterior=my_log_posterior,
    ...     log_pseudo_prior=my_log_pseudo_prior,
    ...     log_proposal=my_log_proposal,
    ...     prob_state=0.2
    ... )
    """

    logger.info("Running state-jump trans-C sampler")
    logger.info("Number of walkers: %d", n_walkers)
    logger.info("Number of states being sampled: %d", n_states)
    logger.info("Dimensions of each state: %s", n_dims)

    random.seed(seed)

    if parallel:  # put random walkers on different processors
        chains = _run_state_jump_sampler_parallel(
            n_walkers,
            n_steps,
            n_states,
            start_positions,
            start_states,
            log_posterior,
            log_pseudo_prior,
            log_proposal,
            prob_state=prob_state,
            n_processors=n_processors,
            progress=progress,
        )
    else:
        chains = _run_state_jump_sampler_serial(
            n_walkers,
            n_steps,
            n_states,
            start_positions,
            start_states,
            log_posterior,
            log_pseudo_prior,
            log_proposal,
            prob_state=prob_state,
            progress=progress,
        )
    return MultiWalkerStateJumpChain(chains)


def _run_state_jump_sampler_parallel(
    n_walkers: int,
    n_steps: int,
    n_states: int,
    start_positions: list[FloatArray],
    start_states: list[int],
    log_posterior: MultiStateDensity,
    log_pseudo_prior: SampleableMultiStateDensity,
    log_proposal: ProposableMultiStateDensity,
    prob_state=0.1,
    n_processors=1,
    progress=False,
) -> list[StateJumpChain]:
    """Run the state jump sampler in parallel mode using multiprocessing.

    THIS HAS NOT BEEN TESTED!
    """
    if n_walkers == 1:
        warnings.warn(
            " Parallel mode used but only a single walker specified. Nothing to parallelize over?"
        )
    if n_processors == 1:
        n_processors = multiprocessing.cpu_count()
    chunksize = int(np.ceil(n_walkers / n_processors))  # set work per processor
    jobs = [
        (start_positions[i], start_states[i]) for i in range(n_walkers)
    ]  # input data for parallel jobs
    func = partial(
        _mcmc_walker,
        n_states=n_states,
        log_posterior=log_posterior,
        log_pseudo_prior=log_pseudo_prior,
        log_proposal=log_proposal,
        n_steps=n_steps,
        prob_state=prob_state,
    )
    if progress:
        with multiprocessing.Pool(processes=n_processors) as pool:
            chains: list[StateJumpChain] = list(
                tqdm(
                    pool.imap_unordered(func, jobs, chunksize=chunksize),
                    total=len(jobs),
                )
            )
    else:
        pool = multiprocessing.Pool(processes=n_processors)
        chains: list[StateJumpChain] = pool.map(func, jobs, chunksize=chunksize)
        pool.close()
        pool.join()

    return chains


def _run_state_jump_sampler_serial(
    n_walkers: int,
    n_steps: int,
    n_states: int,
    start_positions: list[FloatArray],
    start_states: list[int],
    log_posterior: MultiStateDensity,
    log_pseudo_prior: SampleableMultiStateDensity,
    log_proposal: ProposableMultiStateDensity,
    prob_state=0.1,
    progress=False,
) -> list[StateJumpChain]:
    """Run the state jump sampler in serial mode."""

    chains: list[StateJumpChain] = []
    for walker in tqdm(range(n_walkers), disable=not progress):
        initial_state = start_states[walker]
        initial_model = start_positions[walker]
        state_jump_chain = _mcmc_walker(
            n_states,
            initial_state,
            initial_model,
            log_posterior,
            log_pseudo_prior,
            log_proposal,
            n_steps,
            prob_state,
        )

        chains.append(state_jump_chain)

    return chains


def _mcmc_walker(
    n_states: int,
    initial_state: int,
    initial_model: FloatArray,
    log_posterior: MultiStateDensity,
    log_pseudo_prior: SampleableMultiStateDensity,
    log_proposal: ProposableMultiStateDensity,
    n_steps: int,
    prob_state: float,
) -> StateJumpChain:
    chain = StateJumpChain(n_states)
    sample = Sample(model=initial_model, state=initial_state)
    for _ in range(n_steps):  # loop over markov chain steps
        sample, proposal_type, accept = _chain_step(
            sample,
            log_posterior,
            log_pseudo_prior,
            log_proposal,
            n_states,
            prob_state,
        )
        update_chain(chain, sample, proposal_type, accept)

    return chain


def _chain_step(
    current: Sample,
    log_posterior: MultiStateDensity,
    log_pseudo_prior: SampleableMultiStateDensity,
    log_proposal: ProposableMultiStateDensity,
    n_states: int,
    prob_state: float,
) -> tuple[Sample, ProposalType, bool]:
    """Perform a single step of the state jump sampler.

    Returns:
        Sample: The next sample after the step, regardless of acceptance.
        ProposalType: The type of proposal made (within or between state).
        bool: Whether the proposal was accepted or not.
    """
    if random.random() < prob_state:  # Choose to propose a new state
        proposal_type = ProposalType.BETWEEN_STATE
        proposed = _between_state_proposal(log_pseudo_prior, current.state, n_states)
        log_proposal_prob_ratio = _between_state_log_proposal_prob_ratio(
            log_pseudo_prior, proposed, current
        )

        logger.debug(
            "Current state: %d, proposing state: %d", current.state, proposed.state
        )

    else:  # Choose to propose a new model within current state
        proposal_type = ProposalType.WITHIN_STATE
        proposed = _within_state_proposal(log_proposal, current)
        log_proposal_prob_ratio = _within_state_log_proposal_prob_ratio(
            log_proposal, proposed
        )

        logger.debug("Within state %d, proposing model change", current.state)

    log_posterior_prob_ratio = _log_posterior_prob_ratio(
        log_posterior, proposed, current
    )

    # Metropolis-Hastings acceptance criterion
    log_proposal_ratio = log_posterior_prob_ratio + log_proposal_prob_ratio
    accept = log_proposal_ratio >= np.log(random.random())

    next_ = proposed if accept else current

    logger.debug(
        "%s move: current=%s, proposed=%s",
        "Accepting" if accept else "Rejecting",
        current.model,
        proposed.model,
    )

    return next_, proposal_type, accept


def _between_state_proposal(
    log_pseudo_prior: SampleableMultiStateDensity,
    current_state: int,
    n_states: int,
) -> Sample:
    """Propose a new state different from the current state."""
    inactive_states = list(range(n_states))
    inactive_states.remove(current_state)
    proposed_state = random.choice(inactive_states)
    proposed_model = log_pseudo_prior.draw_deviate(proposed_state)
    return Sample(model=proposed_model, state=proposed_state)


def _between_state_log_proposal_prob_ratio(
    log_pseudo_prior: MultiStateDensity, proposed: Sample, current: Sample
) -> float:
    """Calculate the log proposal probability ratio for a between-state proposal.

    This is simply the log of the ratio of the pseudo-prior densities for the current and proposed models.
    """
    log_pseudo_prior_current = log_pseudo_prior(current.model, current.state)
    log_pseudo_prior_proposed = log_pseudo_prior(proposed.model, proposed.state)
    return log_pseudo_prior_current - log_pseudo_prior_proposed


def _within_state_proposal(
    log_proposal: ProposableMultiStateDensity, current: Sample
) -> Sample:
    """Propose a new model within the current state."""
    return Sample(
        state=current.state, model=log_proposal.propose(current.model, current.state)
    )


def _within_state_log_proposal_prob_ratio(
    log_proposal: MultiStateDensity, proposed: Sample
) -> float:
    """Calculate the log proposal probability ratio for a within-state proposal.

    This is independent of the pseudo-prior and is simply the usual log proposal probability ratio for the proposed model in the current state.
    """
    return log_proposal(proposed.model, proposed.state)


def _log_posterior_prob_ratio(
    log_posterior: MultiStateDensity, proposed: Sample, current: Sample
) -> float:
    """Calculate the log posterior probability ratio for a proposed model and state."""
    log_posterior_proposed = log_posterior(proposed.model, proposed.state)
    log_posterior_current = log_posterior(current.model, current.state)
    return log_posterior_proposed - log_posterior_current
