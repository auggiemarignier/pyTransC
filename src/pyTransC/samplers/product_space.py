"""Product-Space Sampling for TransC."""

import random
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from emcee import EnsembleSampler

from ..utils.types import MultiStateDensity
from ._emcee import perform_sampling_with_emcee


@dataclass
class ProductSpace:
    """Define the product space for TransC sampling."""

    n_dims: list[int]

    @property
    def n_states(self) -> int:
        """Number of states in the product space."""
        return len(self.n_dims)

    @property
    def total_n_dim(self) -> int:
        """Total number of dimensions in the product space.

        Sum of dimensions of all states plus one for the state index.
        """
        return sum(self.n_dims) + 1

    def model_vectors2product_space(
        self,
        state: int,
        model_vectors: list[np.ndarray],
    ) -> np.ndarray:
        """Convert a list of model vectors to product space format with the selected state."""
        if not (0 <= state < self.n_states):
            raise ValueError(
                f"State index {state} is out of bounds for product space with {self.n_states} states."
            )
        if any(
            mv.size != expected_nd
            for mv, expected_nd in zip(model_vectors, self.n_dims)
        ):
            raise ValueError(
                "Model vectors do not match the dimensions of the product space."
            )
        return np.concatenate([[state], *model_vectors])

    def product_space2model_vectors(
        self,
        product_space_vector: np.ndarray,
    ) -> tuple[int, list[np.ndarray]]:
        """Convert a product space vector back to a list of model vectors."""
        if product_space_vector.size != self.total_n_dim:
            raise ValueError(
                f"Product space vector size {product_space_vector.size} does not match total dimensions {self.total_n_dim}."
            )
        state = self._clip_state_index(product_space_vector[0])

        k = 1
        model_vectors = []
        for n_dim in self.n_dims:
            model_vectors.append(product_space_vector[k : k + n_dim])
            k += n_dim
        return state, model_vectors

    def _clip_state_index(self, state: float) -> int:
        """Clip the state index to ensure it is within valid bounds.

        The product space vector will in principle always have dtype=float, so we need to ensure the state index is an integer.
        """
        # np.rint rounds to the nearest integer. In the unlikely case of a tie, it rounds to the nearest even integer e.g. 2.5 -> 2.0 (round down) 3.5 -> 4.0 (round up)
        # clip to valid state index range
        return int(np.clip(np.rint(state), 0, self.n_states - 1))


def _get_states_from_emcee_sampler(
    product_space_sampler: EnsembleSampler,
) -> list[list[int]]:
    """Get the states from the emcee sampler."""
    chain = product_space_sampler.get_chain()
    if chain is None or chain.size == 0:
        raise ValueError("The sampler chain is empty or not initialized.")
    state_chain = chain[:, :, 0]
    return np.rint(state_chain).astype("int").tolist()


def _get_models_from_emcee_sampler(
    product_space_sampler: EnsembleSampler,
    n_dims: list[int],
) -> list[list[np.ndarray]]:
    """Get the model vectors from the emcee sampler."""
    samples = product_space_sampler.get_chain()
    if samples is None or samples.size == 0:
        raise ValueError("The sampler chain is empty or not initialized.")

    ind1 = np.cumsum(n_dims) + 1
    ind0 = np.append(np.array([1]), ind1)

    state_chain = np.rint(samples[:, :, 0]).astype("int")
    n_steps, n_walkers = np.shape(state_chain)
    model_chain = []
    for i in range(n_steps):
        m = []
        for j in range(n_walkers):
            m.append(
                samples[
                    i,
                    j,
                    ind0[state_chain[i, j]] : ind1[state_chain[i, j]],
                ]
            )
        model_chain.append(m)

    return model_chain


@dataclass(init=False)
class MultiWalkerProductSpaceChain:
    """Data class to hold the results of the product space sampler.

    This will basically wrap emcee's EnsembleSampler and provide a more convenient interface for accessing the results that is more consistent with the rest of pyTransC.

    This class can only be initialised using the `from_emcee` class method, which will extract the necessary information from an `emcee.EnsembleSampler` instance.
    """

    n_states: int
    model_chain: list[list[np.ndarray]] = field(default_factory=list)
    state_chain: list[list[int]] = field(default_factory=list)

    @classmethod
    def from_emcee(
        cls,
        emcee_sampler: EnsembleSampler,
        n_dims: list[int],
    ) -> "MultiWalkerProductSpaceChain":
        """Create a ProductSpaceChain from an emcee EnsembleSampler."""
        state_chain = _get_states_from_emcee_sampler(emcee_sampler)
        model_chain = _get_models_from_emcee_sampler(emcee_sampler, n_dims)
        n_states = len(n_dims)
        obj = cls.__new__(cls)  # Bypass __init__, since we use init=False
        obj.n_states = n_states
        obj.model_chain = model_chain
        obj.state_chain = state_chain
        return obj

    @property
    def n_walkers(self) -> int:
        """Number of walkers in the chain."""
        if not self.model_chain:
            return 0
        return len(self.state_chain[0])

    @property
    def n_steps(self) -> int:
        """Number of steps in the chain."""
        return len(self.state_chain)

    @property
    def state_chain_tot(self) -> np.ndarray:
        """Cumulative state visit counts for each walker over time.

        Returns:
            np.ndarray: Array of shape (n_steps, n_walkers, n_states) where each entry [i, j, k]
                        is the number of times walker j has visited state k up to step i.
        """
        n_steps = self.n_steps
        n_walkers = self.n_walkers
        n_states = self.n_states
        counts = np.zeros((n_steps, n_walkers, n_states), dtype=int)

        for w in range(n_walkers):
            visits = np.zeros(n_states, dtype=int)
            for t in range(n_steps):
                state = self.state_chain[t][w]
                visits[state] += 1
                counts[t, w] = visits.copy()
        return counts


def run_product_space_sampler(
    product_space: ProductSpace,
    n_walkers: int,
    n_steps: int,
    start_positions: list[np.ndarray],
    start_states: list[int],
    log_posterior: MultiStateDensity,
    log_pseudo_prior: MultiStateDensity,
    seed: int | None = 61254557,
    parallel: bool = False,
    n_processors: int = 1,
    progress: bool = False,
    **kwargs,
) -> MultiWalkerProductSpaceChain:
    """
    MCMC sampler over independent states using emcee fixed dimension sampler over trans-C product space.

    Inputs:
    n_walkers - int               : number of random walkers used by product_space sampler.
    n_steps - int                 : number of steps required per walker.
    pos - n_walkers*n_dims*float   : list of starting locations of markov chains in each state.
    pos_state - n_walkers*int     : list of starting states of markov chains in each state.
    log_posterior()              : user supplied function to evaluate the log-posterior density for the ith state at location x.
                                   calling sequence log_posterior(x,i)
    log_pseudo_prior()           : user supplied function to evaluate the log-pseudo-prior density for the ith state at location x.
                                   calling sequence log_posterior(x,i).
                                   NB: must be normalized over respective state spaces.
    prob_state - float           : probability of proposal a state change per step of Markov chain (otherwise a parameter change within current state is proposed)
    seed - int                   : random number seed
    parallel - bool              : switch to make use of multiprocessing package to parallelize over walkers
    n_processors - int            : number of processors to distribute work across (if parallel=True, else ignored). Default = multiprocessing.cpu_count()/1 if parallel = True/False.
    progress - bool              : switch to report progress to standard out.
    kwargs - dict                : dictionary of optional arguments passed to emcee.

    """

    random.seed(seed)

    if progress:
        print("\nRunning product space trans-C sampler")
        print("\nNumber of walkers               : ", n_walkers)
        print("Number of states being sampled  : ", product_space.n_states)
        print("Dimensions of each state        : ", product_space.n_dims)

    pos_ps = _get_initial_product_space_positions(
        n_walkers, start_states, start_positions, product_space
    )

    log_func = partial(
        product_space_log_prob,
        product_space=product_space,
        log_posterior=log_posterior,
        log_pseudo_prior=log_pseudo_prior,
    )

    sampler = perform_sampling_with_emcee(
        log_prob_func=log_func,
        n_walkers=n_walkers,
        n_steps=n_steps,
        initial_state=pos_ps,
        parallel=parallel,
        n_processors=n_processors,
        progress=progress,
        **kwargs,
    )

    return MultiWalkerProductSpaceChain.from_emcee(sampler, product_space.n_dims)


def product_space_log_prob(
    x: np.ndarray,
    product_space: ProductSpace,
    log_posterior: MultiStateDensity,
    log_pseudo_prior: MultiStateDensity,
):
    """
    Calculate the combined target density for product space vector i.e. sum of log posterior + log pseudo prior density of all other states.

    here input vector is in product space format.

    Inputs:
    x - float array or list : trans-C vectors in product space format. (length = n_walkers*(1 + sum n_dim[i], i=1,...,n_states))
    log_posterior()              : user supplied function to evaluate the log-posterior density for the ith state at location x.
                                    calling sequence log_posterior(x,i)
    log_pseudo_prior()           : user supplied function to evaluate the log-pseudo-prior density for the ith state at location x.
                                    calling sequence log_posterior(x,i).
                                    NB: must be normalized over respective state spaces.


    Returns:
    x - float array or list : trans-C vectors in product space format. (length sum ndim[i], i=1,...,n_states)

    """
    if x[0] < -0.5 or x[0] >= product_space.n_states - 0.5:
        return -np.inf

    state, m = product_space.product_space2model_vectors(x)
    log_prob = log_posterior(m[state], state)
    for i in range(product_space.n_states):
        if i != state:
            new = log_pseudo_prior(m[i], i)
            log_prob += new
    return log_prob


def _get_initial_product_space_positions(
    n_walkers: int,
    start_states: list[int],
    start_positions: list[np.ndarray],
    product_space: ProductSpace,
) -> np.ndarray:
    """Get start positions and states into product space format.

    Args:
        n_walkers (int): Number of walkers.
        start_states (list[int]): List of starting states for each walker.
        start_positions (list[np.ndarray]): List of starting positions for each walker.  Each list element is an array of model vectors for each state.  The indexing is start_positions[state][walker], so start_positions[0].shape = (n_walkers, n_dims[0]), start_positions[1].shape = (n_walkers, n_dims[1]), etc.
        product_space (ProductSpace): The product space object.

    Returns:
        np.ndarray: The initial positions in product space format.
    """
    pos_ps = np.zeros((n_walkers, product_space.total_n_dim))
    for walker, state in enumerate(start_states):
        model_vectors = [start[walker] for start in start_positions]
        pos_ps[walker] = product_space.model_vectors2product_space(state, model_vectors)
    return pos_ps
