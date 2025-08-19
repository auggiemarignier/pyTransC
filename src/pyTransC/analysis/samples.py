"""Functions for extracting and resampling from trans-dimensional MCMC results.

This module provides utilities for post-processing trans-conceptual MCMC chains,
including extracting state-specific samples and resampling according to marginal
likelihood estimates.
"""

from collections.abc import Sequence

import numpy as np

from ..utils.types import MultiStateMultiWalkerResult, StateOrderedEnsemble


def get_transc_samples(
    chains: MultiStateMultiWalkerResult,
    discard: int = 0,
    thin: int = 1,
) -> StateOrderedEnsemble:
    """Extract state-specific sample ensembles from trans-conceptual MCMC chains.

    This function restructures the output of trans-conceptual MCMC samplers
    into separate ensembles for each state, collecting all parameter samples
    that were accepted in each state across all walkers.

    Parameters
    ----------
    chains : MultiStateMultiWalkerResult
        Results from a trans-conceptual MCMC sampler containing state and
        model chains for multiple walkers.
    discard : int, optional
        Number of initial samples to discard as burn-in. Default is 0.
    thin : int, optional
        Thinning factor - use every `thin`-th sample. Default is 1 (no thinning).

    Returns
    -------
    StateOrderedEnsemble
        List of parameter ensembles, one per state. Each ensemble is a 2D array
        with shape (n_samples_in_state, n_parameters_in_state) containing all
        parameter vectors that were sampled in that state.

    Notes
    -----
    The resulting ensembles can be used for:
    - Posterior analysis within individual states
    - Constructing pseudo-priors for subsequent sampling
    - Model comparison and visualization

    Examples
    --------
    >>> # Extract samples from trans-dimensional chains
    >>> ensembles = get_transc_samples(
    ...     chains=sampler_results,
    ...     discard=1000,  # Discard first 1000 samples
    ...     thin=5        # Keep every 5th sample
    ... )
    >>> print(f"State 0 ensemble shape: {ensembles[0].shape}")
    """
    models_chain = [row[discard::thin] for row in chains.model_chain]  # stride the list
    states_chain = chains.state_chain[:, discard::thin]  # stride the array

    state_models = {state: [] for state in range(chains.n_states)}
    for walker_states, walker_models in zip(states_chain, models_chain):
        for state, model in zip(walker_states, walker_models):
            state_models[state].append(model)

    return [np.array(models) for models in state_models.values()]


def resample_ensembles(
    ensemble_per_state: StateOrderedEnsemble,
    relative_marginal_likelihoods: Sequence[float],
    ntd_samples: int = 1000,
) -> StateOrderedEnsemble:
    """Resample from state ensembles according to marginal likelihood weights.

    This function generates a new set of samples by resampling from existing
    state-specific ensembles, where the probability of selecting each state
    is proportional to its estimated marginal likelihood.

    Parameters
    ----------
    ensemble_per_state : StateOrderedEnsemble
        List of parameter ensembles, one per state. Each ensemble should be
        a 2D array with shape (n_samples_in_state, n_parameters_in_state).
    relative_marginal_likelihoods : Sequence[float]
        Relative marginal likelihood estimates for each state.
    ntd_samples : int, optional
        Total number of samples to draw from the resampled ensemble.
        Default is 1000.

    Returns
    -------
    StateOrderedEnsemble
        List of resampled ensembles, one per state, containing samples drawn
        proportionally to the marginal likelihood weights.

    Notes
    -----
    This resampling procedure provides a way to obtain samples from the
    trans-dimensional posterior where each state is represented according
    to its relative support in the data.

    Examples
    --------
    >>> # Resample according to estimated marginal likelihoods
    >>> resampled = resample_ensembles(
    ...     ensemble_per_state=original_ensembles,
    ...     relative_marginal_likelihoods=[0.6, 0.3, 0.1],
    ...     ntd_samples=5000
    ... )
    """
    rng = np.random.default_rng()
    n_states = len(relative_marginal_likelihoods)
    if len(ensemble_per_state) != n_states:
        raise ValueError(
            f"Number of ensembles ({len(ensemble_per_state)}) does not match the number of provided marginal likelihoods ({len(relative_marginal_likelihoods)})."
        )
    n_samples = [len(ensemble) for ensemble in ensemble_per_state]

    states_chain = rng.choice(
        n_states, size=ntd_samples, p=relative_marginal_likelihoods
    )

    model_chain = ntd_samples * [None]
    for i, state in enumerate(states_chain):
        # randomly select models from input state ensembles using evidence weights
        j = rng.choice(n_samples[state])
        model_chain[i] = ensemble_per_state[state][j]

    state_models = {state: [] for state in range(n_states)}
    for state, model in zip(states_chain, model_chain):
        state_models[state].append(model)
    return [np.array(models) for models in state_models.values()]
