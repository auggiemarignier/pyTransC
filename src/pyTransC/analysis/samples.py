"""Functions for obtaining samples from an ensemble."""

from collections.abc import Sequence

import numpy as np

from ..utils.types import MultiStateMultiWalkerResult, StateOrderedEnsemble


def get_transc_samples(
    chains: MultiStateMultiWalkerResult,
    discard: int = 0,
    thin: int = 1,
) -> StateOrderedEnsemble:
    """Restructures MCMC chains into state-wise ensembles.

    Returns:
    transc_ensemble : list of numpy arrays
        A list of ensembles, where each ensemble corresponds to a state and contains the models sampled in that state.
        All the models in a state are concatenated into a single numpy array.
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
    """Given a list of ensembles and their relative marginal likelihoods, sample models from the ensembles.

    Parameters
    ----------
    ensemble_per_state : StateOrderedEnsemble
        A list of ensembles, where each ensemble corresponds to a state and contains the models sampled in that state in a 2D numpy array.
    relative_marginal_likelihoods : Sequence[float]
        The relative marginal likelihoods for each state, used for sampling.
    ntd_samples : int
        The number of samples to draw from the ensembles.

    Returns
    -------
    StateOrderedEnsemble
        A new ensemble of models sampled from the input ensembles.

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
