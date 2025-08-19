"""Test the ensemble resampler."""

import numpy as np

from pytransc.samplers.ensemble_resampler import (
    EnsembleResamplerChain,
    Sample,
    _log_prob_sample,
    _propose_member_in_state,
    _propose_state,
    update_chain,
)


def test_update_chain_accept():
    """Test the update_chain function with an accepted sample."""
    # Create a sample with dummy data
    sample = Sample(member=0, state=1)

    # Create an EnsembleResamplerChain instance
    chain = EnsembleResamplerChain(n_states=3)

    # Update the chain with the sample
    update_chain(chain, sample, proposal_accepted=True)

    assert chain.member_chain == [0]
    assert chain.state_chain == [1]
    assert chain.n_accepted == 1
    assert chain.n_proposed == 1
    assert np.array_equal(chain.state_chain_tot, np.array([[0, 1, 0]]))


def test_update_chain_reject():
    """Test the update_chain function with a rejected sample."""
    # Create a sample with dummy data
    sample = Sample(member=0, state=1)

    # Create an EnsembleResamplerChain instance
    chain = EnsembleResamplerChain(n_states=3)

    # Update the chain with the sample, but reject it
    update_chain(chain, sample, proposal_accepted=False)

    # Note that the member chain and state chain should still be updated.  In an MCMC context, the Sample will be the previous sample, not the proposed one.
    assert chain.member_chain == [0]
    assert chain.state_chain == [1]
    assert chain.n_accepted == 0  # This is the key difference
    assert chain.n_proposed == 1
    assert np.array_equal(chain.state_chain_tot, np.array([[0, 1, 0]]))


def test__log_prob_sample():
    """Test the _log_prob_sample function."""
    # The log probability of a sample is the difference between the log posterior and the log pseudo-prior.

    sample = Sample(member=2, state=1)
    log_posterior_ensemble = np.array(
        [[-1.0, -2.0, -3.0], [-1.5, -2.5, -3.5], [-2.0, -3.0, -4.0]]
    )
    log_pseudo_prior_ensemble = np.array(
        [[-0.5, -1.5, -2.5], [-1.0, -2.0, -3.0], [-1.5, -2.5, -3.5]]
    )

    assert _log_prob_sample(
        sample, log_posterior_ensemble, log_pseudo_prior_ensemble
    ) == -3.5 - (-3.0)


def test__propose_member_in_state() -> None:
    """Test the _propose_member_in_state function.

    Checking that the proposed member index is within the bounds of the state ensemble size.
    """
    n_samples_in_state = 100_000
    for _ in range(1_000):
        assert 0 <= _propose_member_in_state(n_samples_in_state) < n_samples_in_state


def test__propose_state() -> None:
    """Test the _propose_state function.

    Checking that the proposed state index is within the bounds of the number of states and is NEVER the current state.
    """
    n_states = 5
    current_state = 2
    weights = [[1.0] * n_states] * n_states  # Uniform distribution for simplicity
    for _ in range(1_000):
        state = _propose_state(current_state, n_states, weights)
        assert 0 <= state < n_states
        assert state != current_state
