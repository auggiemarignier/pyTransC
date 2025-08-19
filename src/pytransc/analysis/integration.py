"""Functions to estimate the marginal likelihood using Monte Carlo integration."""

import numpy as np

from ..utils.types import FloatArray


def run_ens_mcint(  #  Marginal Likelihoods from Monte Carlo integration
    n_states: int,
    log_posterior_ens: list[FloatArray],
    log_pseudo_prior_ens: list[FloatArray],
):
    """
    Utility routine to perform MCMC sampler over independent states using numerical integration.

    This routine is a faster alternate to running a Markov chain across the ensembles, which is carried out by run_is_ensemble_resampler.
    Calculates relative evidence of each state using previously computed ensembles in each state.

    Inputs:
    log_posterior_ens -  list of floats, [i,j], (i=1,...,n_states;j=1,...,n_samples[i])    : log-posterior of ensembles in each state, where n_samples[i] is the number of samples in the ith state.
    log_pseudo_prior_ens -  list of floats, [i,j], (i=1,...,n_states;j=1,...,n_samples[i]) : log-pseudo prior of samples in each state, where n_samples[i] is the number of samples in the ith state.

    Returns:
    relative_marginal_likelihoods - n_states*float : list of relative evidence/marginal Likelihoods for each state.
    ens_mc_samples - floats.                      : list of density ratios in Monte Carlo integration. format [i][j],[i=1,...,n_states;j=1,...,n_samples[i]).

    """

    ens_r = []
    ens_mc_samples = []
    factor = np.min(
        [
            np.min(log_posterior_ens[i] - log_pseudo_prior_ens[i])
            for i in range(n_states)
        ]
    )
    for state in range(n_states):
        ratio_state = np.exp(
            log_posterior_ens[state] - log_pseudo_prior_ens[state] - factor
        )
        ens_mc_samples.append(ratio_state)
        ens_r.append(np.mean(ratio_state))

    tot = np.sum(ens_r)
    relative_marginal_likelihoods = ens_r / tot
    for state in range(n_states):
        ens_mc_samples[state] /= tot

    return relative_marginal_likelihoods, ens_mc_samples
