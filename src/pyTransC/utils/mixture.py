"""Module for fitting Gaussian mixtures to ensembles of samples in each state."""

import random
from typing import Callable

import numpy as np
from sklearn.mixture import GaussianMixture

from ..exceptions import InputError


def fit_mixture(
    n_states: int,
    ensemble,
    log_posterior_ens,
    verbose=False,
    **kwargs,  # Arguments for sklearn.mixture.GaussianMixture
):
    """
    Utility routine to fit a Gaussian mixture model to input ensemble for an approximate posterior distribution within each state.

    Makes use of sklearn.mixture.GaussianMixture to fit a mixture model.
    Mixture model is available internally for use by function 'run_ens_mcint()' to act as the pseudo prior in each state.

    Inputs:
    ensemble - list of floats [i,n[i],n_dims[i]],(i=0,...,n_states-1) : list containing an ensemble of samples in each state.
                                                                        i is the state;
                                                                        n[i] is the number of samples in state i;
                                                                        n_dims[i] is the number of unknowns in state i.
    log_posterior_ens -  list of floats, [i,n[i]].                  : log-posterior of samples in each state.
    verbose - bool                                                  : parameter to produce info to standard out (default = False).
    **kwargs - dict                                                 : kwargs passed to sklearn.mixture.GaussianMixture fitting routine

    Returns:
    log_pseudo_prior_ens - floats. : list of estimated log-pseudo_prior densities of samples in each state, format [i][j],(i=1,...,n_states;j=1,..., n_walkers*n_samples).


    Attributes:
    n_samples - n_states*int         : list of number of samples in each state.
    ensemble_per_state - floats    : list of posterior samples in each state, format [i][j][k],(i=1,...,n_states;j=1,..., n_walkers*n_samples;k=1,...,ndim[i]).
    log_posterior_ens - floats     : list of log-posterior densities of samples in each state, format [i][j],(i=1,...,n_states;j=1,..., n_walkers*n_samples).
    log_pseudo_prior_ens - floats. : list of estimated log-pseudo_prior densities of samples in each state, format [i][j],(i=1,...,n_states;j=1,..., n_walkers*n_samples).
    gm - functions                 : list of sklearn.mixture.GaussianMixture models fit to samples in each state
    n_comp - int                    : number of components in sklearn.mixture.GaussianMixture models fit to samples in each state
    run_fit - bool                 : bool to keep track of whether run_fitmixture has been called.

    """

    error = False  # check for errors in input shapes of arrays
    if len(ensemble) != len(log_posterior_ens):
        error = True
    for i in range(n_states):
        if len(ensemble[i]) != len(log_posterior_ens[i]):
            error = True
    if error:
        raise InputError(
            msg=" In function fit_mixture: Inconsistent shape of inputs\n"
            + "\n Input data for ensemble differs in shape for input data log_posterior_ens\n"
        )

    gaussian_mixtures = [GaussianMixture(**kwargs).fit(ens) for ens in ensemble]

    if verbose:
        for state, gm in enumerate(gaussian_mixtures):
            print(
                "State : ",
                state,
                "\n means",
                gm.means_,
                " covariances ",
                gm.covariances_,
                " weights ",
                gm.weights_,
            )

    return gaussian_mixtures


def log_pseudo_prior_from_mixtures(
    gaussian_mixtures: list[GaussianMixture],
    ensemble: list[np.ndarray],
    return_pseudo_prior_func=False,
) -> np.ndarray | tuple[Callable, np.ndarray]:
    """Utility routine to compute log pseudo-prior densities from fitted Gaussian mixtures."""

    log_pseudo = [gm.score_samples(ens) for gm, ens in zip(gaussian_mixtures, ensemble)]

    if not return_pseudo_prior_func:
        return log_pseudo

    def log_pseudo_prior(x, state, returndeviate=False, axisdeviate=False):
        """Multi-state log pseudo-prior density and deviate generator."""
        # get mixture model approximation for this state
        gmm = gaussian_mixtures[state]
        if returndeviate:
            if axisdeviate:  # force deviate to single component along random axis
                return _perturb(gmm, x)
            dev = gmm.sample()[0]
            logppx = gmm.score(dev)
            dev = dev[0]
            if not isinstance(dev, np.ndarray):
                dev = np.array([dev])  # deal with 1D case which returns a scalar
            logppx = gmm.score([dev])
            return logppx, dev
        return gmm.score([x])

    return log_pseudo_prior, log_pseudo


def _perturb(gmm, xc):
    pert = gmm.sample()[0] - gmm.means_
    mask = np.ones(pert.shape[1], dtype=bool)
    i = random.choice(np.arange(len(xc)))
    mask[i] = 0
    pert[0, mask] = 0.0
    y = pert + xc
    return gmm.score(y), y[0]
