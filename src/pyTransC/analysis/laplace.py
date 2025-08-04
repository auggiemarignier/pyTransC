"""Functions to perform Laplace approximation of the evidence."""

import numdifftools as nd
import numpy as np
from scipy.optimize import minimize

from ..utils.exceptions import InputError
from ..utils.types import MultiStateDensity


def run_laplace_evidence_approximation(
    n_states: int,
    n_dims: list[int],
    log_posterior: MultiStateDensity | None,
    map_models,
    ensemble_per_state=None,
    log_posterior_ens=None,
    verbose=False,
    optimize=False,
):
    """
    Function to perform Laplace integration for evidence approximation within each state, using either an input log-posterior function, or posterior ensembles.

    Parameters
    ----------
    log_posterior : function    : Supplied function evaluating the log-posterior function
                                    up to a multiplicative constant, for each state.
                                    (Not used if ensemble_per_state and log_posterior_ens lists are provided)
                                    Calling sequence log_posterior(x,state)
    map_models - floats         : List of MAP models in each state where Laplace approximation is evaluated.
                                    If optimize=True and a log_posterior() function is supplied, then
                                    scipy.minimize is used to find MAP models in each state using map_models as starting guesses.
    ensemble_per_state - floats : Optional list of posterior samples in each state, format [i][j][k],(i=1,...,n_samples;j=1,..., n_models;k=1,...,ndim[i]).
    log_posterior_ens - floats  : Optional list of log-posterior densities of samples in each state, format [i][j],(i=1,...,n_states;j=1,..., n_samples).
    optimize, bool              : Logical to decide whether to use optimization for MAP models (Only relevant if log_posterior()) function supplied.)

    Returns:
    -------
    laplace_log_marginal_likelihoods - n_states*float : list of log-evidence/marginal Likelihoods for each state.
    laplace_map_models_per_state - n_states*floats : list of updated MAP models for each state.m if optimize=True.
    laplace_map_log_posteriors - n_states*float : list of log-posteriors at MAP models for each state.
    laplace_hessians - n_states*NxN : list of negative inverse Hessians if posterior function supplied.

    Notes:
    Calculates Laplace approximations to evidence integrals in each state. This is equivalent to fitting a Gaussian about the MAP in model
    space. Here the Hessian in each state is calculated either by numerical differentiation tools (if a log_posterior function is supplied),
    or by taking the covariance of the given ensemble. The MAP model is similarly taken as the maximum of the log-posterior
    (if an ensemble is given), or a MAP model in each is estimated by optimization.
    Alternatively, if the list `map_models` is given then these are used within each state as MAP models.

    This implements equation 10.14 of Raftery (1996) evaluating Laplace's approximation to Bayesian evidence.

    Raftery, A.E. (1996) Hypothesis Testing and Model Selection. In: Gilks, W., Richardson, S. and Speigelhalter, D.J., Eds.,
    Markov Chain Monte Carlo in Practice, Chapman and Hall, 163-187.
    """

    if (ensemble_per_state is None) and (log_posterior_ens is not None):
        raise InputError(
            msg=" In function run_laplace_evidence_approximation: Ensemble probabilities provided as argument without ensemble co-ordinates"
        )

    if (ensemble_per_state is not None) and (log_posterior_ens is None):
        raise InputError(
            msg=" In function run_laplace_evidence_approximation: Ensemble co-ordinates provided as argument without ensemble probabilities"
        )

    if isinstance(ensemble_per_state, list) and isinstance(
        log_posterior_ens, list
    ):  # we use input ensemble
        if verbose:
            print(
                "run_laplace_evidence_approximation: We are using input ensembles rather than input log_posterior function"
            )

        hessians, map_models, map_log_posteriors, log_marginal_likelihoods = (
            _from_ensemble(
                n_dims,
                ensemble_per_state,
                log_posterior_ens,
            )
        )
    elif log_posterior is not None:
        # we are using the supplied log_posterior() function so need Hessian and MAp model
        if verbose:
            if optimize:
                print(
                    "run_laplace_evidence_approximation: We are using input log_posterior function with optimization for MAP models"
                )
            else:
                print(
                    "run_laplace_evidence_approximation: We are using input log_posterior function with provided MAP models"
                )

        hessians, map_models, map_log_posteriors, log_marginal_likelihoods = (
            _from_log_posterior(
                n_states,
                n_dims,
                log_posterior,
                map_models,
                optimize=optimize,
            )
        )
    else:
        raise InputError(
            msg="In function run_laplace_evidence_approximation: Either ensemble_per_state or log_posterior must be provided."
        )

    return (
        hessians,
        map_models,
        map_log_posteriors,
        log_marginal_likelihoods,
    )


def _from_ensemble(
    n_dims: list[int],
    ensemble_per_state: list[np.ndarray],
    log_posterior_ens: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[float]]:
    """Calculate Laplace approximation from ensemble of models and log-posterior values."""
    # we fit a mean and covariance to ensembles in each state
    hessians = []
    map_models = []
    map_log_posteriors = []
    log_marginal_likelihoods = []
    for state in range(len(ensemble_per_state)):  # loop over states
        covar = np.cov(ensemble_per_state[state].T)
        hessians.append(covar)
        j = np.argmax(log_posterior_ens[state])  # get map model index
        map_models.append(ensemble_per_state[state][j])  # get map model
        map_log_posteriors.append(log_posterior_ens[state][j])
        p1 = (n_dims[state] / 2.0) * np.log(2 * np.pi)
        p3 = log_posterior_ens[state][j]
        # get determinant of negative inverse of covariance (-H) (NB determinant sign depends on dimension being odd or even)
        if n_dims[state] % 2:  # dimension is odd number
            if n_dims[state] == 1:
                p2 = 0.5 * np.log(covar)
            else:
                sign, logabsdet = np.linalg.slogdet(covar)
                p2 = 0.5 * sign * logabsdet
        else:
            sign, logabsdet = np.linalg.slogdet(-covar)
            p2 = 0.5 * sign * logabsdet
        log_marginal_likelihoods.append(p1 + p2 + p3)

    return hessians, map_models, map_log_posteriors, log_marginal_likelihoods


def _from_log_posterior(
    n_states: int,
    n_dims: list[int],
    log_posterior: MultiStateDensity,
    map_models: list[np.ndarray],
    optimize: bool = False,
):
    """Calculate Laplace approximation from log-posterior function."""
    hessians = []
    _map_models = []
    map_log_posteriors = []
    log_marginal_likelihoods = []
    for state in range(n_states):
        fun = lambda x: log_posterior(x, state)
        if optimize:
            fun2 = lambda x: -log_posterior(x, state)
            soln = minimize(fun2, map_models[state])
            map_model = soln.x
        else:
            map_model = np.array(map_models[state])
        dfun = nd.Hessian(fun)
        hessians.append(dfun(map_model))
        _map_models.append(map_model)
        map_log_posteriors.append(fun(map_model))

        p1 = (n_dims[state] / 2.0) * np.log(2 * np.pi)
        p3 = fun(map_model)
        # print(det)
        if n_dims[state] == 1:
            p2 = -0.5 * np.log(-dfun(map_model))
        else:
            # det = np.linalg.det(-dfun(map_model))
            sign, logabsdet = np.linalg.slogdet(-dfun(map_model))
            # print(det,np.log(det),sign,logabsdet)
            p2 = -0.5 * sign * logabsdet

        log_marginal_likelihoods.append(p1 + p2 + p3)

    return hessians, _map_models, map_log_posteriors, log_marginal_likelihoods
