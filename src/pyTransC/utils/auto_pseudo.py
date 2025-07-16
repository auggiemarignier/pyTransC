"""Build an automatic pseudo-prior."""

import numpy as np
from scipy import stats

from ..exceptions import InputError
from ..samplers.per_state import run_mcmc_per_state
from .mixture import fit_mixture, log_pseudo_prior_from_mixtures


def build_auto_pseudo_prior(
    n_states: int,
    n_dims: list[int],
    pos,
    log_posterior,
    log_posterior_args=[],
    ensemble_per_state=None,
    log_posterior_ens=None,
    discard=0,
    thin=1,
    autothin=False,
    parallel=False,
    n_samples=1000,
    n_walkers=32,
    return_log_pseudo=False,
    progress=False,
    verbose=False,
    fitmeancov=False,
    **kwargs,  # Arguments for sklearn.mixture.GaussianMixture
):
    """
    Utility routine to build an automatic pseudo_prior function using a Gaussian mixture model approximation of posterior from small ensemble.

    This is intentionally a simple implementation of an automatic pseudo-prior approximation.
    This could serve as a template for building a more sophisticated approximation.

    Parameters
    ----------
    pos - list                  : list of starting points for internal mcmc generator in each state
                                    (Not used if ensemble_per_state and log_posterior_ens lists are provided)
    log_posterior : function    : Supplied function evaluating the log-posterior function
                                    up to a multiplicative constant, for each state. (Not used if ensemble_per_state and log_posterior_ens lists are provided)
                                    Calling sequence log_posterior(x,state,*log_posterior_args)
    log_posterior_args          : list, optional. Additional (optional) arguments required by user function log_posterior.
    ensemble_per_state - floats : Optional list of posterior samples in each state, format [i][j][k],(i=1,...,n_states;j=1,..., n_samples;k=1,...,ndim[i]).
    log_posterior_ens - floats  : Optional list of log-posterior densities of samples in each state, format [i][j],(i=1,...,n_states;j=1,..., n_samples).
    discard - int, or list      : number of output samples to discard (default = 0). (Parameter passed to emcee, also known as `burnin'.)
    thin - int, or list         : frequency of output samples in output chains to accept (default = 1, i.e. all) (Parameter passed to emcee.)
    autothin - bool             : if true ignores input thin value and instead thins the chain by the maximum auto_correlation time estimated (default = False).
    parallel - bool             : switch to make use of multiprocessing package to parallelize over walkers
    n_samples                    : int or list (of size n_states), optional
                                    Number of samples to draw from each state. The default is 1000.
    n_walkers                    : int or list (of size n_states), optional
                                    Number of walkers for mcmc sampling per state. The default is 32.
    fitmeancov                  : bool, switch to fit a simple mean and covariance using linalg. Covariance must be full rank. (Default=False)
    kwargs                      : dict, optional
                                    dictionary of arguments passed to sklean.mixture.GaussianMixture fitting routine

    Returns
    -------
    Auto_log_posterior : function to evaluate a normalized approximation to posterior samples in each state.
                            Calling sequence Auto_log_posterior(x,state,returndeviate=False);
                            where x is the input location in state `state'.
                            Returns logppx (returndeviate=False); or  logppx,dev (if returndeviate=True)
                            where logppx is the log-normalized pseudo prior evaluated at x, and
                            dev is a random deviate drawn from the pseudo-prior.

    """
    if not isinstance(n_walkers, list):
        n_walkers = [n_walkers] * n_states
    if not isinstance(n_samples, list):
        n_samples = [n_samples] * n_states

    if (ensemble_per_state is None) and (log_posterior_ens is not None):
        raise InputError(
            msg=" In function build_auto_pseudo_prior: Ensemble probabilities provided as argument without ensemble co-ordinates"
        )

    if (ensemble_per_state is not None) and (log_posterior_ens is None):
        raise InputError(
            msg=" In function build_auto_pseudo_prior: Ensemble co-ordinates provided as argument without ensemble probabilities"
        )

    if isinstance(ensemble_per_state, list) and isinstance(log_posterior_ens, list):
        print("We are using input ensembles")
    else:  # generate samples for fitting
        ensemble_per_state, log_posterior_ens = run_mcmc_per_state(
            n_states,
            n_dims,
            n_walkers,  # int or list containing number of walkers for each state
            n_samples,  # number of chain steps per walker
            pos,  # starting positions for walkers in each state
            log_posterior,  # log Likelihood x log_prior
            discard=discard,  # number of initial samples to discard along chain
            thin=thin,  # frequency of chain samples retained
            auto_thin=autothin,  # automatic thining of output ensemble by estimated correlation times
            parallel=parallel,
            log_posterior_args=log_posterior_args,  # log posterior additional arguments (optional)
            verbose=verbose,
            progress=progress,  # show progress bar for each state
        )

    if fitmeancov:  # we fit ensemble with a simple mean and covariance
        print(" We are fitting mean and covariance")
        rv_list = []
        log_pseudo = []
        for state_ensemble in ensemble_per_state:
            pseudo_covs_ = np.cov(state_ensemble.T)
            pseudo_means_ = np.mean(state_ensemble.T, axis=1)
            rv = stats.multivariate_normal(mean=pseudo_means_, cov=pseudo_covs_)
            rv_list.append([pseudo_means_, pseudo_covs_, rv])
            log_pseudo.append(rv.logpdf(state_ensemble))

        def log_pseudo_prior(x, state, size=1, returndeviate=False):
            """Multi-state log pseudo-prior density and deviate generator."""
            # get frozen stats.multivariate_normal object with correct mean and covariance
            rv = rv_list[state][2]
            if returndeviate:
                x = rv.rvs(size=size)
                logpseudo = rv.logpdf(x)  # evaluate log prior ignoring dimension prior
                return logpseudo, x
            return rv.logpdf(x)

    else:
        print(" We are fitting Gaussian Mixture Models")
        gaussian_mixtures = fit_mixture(
            n_states,
            ensemble_per_state,
            log_posterior_ens,
            verbose=verbose,
            **kwargs,
        )
        log_pseudo_prior, log_pseudo = log_pseudo_prior_from_mixtures(
            gaussian_mixtures, ensemble_per_state, return_pseudo_prior_func=True
        )

    if return_log_pseudo:
        # return both pseudo_prior function and log pseudo_prior values
        return log_pseudo_prior, log_pseudo

    return log_pseudo_prior
