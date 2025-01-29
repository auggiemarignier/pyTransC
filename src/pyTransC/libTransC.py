# -*- coding: utf-8 -*-
"""
    Trans conceptual McMC sampler class. 
    
    A class that is used to perform a Metropolis random walk across a Trans-C model comprising of independent states.
    Calculates relative Marginal Likelihoods/evidences between states and/or an ensemble of Trans-C samples.
    Each state may have arbitrary dimension, model parameter definition and Likelihood function.
    
    For a description of Trans-C and its relation to Trans-D sampling see 
    
    https://essopenarchive.org/users/841079/articles/1231492-trans-conceptual-sampling-bayesian-inference-with-competing-assumptions
"""
#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.mixture import GaussianMixture
import emcee
import random
import numpy as np
from tqdm import tqdm
import multiprocessing
import warnings
import os
from functools import partial
import scipy.stats as stats
from scipy.optimize import minimize
import numdifftools as nd

multiprocessing.set_start_method("fork")

os.environ["OMP_NUM_THREADS"] = (
    "1"  # turn off automatic parallelisation so that we can use emcee parallelization
)


class Error(Exception):
    """Base class for other exceptions"""

    pass


class Inputerror(Exception):
    """Raised when necessary inputs are missing"""

    def __init__(self, msg=""):
        super().__init__(msg)


class TransC_Sampler(object):  # Independent state MCMC parameter class
    """
    Trans-C McMC sampler class.

    A class that is used to perform a Metropolis random walk across independent states.
    Calculates relative Marginal Likelihoods/evidences between states and/or an ensemble of trans-D samples.
    Each state may have arbitrary dimension, model parameter definition and Likelihood function.

    Four alternate algorithms are available:
        1) Trans-C samping across states in a combined fixed dimension product space.
           Makes use of a third party fixed dimension sampler (emcee is used as a default).
           May be used in serial or parallel modes.
           Advantages - Simple to use; makes use of adaptive/parallel third party sampler;
           Disadvantages - sampling is performed in a higher dimensional space. (=1+sum of dimensions of each state)
           Implemented with function `run_product_space_sampler()`. Creates self.alg = 'TransC-product-space.

        2) Trans-C sampling between and within states, with dimension jumps observing the pseudo-prior detailed balanced condition.
           May be used in serial or parallel modes.
           Advantages - User may choose Pseudo-prior and within state proposal densities; detailed balance condition easily combined with users existing MCMC algorithm;
           Disdavantage - uses a simple Metropolis-Hastings sampler for both within and between state steps.
           Implemented with function `run_is_pseudo_sampler()`. Creates self.alg = 'TransC-pseudo-sampler'.

        3) Trans-C sampling across previously calculated posterior ensembles in each state.
           User may supply posterior ensembles of any size (one per state) with log-posterior and log-Pseudo-prior densities (normalized) calculated at sample locations.
           Alternatively, ensembles and log-Pseudo-prior values may be igenerated internally with class functions 'run_mcmc_per_state()' and `run_fitmixture()'.
           May be used in serial or parallel modes.
           Advantages - makes use of previously calculated posterior ensembles calculated offline, Automatic estimation of Pseudo-Prior;
           Disadvantages - depends on quality of users/generated ensemble.
           Implemented with function `run_is_ensemble_sampler()`. Creates self.alg = 'TransC-ensemble-sampler'.

        4) Relative evidence/marginal Likelihood calculation using Monte Carlo Integration over each state.
           User may supply posterior ensembles of any size (one per state) with log-posterior and log-Pseudo-prior densities (normalized) calculated at sample locations.
           Alternatively, ensembles and log-Pseudo-prior values may be igenerated internally with class functions 'run_mcmc_per_state()' and `run_fitmixture()'.
           Advantages - makes use of previously calculated posterior ensembles; Disadvantages - depends on quality of users/generated ensemble; no Markov chain output
           to inspect. Implemented with function `run_ens_mcint()`. Creates self.alg = 'TransC-integration'.

    In all cases log densities are required of a NORMALIZED pseudo-prior over each state, and for (2) also a function to generate pseudo-prior deviates.
    This may be user supplied or internally calculated with a mixture model.
    Performance in all cases depends on the closeness of the normalized pseduo-prior PDF to posterior PDF multiplied by an arbitrary constant.

    """

    def __init__(self, nstates, ndims):
        """

        Create a TransC-sampler object.

        Inputs:
        nstates - int       : number of independent states.
        ndims - nstates*int : list of number of unknowns in each state.

        Attributes defined:
        nstates - int        : number of independent states.
        ndims - nstates*int  : list of number of unknowns in each state.
        kmax - int           : dimension of product space, or sum of dimensions of all states.
        run_per_state - bool : default bool (False) to keep track of whether run_mcmc_per_state has been called.
        run_fit - bool.      : default bool (False) to keep track of whether run_fitmixture has been called.

        """
        self.nstates = nstates  # number of independent states
        self.ndims = ndims  # number of parameters in each state
        self.run_per_state = False  # default bool for method mcmc
        self.run_fit = False  # default bool for function to fit a mixture model to an ensemble (used by method mcmc)
        self.ps_ndim = np.sum(
            self.ndims
        )  # number of actual model parameters in fixed dimensional product space (dimension is +1 with k)

    def run_mcmc_per_state(  # samples each state with the emcee sampler
        self,
        nwalkers,
        nsteps,
        pos,
        log_posterior,
        log_posterior_args=[],
        discard=0,
        thin=1,
        autothin=False,
        seed=61254557,
        parallel=False,
        nprocessors=1,
        progress=True,
        skip_initial_state_check=False,
        io=False,
        verbose=False,
        **kwargs,
    ):
        """
        Utility routine to run an MCMC sampler independently within each state.
        Creates a set of ensembles of posterior samples for each state.
        Makes use of emcee sampler for posterior sampling.

        This function is for convenience only. Its creates an ensemble of posterior samples within each state which
            - can serve as the input to run_ensemble_sampler()
            - can be used to build an approximate normalized pseudo_prior with build_auto_pseudo_prior().
        Alternatively, the user could supply their own ensembles in each state for these purposes,
        or directly provide their on own log_pseudo_prior function as required.

        Inputs:
        nwalkers - int, or list     : number of random walkers used by emcee sampler.
        nsteps - int                : number of steps required per walker.
        pos - nwalkers*ndims*float  : list of starting points of markov chains in each state.
        log_posterior - func        : user supplied function to evaluate the log-posterior density for the ith state at location x.
                                      calling sequence log_posterior(x,i,*log_posterior_args)
        log_posterior_args - list   : user defined list of additional arguments passed to log_posterior function (optional).
        discard - int, or list      : number of output samples to discard (default = 0). (Parameter passed to emcee, also known as `burnin'.)
        thin - int, or list         : frequency of output samples in output chains to accept (default = 1, i.e. all) (Parameter passed to emcee.)
        autothin - bool             : if True, ignores input thin value and instead thins the chain by the maximum auto_correlation time estimated (default = False).
        seed - int                  : random number seed
        parallel - bool             : switch to make use of multiprocessing package to parallelize over walkers
        nprocessors - int           : number of processors to distribute work across (if parallel=True, else ignored).
                                      Default = multiprocessing.cpu_count() if parallel = True, else 1 if False.
        progress - bool             : switch to report progress of emcee to standard out. (Parameter passed to emcee.)
        kwargs - dict               : dictionary of optional control parameters passed to emcee package to determine sampling behaviour.

        Returns:
        log_posterior_ens - floats. : list of log-posterior densities of samples in each state, format [i][j],(i=1,...,nstates;j=1,..., nwalkers*nsamples).
        ensemble_per_state - floats : list of posterior samples in each state, format [i][j][k],(i=1,...,nstates;j=1,..., nwalkers*nsamples;k=1,...,ndim[i]).


        Attributes defined:
        nwalkers - int              : number of random walkers per state
        ensemble_per_state - floats : list of posterior samples in each state, format [i][j][k],(i=1,...,nstates;j=1,..., nwalkers*nsamples;k=1,...,ndim[i]).
        nsamples - nstates*int      : list of number of samples in each state.
        log_posterior_ens - floats. : list of log-posterior densities of samples in each state, format [i][j],(i=1,...,nstates;j=1,..., nwalkers*nsamples).
        run_per_state - bool.       : bool to keep track of whether run_mcmc_per_state has been called.

        """

        random.seed(seed)
        if type(nwalkers) != list:
            nwalkers = [nwalkers for i in range(self.nstates)]
        if type(discard) != list:
            discard = [discard for i in range(self.nstates)]
        if type(thin) != list:
            thin = [thin for i in range(self.nstates)]
        if type(nsteps) != list:
            nsteps = [nsteps for i in range(self.nstates)]
        if autothin:
            thin = [
                1 for i in range(self.nstates)
            ]  # ignore thining factor because we are post thining by the auto-correlation times
        if type(parallel) == bool:
            parallel = [parallel for i in range(self.nstates)]

        self.nwalkers_per_state = nwalkers
        self.autothin = autothin
        samplers = []

        (samples, log_posterior_ens, auto_correlation) = ([], [], [])
        if progress:
            print("\nRunning within-state sampler separately on each state")
            print("\nNumber of walkers               : ", self.nwalkers_per_state)
            print("\nNumber of states being sampled: ", self.nstates)
            print("Dimensions of each state: ", self.ndims)

        for i in range(self.nstates):  # loop over states

            logfunc = partial(
                self._myfunc, log_posterior=log_posterior, args=[i, *log_posterior_args]
            )
            if parallel[i]:
                if nprocessors == 1:
                    nprocessors = (
                        multiprocessing.cpu_count()
                    )  # set number of processors

                with multiprocessing.Pool(processes=nprocessors) as pool:
                    sampler = emcee.EnsembleSampler(  # instantiate emcee class
                        nwalkers[i], self.ndims[i], logfunc, pool=pool, **kwargs
                    )
                    sampler.run_mcmc(
                        pos[i],
                        nsteps[i],
                        progress=progress,
                        skip_initial_state_check=skip_initial_state_check,
                    )  # run sampler
            else:
                sampler = emcee.EnsembleSampler(
                    nwalkers[i], self.ndims[i], logfunc, **kwargs
                )
                sampler.run_mcmc(
                    pos[i],
                    nsteps[i],
                    progress=progress,
                    skip_initial_state_check=skip_initial_state_check,
                )  # run sampler in current state
            samples.append(
                sampler.get_chain(discard=discard[i], thin=thin[i], flat=True)
            )  # collect state ensemble
            log_posterior_ens.append(
                sampler.get_log_prob(discard=discard[i], thin=thin[i], flat=True)
            )  # collect state log_posterior values

            if autothin:
                if verbose:
                    print("Performing auto thinning of ensemble...")
                auto_correlation.append([sampler.get_autocorr_time(tol=0)])
                if verbose:
                    print(
                        "Auto thinning factor calculated = ",
                        int(np.ceil(np.max(auto_correlation[i]))),
                    )
            samplers.append(sampler)

        self.nprocessors = nprocessors

        if autothin:
            # we now thin the chains using the maximum auto_correlation function for each state to get independent samples for fitting
            # emcee manual suggests tau = sampler.get_autocorr_time(); burnin = int(2 * np.max(tau)); thin = int(0.5 * np.min(tau))
            samples_auto, log_posterior_ens_auto = [], []
            for i in range(self.nstates):
                # thin = int(np.ceil(np.max(auto_correlation[i])))
                thin = int(
                    np.ceil(0.5 * np.min(auto_correlation[i]))
                )  # use emcee suggestion
                burnin = int(
                    np.ceil(2.0 * np.max(auto_correlation[i]))
                )  # use emcee suggestion
                samples_auto.append(samples[i][burnin::thin])
                log_posterior_ens_auto.append(log_posterior_ens[i][burnin::thin])
            samples = samples_auto
            log_posterior_ens = log_posterior_ens_auto
            self.autocorr_times_within_each_state = auto_correlation

        self.ensemble_per_state = samples
        self.log_posterior_ens = log_posterior_ens
        self.run_per_state = True
        self.samplers = samplers
        s = []
        for i in range(self.nstates):
            s.append(len(samples[i]))
        self.nsamples = s  # record number of models in each ensemble

        return samples, log_posterior_ens

    def auto_thin_chains(
        self, samples, log_posterior_ens, verbose=False
    ):  # thin the chains using the maximum auto_correlation function for each state to get independent samples
        """Function to calculate and thin provided samples and log_posterior valuesby their respective auto-correlation times"""
        samples_auto, log_posterior_ens_auto, auto_correlation = [], [], []
        for i in range(self.nstates):
            if verbose:
                print("Performing auto thinning of ensemble...")
            auto_correlation.append(self.samplers[i].get_autocorr_time(tol=0))
            thin = int(np.ceil(np.max(auto_correlation[i])))
            thin = int(
                np.ceil(0.5 * np.min(auto_correlation[i]))
            )  # use emcee suggestion
            burnin = int(
                np.ceil(2.0 * np.max(auto_correlation[i]))
            )  # use emcee suggestion
            if verbose:
                print("Auto thinning factor calculated = ", thin, " burnin ", burnin)
            samples_auto.append(samples[i][burnin::thin])
            log_posterior_ens_auto.append(log_posterior_ens[i][burnin::thin])
        self.autocorr_times_within_each_state = auto_correlation

        return samples_auto, log_posterior_ens_auto

    def _myfunc(self, x, log_posterior, args):
        """Utility function as internal interface to log_posterior()"""
        return log_posterior(x, *args)

    def run_fitmixture(  # fit a mixture of Gaussians to the ensembles of each state
        self,
        ensemble,
        log_posterior_ens,
        verbose=False,
        return_pseudo_prior_func=False,
        **kwargs,  # Arguments for sklearn.mixture.GaussianMixture
    ):
        """
        Utility routine to fit a Gaussian mixture model to input ensemble for an approximate posterior distribution within each state.
        Makes use of sklearn.mixture.GaussianMixture to fit a mixture model.
        Mixture model is available internally for use by function 'run_ens_mcint()' to act as the pseudo prior in each state.

        Inputs:
        ensemble - list of floats [i,n[i],ndims[i]],(i=0,...,nstates-1) : list containing an ensemble of samples in each state.
                                                                          i is the state;
                                                                          n[i] is the number of samples in state i;
                                                                          ndims[i] is the number of unknowns in state i.
        log_posterior_ens -  list of floats, [i,n[i]].                  : log-posterior of samples in each state.
        verbose - bool                                                  : parameter to produce info to standard out (default = False).
        **kwargs - dict                                                 : kwargs passed to sklean.mixture.GaussianMixture fitting routine

        Returns:
        log_pseudo_prior_ens - floats. : list of estimated log-pseudo_prior densities of samples in each state, format [i][j],(i=1,...,nstates;j=1,..., nwalkers*nsamples).


        Attributes:
        nsamples - nstates*int         : list of number of samples in each state.
        ensemble_per_state - floats    : list of posterior samples in each state, format [i][j][k],(i=1,...,nstates;j=1,..., nwalkers*nsamples;k=1,...,ndim[i]).
        log_posterior_ens - floats     : list of log-posterior densities of samples in each state, format [i][j],(i=1,...,nstates;j=1,..., nwalkers*nsamples).
        log_pseudo_prior_ens - floats. : list of estimated log-pseudo_prior densities of samples in each state, format [i][j],(i=1,...,nstates;j=1,..., nwalkers*nsamples).
        gm - functions                 : list of sklearn.mixture.GaussianMixture models fit to samples in each state
        ncomp - int                    : number of components in sklearn.mixture.GaussianMixture models fit to samples in each state
        run_fit - bool                 : bool to keep track of whether run_fitmixture has been called.

        """

        (gm, ncomp) = ([], [])

        error = False  # check for errors in input shapes of arrays
        if len(ensemble) != len(log_posterior_ens):
            error = True
        for i in range(self.nstates):
            if len(ensemble[i]) != len(log_posterior_ens[i]):
                error = True
        if error:
            raise Inputerror(
                msg=" In function run_fitmixture: Inconsistent shape of inputs\n"
                + "\n Input data for ensemble differs in shape for input data log_posterior_ens\n"
            )

        if not self.run_per_state:  # if the input ensemble is external then store it
            self.ensemble_per_state = ensemble  # store input ensemble
            self.log_posterior_ens = log_posterior_ens  # store input ensemble
            nsamples = []  # calculate number of samples per state for ensemble
            for state in range(self.nstates):
                nsamples.append(len(ensemble[state]))
            self.nsamples = nsamples

        for state in range(self.nstates):  # calculate mixture model
            gmstate = GaussianMixture(**kwargs).fit(self.ensemble_per_state[state])
            gm.append(gmstate)
            ncomp.append(gmstate.get_params()["n_components"])

        self.gm = gm  # Gaussian Mixture model for each state
        self.ncomp = ncomp  # number of mixture components in each state
        self.run_fit = True  # set fag to indicate work done

        if verbose:
            for state in range(self.nstates):
                print(
                    "State : ",
                    state,
                    "\n means",
                    self.gm[state].means_,
                    " covariances ",
                    self.gm[state].covariances_,
                    " weights ",
                    self.gm[state].weights_,
                )
                #
        log_pseudo = []
        for cstate in range(self.nstates):
            log_pseudo.append(
                self.gm[cstate].score_samples(self.ensemble_per_state[cstate])
            )

        self.log_pseudo_prior_ens = (
            log_pseudo  # log pseudo prior values fit for all ensembles
        )

        if return_pseudo_prior_func:
            self.gm_pseudo_func = self.gm

            def log_pseudo_prior(
                x, state, returndeviate=False
            ):  # multi-state log pseudo-prior density and deviate generator

                gmm = self.gm_pseudo_func[
                    state
                ]  # get mixture model approximation for this state
                if returndeviate:
                    dev = gmm.sample()[0]
                    logppx = gmm.score(dev)
                    dev = dev[0]
                    if type(dev) != np.ndarray:
                        dev = np.array(
                            [dev]
                        )  # deal with 1D case which returns a scalar
                    logppx = gmm.score([dev])
                    return logppx, dev
                return gmm.score([x])

            return log_pseudo_prior, log_pseudo

        return log_pseudo

    def build_auto_pseudo_prior(
        self,
        pos,
        log_posterior,
        log_posterior_args=[],
        ensemble_per_state=None,
        log_posterior_ens=None,
        discard=0,
        thin=1,
        autothin=False,
        parallel=False,
        nsamples=1000,
        nwalkers=32,
        return_log_pseudo=False,
        progress=False,
        forcesample=False,
        verbose=False,
        fitmeancov=False,
        **kwargs,  # Arguments for sklearn.mixture.GaussianMixture
    ):
        """
        Utility routine to build an automatic pseudo_prior function using a
        Gaussian mixture model approximation of posterior from small ensemble.
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
        ensemble_per_state - floats : Optional list of posterior samples in each state, format [i][j][k],(i=1,...,nstates;j=1,..., nsamples;k=1,...,ndim[i]).
        log_posterior_ens - floats  : Optional list of log-posterior densities of samples in each state, format [i][j],(i=1,...,nstates;j=1,..., nsamples).
        discard - int, or list      : number of output samples to discard (default = 0). (Parameter passed to emcee, also known as `burnin'.)
        thin - int, or list         : frequency of output samples in output chains to accept (default = 1, i.e. all) (Parameter passed to emcee.)
        autothin - bool             : if true ignores input thin value and instead thins the chain by the maximum auto_correlation time estimated (default = False).
        parallel - bool             : switch to make use of multiprocessing package to parallelize over walkers
        nsamples                    : int or list (of size nstates), optional
                                      Number of samples to draw from each state. The default is 1000.
        nwalkers                    : int or list (of size nstates), optional
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
        if type(nwalkers) != list:
            nwalkers = [nwalkers for i in range(self.nstates)]
        if type(nsamples) != list:
            nsamples = [nsamples for i in range(self.nstates)]

        if (ensemble_per_state is None) and (log_posterior_ens is not None):
            raise Inputerror(
                msg=" In function build_auto_pseudo_prior: Ensemble probabilities provided as argument without ensemble co-ordinates"
            )

        if (ensemble_per_state is not None) and (log_posterior_ens is None):
            raise Inputerror(
                msg=" In function build_auto_pseudo_prior: Ensemble co-ordinates provided as argument without ensemble probabilities"
            )

        if type(ensemble_per_state) == list and type(log_posterior_ens) == list:
            print("We are using input ensembles")

            self.ensemble_per_state = (
                ensemble_per_state  # use provided samples for fitting
            )
            self.log_posterior_ens = log_posterior_ens

        else:
            if (
                self.run_per_state and not forcesample
            ):  # if we are not forced to do sampling and samples already exist then we use them for fitting
                pass
            else:  # generate samples for fitting
                self.run_mcmc_per_state(
                    nwalkers,  # int or list containing number of walkers for each state
                    nsamples,  # number of chain steps per walker
                    pos,  # starting positions for walkers in each state
                    log_posterior,  # log Likelihood x log_prior
                    discard=discard,  # number of initial samples to discard along chain
                    thin=thin,  # frequency of chain samples retained
                    autothin=autothin,  # automatic thining of output ensemble by estimated correlation times
                    parallel=parallel,
                    log_posterior_args=log_posterior_args,  # log posterior additional arguments (optional)
                    verbose=verbose,
                    progress=progress,
                )  # show progress bar for each state

        if fitmeancov:  # we fit ensemble with a simple mean and covariance
            print(" We are fitting mean and covariance")
            rvlist, log_pseudo = [], []
            for state in range(len(self.ensemble_per_state)):
                pseudo_covs_ = np.cov(self.ensemble_per_state[state].T)
                pseudo_means_ = np.mean(self.ensemble_per_state[state].T, axis=1)
                rv = stats.multivariate_normal(mean=pseudo_means_, cov=pseudo_covs_)
                rvlist.append([pseudo_means_, pseudo_covs_, rv])
                log_pseudo.append(rv.logpdf(self.ensemble_per_state[state]))
            self.pseudo_fit_params = rvlist

            self.rvlist = rvlist

            def log_pseudo_prior(
                x, state, size=1, returndeviate=False
            ):  # multi-state log pseudo-prior density and deviate generator

                rv = self.rvlist[state][
                    2
                ]  # get frozen stats.multivariate_normal object with correct mean and covariance
                if returndeviate:
                    x = rv.rvs(size=size)
                    logpseudo = rv.logpdf(
                        x
                    )  # evaluate log prior ignoring dimension prior
                    return logpseudo, x
                return rv.logpdf(x)

            # log_pseudo_prior = partial(log_pseudo_prior_full,rvlist=rvlist)

        else:
            log_pseudo = self.run_fitmixture(
                self.ensemble_per_state, self.log_posterior_ens, **kwargs
            )
            self.gm_pseudo_func = self.gm

            def log_pseudo_prior(
                x, state, returndeviate=False, axisdeviate=False
            ):  # multi-state log pseudo-prior density and deviate generator

                gmm = self.gm_pseudo_func[
                    state
                ]  # get mixture model approximation for this state
                if returndeviate:
                    if (
                        axisdeviate
                    ):  # force deviate to single component along random axis
                        return self._perturb(gmm, x)
                    dev = gmm.sample()[0]
                    logppx = gmm.score(dev)
                    dev = dev[0]
                    if type(dev) != np.ndarray:
                        dev = np.array(
                            [dev]
                        )  # deal with 1D case which returns a scalar
                    logppx = gmm.score([dev])
                    return logppx, dev

                return gmm.score([x])

        if return_log_pseudo:
            return (
                log_pseudo_prior,
                log_pseudo,
            )  # return both pseudo_prior function and log pseudo_prior values
        return log_pseudo_prior

    def _perturb(self, gmm, xc):
        pert = gmm.sample()[0] - gmm.means_
        mask = np.ones(pert.shape[1], dtype=bool)
        i = random.choice(np.arange(len(xc)))
        mask[i] = 0
        pert[0, mask] = 0.0
        y = pert + xc
        return gmm.score(y), y[0]

    def run_product_space_sampler(  # Independent state Metropolis algorithm sampling across product space. This is algorithm 'TransC-product-space'
        self,
        nwalkers,
        nsteps,
        pos,
        pos_state,
        log_posterior,
        log_pseudo_prior,
        log_posterior_args=[],
        log_pseudo_prior_args=[],
        seed=61254557,
        parallel=False,
        nprocessors=1,
        progress=False,
        suppresswarnings=False,  # bool to suppress wanrings
        mypool=False,
        skip_initial_state_check=False,
        **kwargs,
    ):
        """
        MCMC sampler over independent states using emcee fixed dimension sampler over trans-D product space.

        Inputs:
        nwalkers - int               : number of random walkers used by pseudo sampler.
        nsteps - int                 : number of steps required per walker.
        pos - nwalkers*ndims*float   : list of starting locations of markov chains in each state.
        pos_state - nwalkers*int     : list of starting states of markov chains in each state.
        log_posterior()              : user supplied function to evaluate the log-posterior density for the ith state at location x.
                                       calling sequence log_posterior(x,i,*log_posterior_args)
        log_pseudo_prior()           : user supplied function to evaluate the log-pseudo-prior density for the ith state at location x.
                                       calling sequence log_posterior(x,i,*log_posterior_args).
                                       NB: must be normalized over respective state spaces.
        log_posterior_args - list    : user defined (optional) list of additional arguments passed to log_posterior. See calling sequence above.
        log_pseudo_prior_args - list : user defined (optional) list of additional arguments passed to log_pseudo_prior. See calling sequence above.
        prob_state - float           : probability of proposal a state change per step of Markov chain (otherwise a parameter change within current state is proposed)
        seed - int                   : random number seed
        parallel - bool              : switch to make use of multiprocessing package to parallelize over walkers
        nprocessors - int            : number of processors to distribute work across (if parallel=True, else ignored). Default = multiprocessing.cpu_count()/1 if parallel = True/False.
        progress - bool              : switch to report progress to standard out.
        suppresswarnings - bool      : switch to suppress warnings.
        mypool - bool                : switch to use local multiprocessing pool for emcee (experimental feature not recommended)
        kwargs - dict                : dictionary of optional arguments passed to emcee.

        Attributes defined/updated:
        nwalkers - int               : number of random walkers per state
        productspace_sampler - class : emcee sampler class used to sample product space.
        alg - string                 : string defining the sampler method used.

        """

        random.seed(seed)

        self.nwalkers = nwalkers
        self.nsteps = nsteps
        if progress:
            print("\nRunning product space trans-D sampler")
            print("\nNumber of walkers               : ", self.nwalkers)
            print("Number of states being sampled  : ", self.nstates)
            print("Dimensions of each state        : ", self.ndims)

        if parallel and not suppresswarnings:  # do some housekeeping checks
            if self.nwalkers == 1:
                warnings.warn(
                    " Parallel mode used but only a single walker specified. Nothing to parallelize over?"
                )

        ndim_ps = np.sum(self.ndims) + 1  # dimension of product space

        pos_ps = self._modelvectors2productspace(
            pos, pos_state, nwalkers
        )  # convert initial walker positions to product space model vectors

        logfunc = partial(
            self._productspace_log_prob,
            log_posterior=log_posterior,
            log_pseudo_prior=log_pseudo_prior,
            log_posterior_args=log_posterior_args,
            log_pseudo_prior_args=log_pseudo_prior_args,
        )

        if parallel:
            if nprocessors == 1:
                nprocessors = (
                    multiprocessing.cpu_count()
                )  # set number of processors equal to those available

            if (
                mypool
            ):  # try to run emcee myself on separate cores (doesn't make sense for emcee to do this as nwalkers > 2*ndim for performance)
                chunksize = int(
                    np.ceil(nwalkers / nprocessors)
                )  # set work per0 processor
                jobs = [
                    pos_ps[i] for i in range(nwalkers)
                ]  # input data for parallel jobs
                print(" nsteps", nsteps)
                func = partial(
                    self._myemcee,
                    nsteps=nsteps,
                    logfunc=logfunc,
                    ndim=ndim_ps,
                    progress=progress,
                    kwargs=kwargs,
                )
                # return func,jobs,nprocessors,chunksize
                result = []
                pool = multiprocessing.Pool(processes=nprocessors)
                res = pool.map(func, jobs, chunksize=chunksize)
                result.append(res)
                pool.close()
                pool.join()
                return result

            else:  # use emcee in parallel

                with multiprocessing.Pool() as pool:
                    sampler = emcee.EnsembleSampler(  # instantiate emcee class
                        nwalkers, ndim_ps, logfunc, pool=pool, **kwargs
                    )

                    sampler.run_mcmc(pos_ps, nsteps, progress=progress)  # run sampler

        else:

            sampler = emcee.EnsembleSampler(  # instantiate emcee class
                nwalkers, ndim_ps, logfunc, **kwargs
            )

            sampler.run_mcmc(
                pos_ps,
                nsteps,
                progress=progress,
                skip_initial_state_check=skip_initial_state_check,
            )  # run sampler

        self.productspace_sampler = sampler
        self.alg = "TransC-product-space"
        self.nprocessors = nprocessors

    def _myemcee(self, pos, nsteps, logfunc, ndim, progress, kwargs):

        # print(' pos',pos)
        # print(' nsteps',nsteps)
        sampler = emcee.EnsembleSampler(  # instantiate emcee class with a single walker
            1, ndim, logfunc, **kwargs
        )

        sampler.run_mcmc(pos, nsteps, progress=progress)  # run sampler

        return sampler

    def run_pseudo_sampler(  # Independent state MCMC sampler on product space with proposal equal to pseduo prior
        self,
        nwalkers,
        nsteps,
        pos,
        pos_state,
        log_posterior,
        log_pseudo_prior,
        log_proposal,
        log_posterior_args=[],
        log_pseudo_prior_args=[],
        log_proposal_args=[],
        prob_state=0.1,
        seed=61254557,
        parallel=False,
        nprocessors=1,
        progress=False,
        suppresswarnings=False,
        verbose=False,
    ):
        """

        MCMC sampler over independent states using a Metropolis-Hastings algorithm and proposal equal to the supplied pseudo-prior function.

        Calculates Markov chain across states using pseudo

        Inputs:
        nwalkers - int               : number of random walkers used by pseudo sampler.
        nsteps - int                 : number of steps required per walker.
        pos - nwalkers*ndims*float   : list of starting locations of markov chains in each state.
        pos_state - nwalkers*int     : list of starting states of markov chains in each state.
        log_posterior()              : user supplied function to evaluate the log-posterior density for the ith state at location x.
                                       calling sequence log_posterior(x,i,*log_posterior_args)
        log_pseudo_prior()           : user supplied function to evaluate the log-pseudo-prior density for the ith state at location x.
                                       calling sequence log_posterior(x,i,*log_posterior_args).
                                       NB: must be normalized over respective state spaces.
        log_proposal()               : user supplied function to generate random deviate for ith state
                                       calling sequence log_proposal(xc,i,*log_proposal_args), where xc is the current location of the chain (allows for relative proposals)
                                       This is only used for within state moves, and not for between state moves for which it is effectively replaced by the pseudo-prior.
        log_posterior_args - list    : user defined (optional) list of additional arguments passed to log_posterior. See calling sequence above.
        log_pseudo_prior_args - list : user defined (optional) list of additional arguments passed to log_pseudo_prior. See calling sequence above.
        log_proposal_args - list     : user defined (optional) list of additional arguments passed to log_proposal. See calling sequence above.
        prob_state - float           : probability of proposal a state change per step of Markov chain (otherwise a parameter change within current state is proposed)
        seed - int                   : random number seed
        parallel - bool              : switch to make use of multiprocessing package to parallelize over walkers
        nprocessors - int            : number of processors to distribute work across (if parallel=True, else ignored). Default = multiprocessing.cpu_count()/1 if parallel = True/False.
        progress - bool              : switch to report progress to standard out.
        suppresswarnings - bool      : switch to report detailed workings to standard out.
        verbose - bool               : switch to report detailed workings to standard out.

        Attributes defined/updated:
        nsamples - int                                : list of number of samples in each state (calculated from input ensembles if provided).
        nwalkers - int                                : number of random walkers used by pseudo sampler.
        state_chain - nwalkers*nsteps*int             : list of states visited along the trans-D chain.
        state_chain_tot - nwalkers*nsteps*int         : array of cumulative number of visits to each state along the chains.
        model_chain - floats                          : list of trans-D sample along chain.
        alg - string                                  : string defining the sampler method used.

        Notes:
        A simple Metropolis-Hastings MCMC algorithm is used and applied to the product space formulation. Here moves between states are assumed to only perturb the state variable, k-> k'.
        This means that one only needs to generate a new model in state k' from the pseudo-prior of k'. The M-H condition then only involves the current model in state k and the new model in state k',
        with the acceptance criterion then equal to the ratio of the posteriors multiplied by the ratio of the normalized pseudo-priors.
        For within state moves the algorithm becomes normal M-H using a user supplied proposal function to generate new deviates within state k. The user can define this as relative to current model,
        or according to a prescribed PDF within the respective state, e.g. the pseudo-prior again. An independent user supplied proposal function is provided for flexibility.

        """

        self.nsteps = nsteps
        self.nwalkers = nwalkers

        if progress:
            print("\nRunning pseudo-prior trans-D sampler")
            print("\nNumber of walkers               : ", self.nwalkers)
            print("Number of states being sampled  : ", self.nstates)
            print("Dimensions of each state        : ", self.ndims)

        if parallel and not suppresswarnings:  # do some housekeeping checks
            if self.nwalkers == 1:
                warnings.warn(
                    " Parallel mode used but only a single walker specified. Nothing to parallelize over?"
                )

        random.seed(seed)
        self.nsteps = nsteps
        state_chain_tot = np.zeros((nwalkers, nsteps, self.nstates), dtype=int)
        state_chain = np.zeros((nwalkers, nsteps), dtype=int)
        model_chain = []
        accept_within = np.zeros(nwalkers)
        accept_between = np.zeros(nwalkers)
        prop_within = np.zeros(nwalkers)
        prop_between = np.zeros(nwalkers)

        if parallel:  # put random walkers on different processors

            if nprocessors == 1:
                nprocessors = (
                    multiprocessing.cpu_count()
                )  # set number of processors equal to those available
            chunksize = int(np.ceil(nwalkers / nprocessors))  # set work per processor
            jobs = [
                (pos_state[i], pos[i]) for i in range(nwalkers)
            ]  # input data for parallel jobs
            func = partial(
                self._mcmc_walker,
                log_posterior=log_posterior,
                log_pseudo_prior=log_pseudo_prior,
                log_proposal=log_proposal,
                log_posterior_args=log_posterior_args,
                log_pseudo_prior_args=log_pseudo_prior_args,
                log_proposal_args=log_proposal_args,
                nsteps=nsteps,
                prob_state=prob_state,
                verbose=verbose,
            )
            result = []
            if progress:
                with multiprocessing.Pool(processes=nprocessors) as pool:
                    res = list(
                        tqdm(
                            pool.imap_unordered(func, jobs, chunksize=chunksize),
                            total=len(jobs),
                        )
                    )
            else:
                pool = multiprocessing.Pool(processes=nprocessors)
                res = pool.map(func, jobs, chunksize=chunksize)
            result.append(res)
            pool.close()
            pool.join()
            for i in range(nwalkers):  # decode the output
                state_chain_tot[i] = result[0][i][2]
                state_chain[i] = result[0][i][1]
                model_chain.append(result[0][i][0])
                accept_within[i] = result[0][i][3]
                accept_between[i] = result[0][i][5]
                prop_within[i] = result[0][i][4]
                prop_between[i] = result[0][i][6]

        else:
            for walker in self._myrange(progress, nwalkers):  # loop over walkers
                cstate = pos_state[walker]  # initial state
                cmodel = pos[walker]
                out = self._mcmc_walker(
                    [cstate, cmodel],
                    log_posterior,
                    log_pseudo_prior,
                    log_posterior_args,
                    log_pseudo_prior_args,
                    log_proposal,
                    log_proposal_args,
                    nsteps,
                    prob_state,
                    verbose,
                )

                (
                    chain,
                    state_chainw,
                    state_chain_totw,
                    accept_within[walker],
                    prop_within[walker],
                    accept_between[walker],
                    prop_between[walker],
                ) = out
                state_chain_tot[walker] = state_chain_totw
                state_chain[walker] = state_chainw
                model_chain.append(chain)  # record locations visited for this walker

        self.alg = "TransC-pseudo-sampler"
        self.state_chain_tot = np.swapaxes(state_chain_tot, 0, 1)
        self.state_chain = state_chain.T
        self.model_chain = model_chain
        self.accept_within_per_walker = accept_within / prop_within
        self.accept_between_per_walker = accept_between / prop_between
        self.accept_within = 100 * np.mean(self.accept_within_per_walker)
        self.accept_between = 100 * np.mean(self.accept_between_per_walker)
        self.nprocessors = nprocessors

    def _myrange(self, progress, length):
        if progress:
            return tqdm(range(length))
        return range(length)

    def _mcmc_walker(
        self,
        cstate_cmodel,
        log_posterior,
        log_pseudo_prior,
        log_posterior_args,
        log_pseudo_prior_args,
        log_proposal,
        log_proposal_args,
        nsteps,
        prob_state,
        verbose,
    ):

        cstate, cmodel = cstate_cmodel
        visits = np.zeros(self.nstates, dtype=int)
        lpostc = log_posterior(
            cmodel, cstate, *log_posterior_args
        )  # initial log-posterior
        lpseudoc = log_pseudo_prior(
            cmodel, cstate, *log_pseudo_prior_args
        )  # initial log-pseudo prior
        chain = []
        state_chain_tot = np.zeros((nsteps, self.nstates), dtype=int)
        state_chain = np.zeros((nsteps), dtype=int)
        prop_between, prop_within, accept_within, accept_between = 0, 0, 0, 0

        for chainstep in range(nsteps):  # loop over markov chain steps

            if random.random() < prob_state:  # Choose to propose a new state

                states = [j for j in range(self.nstates)]  # list of all states
                states.remove(cstate)  # list of available states
                pstate = random.choice(states)  # choose proposed state
                if verbose:
                    print("current state", cstate, " propose state", pstate)
                within = False
                prop_between += 1
                lpseudop, pmodel = log_pseudo_prior(
                    None, pstate, *log_pseudo_prior_args, returndeviate=True
                )  # log pseudo-prior for proposed state and generate it

                logpert = lpseudoc - lpseudop  # log difference in pseduo-priors

            else:  # Choose to propose a new model within current state

                pstate = np.copy(cstate)  # retain current state
                if verbose:
                    print("within state", cstate, " model change")
                within = True
                prop_within += 1
                logpert, pmodel = log_proposal(
                    cmodel, pstate, *log_proposal_args
                )  # generate proposed model in current state and calculate log density ratio

            lpostp = log_posterior(
                pmodel, pstate, *log_posterior_args
            )  # log posterior for proposed state

            logr = lpostp - lpostc + logpert  # Metropolis-Hastings acceptance criterion

            if logr >= np.log(random.random()):  # Accept move
                if verbose:
                    print(" Accept move")
                    print(" cmodel", cmodel, "pmodel", pmodel)
                visits[pstate] += 1
                cstate = np.copy(pstate)
                cmodel = np.copy(pmodel)
                lpostc = np.copy(lpostp)
                if within:
                    lpseudop = log_pseudo_prior(
                        pmodel, pstate, *log_pseudo_prior_args
                    )  # record log pseudo-prior for new state
                lpseudoc = np.copy(lpseudop)
                if within:
                    accept_within += 1
                else:
                    accept_between += 1
            else:  # Reject move
                if verbose:
                    print(" Reject move")
                    print(" cmodel", cmodel, "pmodel", pmodel)
                visits[cstate] += 1

            chain.append(cmodel)
            state_chain[chainstep] = cstate  # record state for this step and walker
            state_chain_tot[chainstep] = (
                visits  # record cumulative tally of states visited for this step and walker
            )

        return (
            chain,
            state_chain,
            state_chain_tot,
            accept_within,
            prop_within,
            accept_between,
            prop_between,
        )

    def run_ensemble_sampler(  # Independent state Marginal Likelihoods from pre-computed posterior and pseduo prior ensembles
        self,
        nwalkers,
        nsteps,
        log_posterior_ens=None,
        log_pseudo_prior_ens=None,
        seed=61254557,
        parallel=False,
        nprocessors=1,
        stateproposalweights=None,
        progress=False,
    ):
        """
        MCMC sampler over independent states using a Markov Chain.

        Calculates relative evdience of each state by sampling over previously computed posterior ensembles for each state.
        Requires only log density values for posterior and pseudo priors at the sampe locations (not actual samples).
        This routine is an alternate to run_ens_mcint(), using the same inputs of log density values of posterior samples within each state.
        Here a single Markov chain is used.

        Inputs:
        nwalkers - int                                                       : number of random walkers used by ensemble sampler.
        nsteps - int                                                         : number of Markov chain steps to perform
        log_posterior_ens -  list of floats, [i,n[i]], (i=1,...,nstates)     : log-posterior of ensembles in each state, where n[i] is the number of samples in the ith state.
        log_pseudo_prior_ens -  list of floats, [i,n[i]], (i=1,...,nstates)  : log-pseudo prior of samples in each state, where n[i] is the number of samples in the ith state.
        seed - int                                                           : random number seed
        parallel - bool                                                      : switch to make use of multiprocessing package to parallelize over walkers
        nprocessors - int                                                    : number of processors to distribute work across (if parallel=True, else ignored). Default = multiprocessing.cpu_count()/1 if parallel = True/False.
        progress - bool                                                      : option to write diagnostic info to standard out

        Attributes defined/updated:
        nstates - int                                 : number of independent states (calculated from input ensembles if provided).
        nsamples - int                                : list of number of samples in each state (calculated from input ensembles if provided).
        state_chain_tot - nsamples*int                : array of states visited along the trans-D chain.
        alg - string                                  : string defining the sampler method used.


        Notes:
        The input posterior samples and log posterior values in each state can be either be calcuated using utility routine 'run_mcmc_per_state', or provided by the user.
        The input log values of pseudo prior samples in each state can be either be calcuated using utility routine 'run_fitmixture', or provided by the user.

        """

        if (
            not self.run_fit
        ):  # mixture fitting has not been performed and so we need input ensembles

            if (log_pseudo_prior_ens is None) and (log_posterior_ens is not None):
                raise Inputerror(
                    msg=" In function run_is_ensemble_sampler: Ensemble probabilities provided as argument without pseudo-prior probabilities"
                )

            if (log_pseudo_prior_ens is not None) and (log_posterior_ens is None):
                raise Inputerror(
                    msg=" In function run_is_ensemble_sampler: Pseudo-prior probabilities provided as argument without ensemble probabilities"
                )

            if (log_pseudo_prior_ens is None) and (log_posterior_ens is None):
                raise Inputerror(
                    msg=" In function run_is_ensemble_sampler: Pseudo-prior probabilities and ensemble probabilities not provided"
                )

            self.nstates = len(log_posterior_ens)
            self.nsamples = [len(a) for a in log_posterior_ens]

        else:  # mixture fitting has been performed
            if (log_pseudo_prior_ens is not None) and (
                log_posterior_ens is not None
            ):  # we use input ensembles if available
                self.nstates = len(log_posterior_ens)
                self.nsamples = [len(a) for a in log_posterior_ens]

        self.nwalkers = nwalkers
        self.nsteps = nsteps
        print("\nRunning ensemble trans-D sampler")
        print("\nNumber of walkers               : ", self.nwalkers)
        print("Number of states being sampled  : ", self.nstates)
        print("Dimensions of each state        : ", self.ndims)

        random.seed(seed)
        state_chain_tot = np.zeros((nwalkers, nsteps, self.nstates), dtype=int)
        state_chain = np.zeros((nwalkers, nsteps), dtype=int)
        accept_between = np.zeros(nwalkers, dtype=int)
        if parallel:

            if nprocessors == 1:
                nprocessors = (
                    multiprocessing.cpu_count()
                )  # set number of processors equal to those available
            chunksize = int(np.ceil(nwalkers / nprocessors))  # set work per processor
            jobs = random.choices(
                range(self.nstates), k=nwalkers
            )  # input data for parallel jobs
            func = partial(
                self._mcmc_walker_ens,  # create reduced one argument function for passing to pool.map())
                nsteps=nsteps,
                log_posterior_ens=log_posterior_ens,
                log_pseudo_prior_ens=log_pseudo_prior_ens,
                stateproposalweights=stateproposalweights,
            )
            result = []
            if progress:
                with multiprocessing.Pool(processes=nprocessors) as pool:
                    res = list(
                        tqdm(
                            pool.imap(func, jobs, chunksize=chunksize), total=len(jobs)
                        )
                    )
            else:
                pool = multiprocessing.Pool(processes=nprocessors)
                res = pool.map(func, jobs, chunksize=chunksize)
            result.append(res)
            pool.close()
            pool.join()
            for i in range(nwalkers):  # decode the output
                state_chain_tot[i] = result[0][i][1]
                state_chain[i] = result[0][i][0]
                accept_between[i] = result[0][i][2]

            pass
        else:

            for walker in self._myrange(progress, nwalkers):
                cstate = random.choice(
                    range(self.nstates)
                )  # choose initial current state randomly
                state_chain[walker], state_chain_tot[walker], accept_between[walker] = (
                    self._mcmc_walker_ens(
                        cstate,
                        nsteps,
                        log_posterior_ens,
                        log_pseudo_prior_ens,
                        stateproposalweights=stateproposalweights,
                    )
                )  # carry out an mcmc walk between ensembles

        self.alg = "TransC-ensemble-sampler"
        self.state_chain_tot = np.swapaxes(state_chain_tot, 0, 1)
        self.state_chain = state_chain.T
        self.accept_within_per_walker = 1.0 * np.ones(nwalkers)
        self.accept_between_per_walker = accept_between / nsteps
        self.accept_within = 100 * np.mean(self.accept_within_per_walker)
        self.accept_between = 100 * np.mean(self.accept_between_per_walker)
        self.nprocessors = nprocessors

    def _mcmc_walker_ens(
        self,
        cstate,
        nsteps,
        log_posterior_ens,
        log_pseudo_prior_ens,
        stateproposalweights=None,
        verbose=False,
    ):
        """Internal one chain MCMC sampler used by run_ensemble_sampler()"""

        visits = np.zeros(self.nstates)
        state_chain_tot = np.zeros((nsteps, self.nstates), dtype=int)
        state_chain = np.zeros((nsteps), dtype=int)
        cmember = random.choice(
            range(self.nsamples[cstate])
        )  # randomly choose ensemble member from current state
        visits[cstate] += 1
        state_chain[0] = cstate  # record initial state for this step and walker
        state_chain_tot[0] = visits  # record initial current state visited by chain
        accept = 0
        if stateproposalweights is None:
            stateproposalweights = np.ones((self.nstates, self.nstates))
        else:
            np.fill_diagonal(stateproposalweights, 0.0)  # ensure diagonal is zero
            stateproposalweights = stateproposalweights / stateproposalweights.sum(
                axis=1, keepdims=1
            )  # set row sums to unity

        for chainstep in range(nsteps - 1):  # loop over markov chain steps

            states = [j for j in range(self.nstates)]  # list of all states
            states.remove(cstate)  # list of available states
            # weights = stateweights[np.ix_(np.delete(np.arange(self.nstates),cstate),np.delete(np.arange(self.nstates),cstate))]
            weights = stateproposalweights[
                cstate, np.delete(np.arange(self.nstates), cstate)
            ]
            # pstate = random.choice(states)  # choose proposed state
            pstate = random.choices(states, weights=weights)[0]  # choose proposed state
            pmember = random.choice(
                range(self.nsamples[pstate])
            )  # randomly select ensemble member from proposed state

            if (log_pseudo_prior_ens is not None) and (
                log_posterior_ens is not None
            ):  # use provided pseudo-priors

                lpseudoc = log_pseudo_prior_ens[cstate][
                    cmember
                ]  # log pseudo-prior for current state
                lpseudop = log_pseudo_prior_ens[pstate][
                    pmember
                ]  # log pseudo-prior for proposed state
                lpostc = log_posterior_ens[cstate][
                    cmember
                ]  # log posterior for current state
                lpostp = log_posterior_ens[pstate][
                    pmember
                ]  # log posterior for proposed state

            elif self.run_fit:  # use internally calculated pseudo-priors

                lpseudoc = self.log_pseudo_prior_ens[cstate][
                    cmember
                ]  # log pseudo-prior for current state
                lpseudop = self.log_pseudo_prior_ens[pstate][
                    pmember
                ]  # log pseudo-prior for proposed state
                lpostc = self.log_posterior_ens[cstate][
                    cmember
                ]  # log posterior for current state
                lpostp = self.log_posterior_ens[pstate][
                    pmember
                ]  # log posterior for proposed state

            lprop = np.log(stateproposalweights[pstate, cstate]) - np.log(
                stateproposalweights[cstate, pstate]
            )

            logr = (
                lpostp + lpseudoc - lpostc - lpseudop + lprop
            )  # Metropolis-Hastings acceptance criteria

            if logr >= np.log(random.random()):  # Accept move between states
                visits[pstate] += 1
                cstate = np.copy(pstate)
                cmember = np.copy(pmember)
                accept += 1
            else:

                # Reject move between states

                visits[cstate] += 1

            state_chain[chainstep + 1] = cstate  # record state for this step and walker
            state_chain_tot[chainstep + 1] = (
                visits  # record current state visited by chain
            )

        self.stateproposalweights = stateproposalweights

        return state_chain, state_chain_tot, accept

    def run_ens_mcint(  #  Marginal Likelihoods from Monte Carlo integration
        self,
        log_posterior_ens=None,
        log_pseudo_prior_ens=None,
        return_marginallikelihoods=False,
    ):
        """
        Utility routine to perform MCMC sampler over independent states using numerical integration.

        This routine is a faster alternate to running a Markov chain across the ensembles, which is carried out by run_is_ensemble_sampler.
        Calculates relative evdience of each state using previously computed ensembles in each state.

        Inputs:
        log_posterior_ens -  list of floats, [i,j], (i=1,...,nstates;j=1,...,nsamples[i])    : log-posterior of ensembles in each state, where nsamples[i] is the number of samples in the ith state.
        log_pseudo_prior_ens -  list of floats, [i,j], (i=1,...,nstates;j=1,...,nsamples[i]) : log-pseudo prior of samples in each state, where nsamples[i] is the number of samples in the ith state.
        return_marginallikelihoods - bool                                                    : option to return the calculated relative evidence/marginal likelihood of each state.

        Attributes defined/updated:
        nstates - int                                 : number of independent states (calculated from input ensembles if provided).
        nsamples - int                                : list of number of samples in each state (calculated from input ensembles if provided).
        relative_marginal_likelihoods - nstates*float : list of relative evidence/marginal Likelihoods for each state.
        ens_mc_samples - floats.                      : list of density ratios in Monte Carlo integration. format [i][j],[i=1,...,nstates;j=1,...,nsamples[i]).
        alg - string                                  : string defining the sampler method used.

        Notes:
        The input log values of posterior samples in each state can be either be calculated using utility routine 'run_mcmc_per_state', or provided by the user.
        The input log values of pseudo prior samples in each state can be either be calcuated using utility routine 'run_fitmixture', or provided by the user.

        """

        if (
            not self.run_fit
        ):  # mixture fitting has not been performed and so we need input ensembles, so check that we have them

            if (log_pseudo_prior_ens is None) and (log_posterior_ens is not None):
                raise Inputerror(
                    msg=" In function run_ens_mcint: Ensemble probabilities provided as argument without pseudo-prior probabilities"
                )

            if (log_pseudo_prior_ens is not None) and (log_posterior_ens is None):
                raise Inputerror(
                    msg=" In function run_ens_mcint: Pseudo-prior probabilities provided as argument without ensemble probabilities"
                )

            if (log_pseudo_prior_ens is None) and (log_posterior_ens is None):
                raise Inputerror(
                    msg=" In function run_ens_mcint: Pseudo-prior probabilities and ensemble probabilities not provided"
                )

            self.nstates = len(log_posterior_ens)
            self.nsamples = [len(a) for a in log_posterior_ens]

        else:  # mixture fitting has been performed
            if (log_pseudo_prior_ens is not None) and (
                log_posterior_ens is not None
            ):  # we use input ensembles if available
                self.nstates = len(log_posterior_ens)
                self.log_pseudo_prior_ens = log_pseudo_prior_ens
                self.log_posterior_ens = log_posterior_ens
            else:
                log_pseudo_prior_ens = self.log_pseudo_prior_ens
                log_posterior_ens = self.log_posterior_ens

        self.alg = "TransC-integration"

        ens_r = []
        ens_mc_samples = []
        factor = np.min(
            [
                np.min(log_posterior_ens[i] - log_pseudo_prior_ens[i])
                for i in range(self.nstates)
            ]
        )
        for state in range(self.nstates):
            ratio_state = np.exp(
                log_posterior_ens[state] - log_pseudo_prior_ens[state] - factor
            )
            ens_mc_samples.append(ratio_state)
            ens_r.append(np.mean(ratio_state))

        tot = np.sum(ens_r)
        for state in range(self.nstates):
            ens_mc_samples[state] /= tot
        self.ens_mc_samples = ens_mc_samples

        if return_marginallikelihoods:
            return ens_r / tot
        else:
            self.relative_marginal_likelihoods = ens_r / tot

    def run_laplace_evidence_approximation(
        self,
        log_posterior,
        map_models, 
        log_posterior_args=[],
        ensemble_per_state=None,
        log_posterior_ens=None,
        verbose=False,
        optimize=False,
        **kwargs,  # Arguments for sklearn.mixture.GaussianMixture
    ):
        """
        Function to perform Laplace interation for evidence approximation within each state, using either an input log-posterior function, or posterior ensembles.

        Parameters
        ----------
        log_posterior : function    : Supplied function evaluating the log-posterior function
                                      up to a multiplicative constant, for each state. 
                                      (Not used if ensemble_per_state and log_posterior_ens lists are provided)
                                      Calling sequence log_posterior(x,state,*log_posterior_args)
        map_models - floats         : List of MAP models in each state where Laplace approximation is evaluated. 
                                      If optimize=True and a log_posterior() function is supplied, then 
                                      scipy.minimize is used to find MAP models in each state using map_models as starting guesses.
        log_posterior_args          : Optional list of additional arguments required by user function log_posterior.
        ensemble_per_state - floats : Optional list of posterior samples in each state, format [i][j][k],(i=1,...,nsamples;j=1,..., nmodels;k=1,...,ndim[i]).
        log_posterior_ens - floats  : Optional list of log-posterior densities of samples in each state, format [i][j],(i=1,...,nstates;j=1,..., nsamples).
        optimize, bool              : Logical to decide whether to use optimization for MAP models (Only relevant if log_posterior()) function supplied.)
        
        Attributes defined:
        -------
        log_marginal_likelihoods_laplace - nstates*float : list of log-evidence/marginal Likelihoods for each state.

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
            raise Inputerror(
                msg=" In function run_laplace_evidence_approximation: Ensemble probabilities provided as argument without ensemble co-ordinates"
            )

        if (ensemble_per_state is not None) and (log_posterior_ens is None):
            raise Inputerror(
                msg=" In function run_laplace_evidence_approximation: Ensemble co-ordinates provided as argument without ensemble probabilities"
            )

        if type(ensemble_per_state) == list and type(log_posterior_ens) == list: # we use input ensemble
            if(verbose): print("run_laplace_evidence_approximation: We are using input ensembles rather than input log_posterior function")

            # we fit a mean and covariance to esnembles in each state
            #if(verbose): print("Fitting mean and covariance of input ensembles")
            covs_, maps_,lpms_ = [],[],[]
            lml = []
            for state in range(len(ensemble_per_state)): #loop over states
                covs_.append(np.cov(ensemble_per_state[state].T)) # calculate covariance matrices for state
                j = np.argmax(log_posterior_ens[state])  # get map model index  
                maps_.append(ensemble_per_state[state][j])  # get map model 
                lpms_.append(log_posterior_ens[state][j])
                covar = np.cov(ensemble_per_state[state].T)
                p1 = ((self.ndims[state]/2.)*np.log(2*np.pi))
                p3 = log_posterior_ens[state][j]
                # get determinant of negative inverse of covariance (-H) (NB determinant sign depends on dimension being odd or even)
                if(self.ndims[state] % 2): # dimension is odd number
                    if(self.ndims[state] == 1):
                        p2 = 0.5*np.log(covar)
                    else:
                        p2 = 0.5*np.log(np.linalg.det(covar))
                else:
                    p2 = 0.5*np.log(np.linalg.det(-covar))
                lml.append(p1+p2+p3)
            laplace_hessians =  covs_
            map_models_per_state = maps_
            map_log_posteriors = lpms_
        else: # we are using the supplied log_posterior() function so need Hessian and MAp model
            if(verbose): 
                if(optimize): 
                    print("run_laplace_evidence_approximation: We are using input log_posterior function with optimization for MAP models")
                else:
                    print("run_laplace_evidence_approximation: We are using input log_posterior function with provided MAP models")

            laplace_hessians,map_models_per_state,map_log_posteriors = [],[],[]
            lml = []
            for state in range(self.nstates):
                fun = lambda x: log_posterior(x, state, *log_posterior_args)
                if(optimize):
                    fun2 = lambda x: -log_posterior(x, state, *log_posterior_args)
                    soln = minimize(fun2, map_models[state])
                    map_model = soln.x
                else:
                    map_model = np.array(map_models[state])
                dfun = nd.Hessian(fun)
                laplace_hessians.append(dfun(map_model))
                map_models_per_state.append(map_model)
                map_log_posteriors.append(fun(map_model))
                
                p1 = ((self.ndims[state]/2.)*np.log(2*np.pi))
                p3 = fun(map_model)
                det = np.linalg.det(-dfun(map_model))
                #print(det)
                p2 = -0.5*np.log(det)
                lml.append(p1+p2+p3)
        
        self.log_marginal_likelihoods_laplace = lml
            
    def get_visits_to_states(  # calculate evolution of relative visits to each state along chain
        self,
        discard=0,
        thin=1,
        normalize=False,
        flat=False,
        walker_average="median",
        return_samples=False,
        calc_autocorr=True,
        ntd_samples=None,
    ):
        """
        Utility routine to retrieve proportion of visits to each state as a function of chain step, i.e. calculates the relative evidence/marginal Liklihoods of states.
        Collects information from previously run sampler. Can be used to diagnose performance and convergence.

        Inputs:
        discard - int               : number of output samples to discard (default = 0). (Also known as `burnin'.)
        thin - int                  : frequency of output samples in output chains to accept (default = 1, i.e. all)
        normalize - bool            : switch to calculate normalize relative evidence (True) or total visits to each state (False).
        flat - bool                 : switch to flatten walkers to a single chain (True) or calculate properties per chain (False).
                                      if false then information per chain can indicate whether some chains have not converged.
        walker_average - string     : indicates type of average pf visit statistics to calculate per chain step. Options are: 'median' (default) or 'mean'.
                                      'median' provides more diagnostics if a subset of chains have not converged and statistics are outliers.
        return_samples - bool       : switch to (optionally) return a record of visits to states for each step of each chain.

        Returns:
        visits - list int           : distribution of states visited as a function of chain step.
                                      either per chain (flat=False), or overall (flat=True).
                                      either normalized (normalize=True) or raw numbers (normalize=False).
                                      size equal to number of Markov chain steps retained (depends on discard and thin values).
        samples - list              : actual indices of states visited along each Markov chain (return_samples=True).
                                      used to view details of chain movement between states, largely for convergence checks.

        Attributes defined/updated:
        state_chain - ints                   : list of states visited along markov chains
        relative_marginal_likelihoods        : ratio of evidences/marginal Likelihods of each state
        state_changes_perwalker - array ints : number of times the walker changed state along the markov chain
        total_state_changes - int            : total number of state changes for all walkers

        """
        if (
            self.alg == "TransC-product-space"
        ):  # calculate fraction of visits to each state along chain averaged over walkers

            samples = self.productspace_sampler.get_chain(
                discard=discard, thin=thin
            )  # collect model ensemble
            self.state_chain = np.rint(samples[:, :, 0]).astype("int")
            visits = np.zeros(
                (np.shape(samples)[0], np.shape(samples)[1], self.nstates)
            )
            for i in range(self.nstates):
                visits[:, :, i] = np.cumsum(self.state_chain == i, axis=0)
            self.state_chain_tot = visits
            if normalize:
                for i in range(self.nstates):
                    visits /= np.sum(visits, axis=2)[:, :, np.newaxis]
            out = visits
            if flat and walker_average == "mean":
                out = np.mean(
                    visits, axis=1
                )  # fraction of visits to each state along chain averaged over walkers
            if flat and walker_average == "median":
                out = np.median(
                    visits, axis=1
                )  # fraction of visits to each state along chain averaged over walkers
            if flat:
                self.relative_marginal_likelihoods = out[-1]
            else:
                self.relative_marginal_likelihoods = np.mean(out[-1], axis=0)
            samples = self.state_chain

            self.acceptance_rate_perwalker = (
                self.productspace_sampler.acceptance_fraction
            )
            self.acceptance_rate = 100 * np.mean(self.acceptance_rate_perwalker)
            if calc_autocorr:
                self.mean_autocorr_time = np.mean(
                    self.productspace_sampler.get_autocorr_time(tol=0)
                )  # mean autocorrelation time in steps for all parameters
                self.max_autocorr_time = np.max(
                    self.productspace_sampler.get_autocorr_time(tol=0)
                )  # max autocorrelation time in steps for all parameters
                self.autocorr_time_for_between_state_jumps = (
                    self.productspace_sampler.get_autocorr_time(tol=0)[0]
                )

        elif (
            self.alg == "TransC-integration"
        ):  # generate samples over states with calculated evidences as weights
            if ntd_samples is None:
                ntd_samples = len(self.ens_mc_samples[0])
            samples = np.random.choice(
                self.nstates, ntd_samples, p=self.relative_marginal_likelihoods
            )
            return samples
        else:

            visits = self.state_chain_tot[discard::thin, :, :].astype("float")
            if normalize:
                visits /= np.sum(visits, axis=2)[:, :, np.newaxis]
            out = visits
            samples = self.state_chain[discard::thin, :]
            if flat and walker_average == "mean":
                out = np.mean(
                    visits, axis=1
                )  # fraction of visits to each state along chain averaged over walkers
            if flat and walker_average == "median":
                out = np.median(
                    visits, axis=1
                )  # fraction of visits to each state along chain averaged over walkers
            if flat:
                rml = out[-1]
            else:
                rml = np.mean(out[-1], axis=0)
            if normalize:
                rml /= np.sum(rml)
            self.relative_marginal_likelihoods = rml
            self.autocorr_time_for_between_state_jumps = autocorr_fardal(
                self.state_chain.T
            )

        changes = np.zeros(self.nwalkers, dtype=int)
        for i in range(self.nwalkers):
            changes[i] = np.count_nonzero(samples.T[i][1:] - samples.T[i][:-1])

        self.state_changes_perwalker = changes
        self.total_state_changes = np.sum(changes)
        if self.alg != "TransC-integration":
            self.acceptance_rate_between_states = (
                100 * self.total_state_changes * thin / (self.nwalkers * self.nsteps)
            )

        if return_samples:
            return (
                out,
                samples,
            )  # fraction of visits to state along chain with states along chain for all walkers
        return out  # fraction of visits to each state along chain for all walkers

    def get_transc_samples(
        self,
        ntd_samples=1000,
        discard=0,
        thin=1,
        returnchains=False,
        ensemble_per_state=None,
        flat=False,
        verbose=False,
    ):  # generate a trans-d ensemble from either TransC-ens or TransC-ps samplers
        """
        Utility routine to retrieve list of trans-C model space samples, previously calculated by
        either run_is_ensemble_sampler(),run_product_space_sampler() or run_ens_mcint() or run_is_pseudo_sampler.
        For algorithms TransC-ensemble-sampler and TransC-integration the input variable 'ntd_samples' determines the number of trans-D model space samples generated and then returned.
        For algorithms TransC-product-space and TransC-pseudo-sampler the number of trans-D model space samples is determined by the original sampler and modified by chain thinning (see `discard` and `thin` parameters).

        Inputs:
        ntd_samples - int           : number of trans-D samples to generate [for algs 'TransC-integration' or 'TransC-ensemble-sampler'].
        discard - int               : number of output samples to discard, also known as `burnin' (default = 0)) [only relevant if algs = 'TransC-product-space' or 'TransC-pseudo-sampler'].
        thin - int                  : frequency of output samples in output chains to accept (default = 1, i.e. all) [only relevant if algs = 'TransC-product-space' or 'TransC-pseudo-sampler'].
        returnchains - bool         : switch to return states of each trans_d sample returned. (Default = False)
        verbose - bool              : switch to print some diagnostic info to standard out.


        Returns:
        transd_ensemble - list      : list of trans-D samples ordered by state.
                                      if flat=True, format is [state,i],i=1,...,n(state); where n(state) is the number of models generated in state i.
                                      otherwise format is [state,walker,i], where samples are also separated by their walker.
                                      if alg is `TransC-product-space` or `TransC-pseudo-sampler` size of ensemble returned depends on values of discard and flat.
                                      if alg is `TransC-ensemble-sampler` or `TransC-integration`,  size of ensemble returned is given by ntd_samples.
        model_chain - floats        : list of trans-D samples. For ensemble generated by `run_product_space_sampler`, number determined by
                                      discard and thin (only if returnchains = True).
        states_chain - ints. : list of states of trans-D samples (only if returnchains = True).

        """
        # if(not hasattr(self, 'relative_marginal_likelihoods')): # need to call get_visits_to_states for marginal Likelihoods

        if (
            self.alg == "TransC-ensemble-sampler" or self.alg == "TransC-integration"
        ):  # draw random trans-D models according to relative marginals for TransC-ens sampler

            if hasattr(
                self, "ensemble_per_state"
            ):  # we have an internally calculated ensemble per state
                ensemble_per_state = self.ensemble_per_state
            elif (
                ensemble_per_state is not None
            ):  # no ensemble provided either internally or via calling sequence
                pass
            else:
                raise Inputerror(
                    msg=" In function get_transc_samples: No ensemble provided either as argument ensemble_per_state or generated via self.run_mcmc_per_state()"
                )

            if verbose:
                print(
                    "\n Generating trans-dimensional ensemble of size ",
                    ntd_samples,
                    " using algorithm: ",
                    self.alg,
                    "\n",
                )

            # if(self.alg == 'TransC-integration'):
            # states_chain = self.get_visits_to_states(normalize=True,flat=True)

            states_chain = np.random.choice(
                self.nstates, size=ntd_samples, p=self.relative_marginal_likelihoods
            )

            model_chain = ntd_samples * [None]
            for i in range(
                ntd_samples
            ):  # randomly select models from input state ensembles using evidence weights
                j = np.random.choice(self.nsamples[states_chain[i]])
                model_chain[i] = ensemble_per_state[states_chain[i]][j]

            transd_ensemble = (
                []
            )  # create transd ensemble of models ordered by states for a single walker
            for i in range(self.nstates):
                ind = [num for num, n in enumerate(states_chain) if n == i]
                transd_ensemble.append(np.array([model_chain[j] for j in ind]))

        elif (
            self.alg == "TransC-product-space"
        ):  # build trans-D model ensemble from product space chains for TransC-ps sampler

            samples = self.productspace_sampler.get_chain(
                discard=discard, thin=thin, flat=flat
            )  # collect model ensemble
            ind1 = np.cumsum(self.ndims) + 1
            ind0 = np.append(np.array([1]), ind1)

            transd_ensemble = []  # create transd ensemble of models ordered by states

            if flat:  # combine walkers
                self.state_chain = np.rint(samples[:, 0]).astype("int")
                model_chain = []
                nsteps = np.shape(self.state_chain)[0]
                for i in range(nsteps):
                    model_chain.append(
                        samples[
                            i, ind0[self.state_chain[i]] : ind1[self.state_chain[i]]
                        ]
                    )

                for i in range(self.nstates):
                    ind = [num for num, n in enumerate(self.state_chain) if n == i]
                    transd_ensemble.append(np.array([model_chain[j] for j in ind]))

            else:  # separate ensemble by walkers
                self.state_chain = np.rint(samples[:, :, 0]).astype("int")
                nsteps, nwalkers = np.shape(self.state_chain)
                model_chain = []
                for i in range(nsteps):
                    m = []
                    for j in range(nwalkers):
                        m.append(
                            samples[
                                i,
                                j,
                                ind0[self.state_chain[i, j]] : ind1[
                                    self.state_chain[i, j]
                                ],
                            ]
                        )
                    model_chain.append(m)

                st = np.transpose(self.state_chain)
                nwalkers = np.shape(self.state_chain)[1]
                for i in range(self.nstates):
                    t = []
                    for k in range(nwalkers):
                        ind = [num for num, n in enumerate(st[k]) if n == i]
                        t.append(np.array([model_chain[j][k] for j in ind]))
                    transd_ensemble.append(t)

            states_chain = self.state_chain

        elif (
            self.alg == "TransC-pseudo-sampler"
        ):  # build trans-D model ensemble from product space chains for TransC-pseudo-sampler

            model_chain = [
                row[discard::thin] for row in self.model_chain
            ]  # stride the list
            states_chain = self.state_chain[discard::thin, :]  # stride the array

            transd_ensemble = []  # create transd ensemble of models ordered by states
            nsteps, nwalkers = np.shape(states_chain)
            a = np.transpose(states_chain)

            for i in range(self.nstates):
                t = []
                for k in range(nwalkers):
                    ind = [num for num, n in enumerate(a[k]) if n == i]
                    t.append([model_chain[k][j] for j in ind])
                if flat:  # combine walkers
                    transd_ensemble.append(self._flatten_extend(t))
                else:  # separate ensemble by walkers
                    transd_ensemble.append(t)

        if returnchains:
            return transd_ensemble, model_chain, states_chain
        return transd_ensemble

    def _productspacevector2model(
        self, x
    ):  # convert a combined product space model space vector to model vector in each state
        """
        Internal utility routine to convert a single vector in product state format to a list of vectors of differing length
        one per state. This routine is the inverse operation to routine '_modelvectors2productspace()'

        Inputs:
        x - float array or list : trans-D vectors in product space format. (length sum ndim[i], i=1,...,nstates)

        Returns:
        m - list of floats      : list of trans-D vectors one per state with format
                                  m[i][v[i]], (i=1,...,nstates) where i is the state and v[i] is a model vector in state i.

        """
        m = []
        kk = 1
        for k in range(self.nstates):
            m.append(x[kk : kk + self.ndims[k]])
            kk += self.ndims[k]
        return m

    def _modelvectors2productspace(
        self, m, states, nwalkers
    ):  # convert model space vectors in each state to product space vectors
        """
        Internal utility routine to convert a list of vectors of differing length one per state to a single vector in product state format.
        This routine is the inverse operation to routine '_productspacevector2model()' but over multiple walkers.

        Inputs:
        m - list of floats arrays      : list of trans-D vectors one per state with format
                                          m[i][v[i]], (i=1,...,nstates) where i is the state and v[i] is a vector in state i.
        states - nwalkers*int          : list of states for each walker/chain.
        nwalkers - int                 : number of walkers.

        Returns:
        x - float array or list : trans-D vectors in product space format. (length = nwalkers*(1 + sum ndim[i], i=1,...,nstates))

        """
        x = np.zeros((nwalkers, self.ps_ndim + 1))
        for j in range(nwalkers):
            x[j, 0] = states[j]
            x[j, 1:] = np.concatenate(([m[i][j] for i in range(self.nstates)]))
        return x

    def _productspace_log_prob(
        self,
        x,
        log_posterior,
        log_pseudo_prior,
        log_posterior_args,
        log_pseudo_prior_args,
    ):  # Calculate product space target PDF from posterior and pseudo-priors in each state
        """
        Internal utility routine to calculate the combined target density for product space vector.
        i.e. sum of log posterior + log pseudo prior density of all states
        here input vector is in product space format.

        Inputs:
        x - float array or list : trans-D vectors in product space format. (length = nwalkers*(1 + sum ndim[i], i=1,...,nstates))
        log_posterior()              : user supplied function to evaluate the log-posterior density for the ith state at location x.
                                       calling sequence log_posterior(x,i,*log_posterior_args)
        log_pseudo_prior()           : user supplied function to evaluate the log-pseudo-prior density for the ith state at location x.
                                       calling sequence log_posterior(x,i,*log_posterior_args).
                                       NB: must be normalized over respective state spaces.
        log_posterior_args - list    : user defined (optional) list of additional arguments passed to log_posterior. See calling sequence above.
        log_pseudo_prior_args - list : user defined (optional) list of additional arguments passed to log_pseudo_prior. See calling sequence above.


        Returns:
        x - float array or list : trans-D vectors in product space format. (length sum ndim[i], i=1,...,nstates)

        """
        if x[0] < -0.5 or x[0] >= self.nstates - 0.5:
            return -np.inf
        state = int(np.rint(x[0]))
        state = int(np.min((state, self.nstates - 1)))
        state = int(np.max((state, 0)))
        m = self._productspacevector2model(x)
        log_prob = log_posterior(m[state], state, *log_posterior_args)
        for i in range(self.nstates):
            if i != state:
                new = log_pseudo_prior(m[i], i, *log_pseudo_prior_args)
                # print(' i ',i,'\nm',m[i],'\nlog_pseudo',new)
                log_prob += new
        return log_prob

    def _flatten_extend(self, matrix):
        flat_list = []
        for row in matrix:
            flat_list.extend(row)
        return np.array(flat_list)


# Following the suggestion from Goodman & Weare (2010) we implement routines for auto_correlation calculations


def autocorr_gw2010(y, c=5.0):
    """Auto correlation utility routine following Goodman & Weare (2010)"""
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_fardal(y, c=5.0):
    """Auto correlation utility routine for improved auto correlation time estimate as per emcee notes
    see https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
    """
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def next_pow_two(n):
    """Auto correlation utility routine following Goodman & Weare (2010)"""
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    """Auto correlation utility routine following Goodman & Weare (2010)"""
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


def auto_window(taus, c):
    """Auto correlation utility routine for Automated windowing procedure following Sokal (1989)"""
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1
