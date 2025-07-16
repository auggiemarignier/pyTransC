"""
Trans conceptual McMC sampler class.

A class that is used to perform a Metropolis random walk across a Trans-C model comprising of independent states.
Calculates relative Marginal Likelihoods/evidences between states and/or an ensemble of Trans-C samples.
Each state may have arbitrary dimension, model parameter definition and Likelihood function.

For a description of Trans-C and its relation to Trans-D sampling see

https://essopenarchive.org/users/841079/articles/1231492-trans-conceptual-sampling-bayesian-inference-with-competing-assumptions
"""

import multiprocessing
import os

import numpy as np

from .analysis.integration import run_ens_mcint
from .analysis.laplace import run_laplace_evidence_approximation
from .analysis.samples import get_transc_samples
from .analysis.visits import get_visits_to_states
from .exceptions import InputError
from .samplers.ensemble_resampler import run_ensemble_resampler
from .samplers.product_space import run_product_space_sampler
from .samplers.state_jump import run_state_jump_sampler
from .utils.auto_pseudo import build_auto_pseudo_prior
from .utils.mixture import fit_mixture, log_pseudo_prior_from_mixtures

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    def worker():
        """Worker function to test multiprocessing setup."""
        print("Worker process")

    p = multiprocessing.Process(target=worker)
    p.start()
    p.join()

os.environ["OMP_NUM_THREADS"] = (
    "1"  # turn off automatic parallelisation so that we can use emcee parallelization
)


class TransC_Sampler:  # Independent state MCMC parameter class
    """
    Trans-C McMC sampler class.

    A class that is used to perform a Metropolis random walk across independent states.
    Calculates relative Marginal Likelihoods/evidences between states and/or an ensemble of trans-C samples.
    Each state may have arbitrary dimension, model parameter definition and Likelihood function.

    Five alternate algorithms are available:
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
           Implemented with function `run_state_jump_sampler()`. Creates self.alg = 'TransC-state-jump-sampler'.

        3) Trans-C sampling across previously calculated posterior ensembles in each state.
           User may supply posterior ensembles of any size (one per state) with log-posterior and log-Pseudo-prior densities (normalized) calculated at sample locations.
           Alternatively, ensembles and log-Pseudo-prior values may be igenerated internally with class functions 'run_mcmc_per_state()' and `run_fitmixture()'.
           May be used in serial or parallel modes.
           Advantages - makes use of previously calculated posterior ensembles calculated offline, Automatic estimation of Pseudo-Prior;
           Disadvantages - depends on quality of users/generated ensemble.
           Implemented with function `run_ensemble_resampler()`. Creates self.alg = 'TransC-ensemble-resampler'.

        4) Relative evidence/marginal Likelihood calculation using Monte Carlo Integration over each state.
           User may supply posterior ensembles of any size (one per state) with log-posterior and log-Pseudo-prior densities (normalized) calculated at sample locations.
           Alternatively, ensembles and log-Pseudo-prior values may be igenerated internally with class functions 'run_mcmc_per_state()' and `run_fitmixture()'.
           Advantages - makes use of previously calculated posterior ensembles; Disadvantages - depends on quality of users/generated ensemble; no Markov chain output
           to inspect. Implemented with function `run_ens_mcint()`. Creates self.alg = 'TransC-integration'.

        5) Relative evidence/marginal Likelihood calculation using Laplace Integration over each state.
           User may supply log-posterior function and MAP models. If optimize=True Map models are starting points for maximization of log_posterior.
           User may optionally supply posterior ensembles of any size (one per state) with log-posterior and log-Pseudo-prior densities (normalized) calculated at sample locations.
           Advantages - A simple Gaussian approximation to Posterior PDF about the MAP models in each state.
           Implemented with `run_laplace_evidence_approximation()' creates self.laplace_log_marginal_likelihood, self.laplace_map_models_per_state, self.laplace_hessians


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
        Deprecated: use samplers.per_state.run_mcmc_per_state() instead.

        Maintained for backwards compatibility with old code.
        """
        from .samplers.per_state import run_mcmc_per_state

        samples_per_state, log_posterior_per_state = run_mcmc_per_state(
            self.nstates,
            self.ndims,
            nwalkers,
            nsteps,
            pos,
            log_posterior,
            log_posterior_args=log_posterior_args,
            discard=discard,
            thin=thin,
            auto_thin=autothin,
            seed=seed,
            parallel=parallel,
            n_processors=nprocessors,
            progress=progress,
            skip_initial_state_check=skip_initial_state_check,
            io=io,
            verbose=verbose,
            **kwargs,
        )
        self.ensemble_per_state = samples_per_state
        self.log_posterior_ens = log_posterior_per_state
        self.run_per_state = True  # set flag to indicate work done
        self.nsamples = [len(samples) for samples in samples_per_state]

        return samples_per_state, log_posterior_per_state

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
        if not self.run_per_state:  # if the input ensemble is external then store it
            self.ensemble_per_state = ensemble  # store input ensemble
            self.log_posterior_ens = log_posterior_ens  # store input ensemble
            nsamples = []  # calculate number of samples per state for ensemble
            for state in range(self.nstates):
                nsamples.append(len(ensemble[state]))
            self.nsamples = nsamples

        self.gm = fit_mixture(
            self.nstates,
            self.ensemble_per_state,
            self.log_posterior_ens,
            verbose,
            **kwargs,
        )
        self.run_fit = True

        return log_pseudo_prior_from_mixtures(
            self.gm, ensemble, return_pseudo_prior_func
        )

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
        if self.run_per_state and not forcesample:
            print(" We are using previously generated samples for fitting pseudo-prior")
            ensemble_per_state = self.ensemble_per_state
            log_posterior_ens = self.log_posterior_ens

        return build_auto_pseudo_prior(
            self.nstates,
            self.ndims,
            pos,
            log_posterior,
            log_posterior_args=log_posterior_args,
            ensemble_per_state=ensemble_per_state,
            log_posterior_ens=log_posterior_ens,
            discard=discard,
            thin=thin,
            autothin=autothin,
            parallel=parallel,
            n_samples=nsamples,
            n_walkers=nwalkers,
            return_log_pseudo=return_log_pseudo,
            progress=progress,
            verbose=verbose,
            fitmeancov=fitmeancov,
            **kwargs,
        )

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
        MCMC sampler over independent states using emcee fixed dimension sampler over trans-C product space.

        Inputs:
        nwalkers - int               : number of random walkers used by product_space sampler.
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
        self.alg = "TransC-product-space"

        sampler = run_product_space_sampler(
            nwalkers,
            nsteps,
            self.nstates,
            self.ndims,
            pos,
            pos_state,
            log_posterior,
            log_pseudo_prior,
            log_posterior_args=log_posterior_args,
            log_pseudo_prior_args=log_pseudo_prior_args,
            seed=seed,
            parallel=parallel,
            n_processors=nprocessors,
            progress=progress,
            suppress_warnings=suppresswarnings,
            my_pool=mypool,
            skip_initial_state_check=skip_initial_state_check,
            **kwargs,
        )

        # In the mypool=True case this will be a list of samplers
        # but I don't think it is correctly handled in downstream functions
        # e.g. get_visits_to_states
        self.productspace_sampler = sampler
        self.nwalkers = nwalkers
        self.nsteps = nsteps

    def run_state_jump_sampler(  # Independent state MCMC sampler on product space with proposal equal to pseudo prior
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

        Calculates Markov chain across states for state jump sampler

        Inputs:
        nwalkers - int               : number of random walkers used by state jump sampler.
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
        nwalkers - int                                : number of random walkers used by state jump sampler.
        state_chain - nwalkers*nsteps*int             : list of states visited along the trans-C chain.
        state_chain_tot - nwalkers*nsteps*int         : array of cumulative number of visits to each state along the chains.
        model_chain - floats                          : list of trans-C sample along chain.
        alg - string                                  : string defining the sampler method used.

        Notes:
        A simple Metropolis-Hastings MCMC algorithm is used and applied to the product space formulation. Here moves between states are assumed to only perturb the state variable, k-> k'.
        This means that one only needs to generate a new model in state k' from the pseudo-prior of k'. The M-H condition then only involves the current model in state k and the new model in state k',
        with the acceptance criterion then equal to the ratio of the posteriors multiplied by the ratio of the normalized pseudo-priors.
        For within state moves the algorithm becomes normal M-H using a user supplied proposal function to generate new deviates within state k. The user can define this as relative to current model,
        or according to a prescribed PDF within the respective state, e.g. the pseudo-prior again. An independent user supplied proposal function is provided for flexibility.

        """
        self.alg = "TransC-state-jump-sampler"

        (
            model_chain,
            state_chain,
            state_chain_tot,
            accept_within,
            prop_within,
            accept_between,
            prop_between,
        ) = run_state_jump_sampler(
            nwalkers,
            nsteps,
            self.nstates,
            self.ndims,
            pos,
            pos_state,
            log_posterior,
            log_pseudo_prior,
            log_proposal,
            log_posterior_args=log_posterior_args,
            log_pseudo_prior_args=log_pseudo_prior_args,
            log_proposal_args=log_proposal_args,
            prob_state=prob_state,
            seed=seed,
            parallel=parallel,
            n_processors=nprocessors,
            progress=progress,
            suppress_warnings=suppresswarnings,
            verbose=verbose,
        )

        self.state_chain_tot = np.swapaxes(state_chain_tot, 0, 1)
        self.state_chain = state_chain.T
        self.model_chain = model_chain
        self.accept_within_per_walker = accept_within / prop_within
        self.accept_between_per_walker = accept_between / prop_between
        self.accept_within = 100 * np.mean(self.accept_within_per_walker)
        self.accept_between = 100 * np.mean(self.accept_between_per_walker)
        self.nwalkers = nwalkers
        self.nsteps = nsteps

    def run_ensemble_resampler(  # Independent state Marginal Likelihoods from pre-computed posterior and pseudo prior ensembles
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
        nwalkers - int                                                       : number of random walkers used by ensemble resampler.
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
        state_chain_tot - nsamples*int                : array of states visited along the trans-C chain.
        alg - string                                  : string defining the sampler method used.


        Notes:
        The input posterior samples and log posterior values in each state can be either be calcuated using utility routine 'run_mcmc_per_state', or provided by the user.
        The input log values of pseudo prior samples in each state can be either be calcuated using utility routine 'run_fitmixture', or provided by the user.

        """

        state_chain, state_chain_tot, accept_between = run_ensemble_resampler(
            nwalkers,
            nsteps,
            self.ndims,
            log_posterior_ens=log_posterior_ens,
            log_pseudo_prior_ens=log_pseudo_prior_ens,
            seed=seed,
            parallel=parallel,
            n_processors=nprocessors,
            state_proposal_weights=stateproposalweights,
            progress=progress,
        )

        self.alg = "TransC-ensemble-resampler"
        self.state_chain_tot = np.swapaxes(state_chain_tot, 0, 1)
        self.state_chain = state_chain.T
        self.accept_within_per_walker = 1.0 * np.ones(nwalkers)
        self.accept_between_per_walker = accept_between / nsteps
        self.accept_within = 100 * np.mean(self.accept_within_per_walker)
        self.accept_between = 100 * np.mean(self.accept_between_per_walker)
        self.nwalkers = nwalkers
        self.nsteps = nsteps

    def run_ens_mcint(  #  Marginal Likelihoods from Monte Carlo integration
        self,
        log_posterior_ens=None,
        log_pseudo_prior_ens=None,
        return_marginallikelihoods=False,
    ):
        """
        Utility routine to perform MCMC sampler over independent states using numerical integration.

        This routine is a faster alternate to running a Markov chain across the ensembles, which is carried out by run_is_ensemble_resampler.
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

        if not self.run_fit:  # mixture fitting has not been performed and so we need input ensembles, so check that we have them
            if (log_pseudo_prior_ens is None) and (log_posterior_ens is not None):
                raise InputError(
                    msg=" In function run_ens_mcint: Ensemble probabilities provided as argument without pseudo-prior probabilities"
                )

            if (log_pseudo_prior_ens is not None) and (log_posterior_ens is None):
                raise InputError(
                    msg=" In function run_ens_mcint: Pseudo-prior probabilities provided as argument without ensemble probabilities"
                )

            if (log_pseudo_prior_ens is None) and (log_posterior_ens is None):
                raise InputError(
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

        self.relative_marginal_likelihoods, self.ens_mc_samples = run_ens_mcint(
            self.nstates,
            log_posterior_ens,
            log_pseudo_prior_ens,
        )
        if return_marginallikelihoods:
            return self.relative_marginal_likelihoods

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
        Function to perform Laplace integration for evidence approximation within each state, using either an input log-posterior function, or posterior ensembles.

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
        laplace_log_marginal_likelihoods - nstates*float : list of log-evidence/marginal Likelihoods for each state.
        laplace_map_models_per_state - nstates*floats : list of updated MAP models for each state.m if optimize=True.
        laplace_map_log_posteriors - nstates*float : list of log-posteriors at MAP models for each state.
        laplace_hessians - nstates*NxN : list of negative inverse Hessians if posterior function supplied.

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

        (
            self.laplace_hessians,
            self.laplace_map_models_per_state,
            self.laplace_map_log_posteriors,
            self.laplace_log_marginal_likelihoods,
        ) = run_laplace_evidence_approximation(
            self.nstates,
            self.ndims,
            log_posterior,
            map_models,
            log_posterior_args=log_posterior_args,
            ensemble_per_state=ensemble_per_state,
            log_posterior_ens=log_posterior_ens,
            verbose=verbose,
            optimize=optimize,
        )

    def get_visits_to_states(  # calculate evolution of relative visits to each state along chain
        self,
        discard=0,
        thin=1,
        normalize=False,
        flat=False,
        walker_average="median",
        return_samples=False,
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
        if self.alg == "TransC-product-space":
            self.state_chain = None
            self.state_chain_tot = None
        else:
            self.productspace_sampler = None
        (
            out,
            samples,
            self.relative_marginal_likelihoods,
            self.state_changes_perwalker,
            self.total_state_changes,
            self.acceptance_rate_between_states,
            self.autocorr_time_for_between_state_jumps,
            *extra_product_space_outputs,
        ) = get_visits_to_states(
            self.alg,
            self.nstates,
            self.nwalkers,
            self.nsteps,
            self.state_chain_tot,
            self.state_chain,
            product_space_sampler=self.productspace_sampler,
            discard=discard,
            thin=thin,
            normalize=normalize,
            flat=flat,
            walker_average=walker_average,
        )

        if self.alg == "TransC-product-space":
            (
                self.acceptance_rate_per_walker,
                self.acceptance_rate,
                self.mean_autocorr_time,
                self.max_autocorr_time,
            ) = extra_product_space_outputs

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
    ):  # generate a trans-c ensemble from either TransC-ens or TransC-ps samplers
        """
        Utility routine to retrieve list of trans-C model space samples, previously calculated by either run_is_ensemble_resampler(),run_product_space_sampler() or run_ens_mcint() or run_state_jump_sampler.

        For algorithms TransC-ensemble-resampler and TransC-integration the input variable 'ntd_samples' determines the number of trans-C model space samples generated and then returned.
        For algorithms TransC-product-space and TransC-state-jump-sampler the number of trans-C model space samples is determined by the original sampler and modified by chain thinning (see `discard` and `thin` parameters).

        Inputs:
        ntd_samples - int           : number of trans-C samples to generate [for algs 'TransC-integration' or 'TransC-ensemble-resampler'].
        discard - int               : number of output samples to discard, also known as `burnin' (default = 0)) [only relevant if algs = 'TransC-product-space' or 'TransC-state-jump-sampler'].
        thin - int                  : frequency of output samples in output chains to accept (default = 1, i.e. all) [only relevant if algs = 'TransC-product-space' or 'TransC-state-jump-sampler'].
        returnchains - bool         : switch to return states of each trans_d sample returned. (Default = False)
        verbose - bool              : switch to print some diagnostic info to standard out.


        Returns:
        transd_ensemble - list      : list of trans-C samples ordered by state.
                                      if flat=True, format is [state,i],i=1,...,n(state); where n(state) is the number of models generated in state i.
                                      otherwise format is [state,walker,i], where samples are also separated by their walker.
                                      if alg is `TransC-product-space` or `TransC-state-jump-sampler` size of ensemble returned depends on values of discard and flat.
                                      if alg is `TransC-ensemble-resampler` or `TransC-integration`,  size of ensemble returned is given by ntd_samples.
        model_chain - floats        : list of trans-C samples. For ensemble generated by `run_product_space_sampler`, number determined by
                                      discard and thin (only if returnchains = True).
        states_chain - ints. : list of states of trans-C samples (only if returnchains = True).

        """
        # if(not hasattr(self, 'relative_marginal_likelihoods')): # need to call get_visits_to_states for marginal Likelihoods

        # Hacky way to ensure that attributes are set up correctly
        _attrs = [
            "nsamples",
            "state_chain",
            "model_chain",
            "relative_marginal_likelihoods",
            "ensemble_per_state",
            "productspace_sampler",
        ]
        for _attr in _attrs:
            if not hasattr(self, _attr):
                print("Setting attribute", _attr, "to None")
                setattr(self, _attr, None)

        _return = get_transc_samples(
            self.alg,
            self.nstates,
            self.ndims,
            self.nsamples,
            self.state_chain,
            self.model_chain,
            self.relative_marginal_likelihoods,
            self.productspace_sampler,
            ntd_samples=ntd_samples,
            discard=discard,
            thin=thin,
            ensemble_per_state=ensemble_per_state,
            flat=flat,
        )

        if returnchains:
            transd_ensemble, model_chain, states_chain = _return
            return transd_ensemble, model_chain, states_chain
        else:
            transd_ensemble = _return
            return transd_ensemble
