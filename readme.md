# pyTransC

![Python3](https://img.shields.io/badge/python-3.x-brightgreen.svg)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

_Python library for implementing TransC MCMC sampling_


This repository contains source code to implement three Trans-Conceptual MCMC sampling algorithms as described in the article 
[Sambridge, Valentine and Hauser (2025)](https://essopenarchive.org/users/841079/articles/1231492-trans-conceptual-sampling-bayesian-inference-with-competing-assumptions).


## Installation

```
pip install git+https://github.com/inlab-geo/pyTransC
```
## Documentation

This package of with a single class `TransC_Sampler` implementing three separate MCMC samplers across independent model states implemented as functions of the class

`run_product_space_sampler()` - implements a fixed dimension MCMC sampler over the product space of the states, and extracts a TransC/TransD ensemble. 

`run_state_jump_sampler()` - implements an RJ-MCMC style algorithm using pseudo-prior proposals and balance conditions. 

`run_ensemble_resampler()` - implements a single parameter Metropolis sampler over the state indicator variable. Requires posterior ensembles in each state to be precomputed.

Other utility functions include:

`run_mcmc_per_state()` - performs MCMC sampling within each state.

`build_auto_pseudo_prior()` - fits a mixture model to posterior ensembles in each state to act as a pseudo-prior function.

`get_transc_samples()` - creates posterior TransC/TransD ensemble from results of any sample.

Here is the docstring of the function `run_ensemble_resampler()`:

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
        state_chain_tot - nsamples*int                : array of states visited along the trans-D chain.
        alg - string                                  : string defining the sampler method used.


        Notes:
        The input posterior samples and log posterior values in each state can be either be calcuated using utility routine 'run_mcmc_per_state', or provided by the user.
        The input log values of pseudo prior samples in each state can be either be calcuated using utility routine 'run_fitmixture', or provided by the user.

        """

## Example

```python
import numpy as np
from pyTransC import TransC_Sampler         # TransC library class
```
Detailed examples of showing implementation of all three samplers can be found in

[`examples/Gaussians`](./examples/Gaussians/) - Sampling across unnormalised Mulit-dimensional Gaussians with all three samplers.

[`examples/AirborneEM`](./examples/AirborneEM) - Ensemble Sampler applied to Airborne EM data.

## Licensing
`pyTransC` is released as BSD-2-Clause licence.

## Citations and Acknowledgments

> *Sambridge, M., Valentine, A. & Hauser, J., 2025. Trans-Conceptual Sampling: Bayesian Inference With Competing Assumptions, JGR Solid Earth, submitted.*
