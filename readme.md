# pytransc

![Python3](https://img.shields.io/badge/python-3.x-brightgreen.svg)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

_Python library for implementing TransC MCMC sampling_


This repository contains source code to implement three Trans-Conceptual MCMC sampling algorithms as described in the article 
[Sambridge, Valentine and Hauser (2025)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JB030470).


## Installation

```
pip install git+https://github.com/inlab-geo/pytransc
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
       Run MCMC sampler with direct jumps between states of different states.

           This function implements trans-conceptual MCMC using a Metropolis-Hastings
           algorithm that can propose jumps between states with different numbers of
           parameters. Between-state moves use the pseudo-prior as the proposal, while
           within-state moves use a user-defined proposal function.

           Parameters
           ----------
           n_walkers : int
               Number of random walkers used by the state jump sampler.
           n_steps : int
               Number of MCMC steps required per walker.
           n_states : int
               Number of independent states in the problem.
           n_dims : list of int
               List of parameter dimensions for each state.
           start_positions : list of FloatArray
               Starting parameter positions for each walker. Each array should contain
               the initial parameter values for the corresponding starting state.
           start_states : list of int
               Starting state indices for each walker.
           log_posterior : MultiStateDensity
               Function to evaluate the log-posterior density at location x in state i.
               Must have signature log_posterior(x, state) -> float.
           log_pseudo_prior : SampleableMultiStateDensity
               Object with methods:
               - __call__(x, state) -> float: evaluate log pseudo-prior at x for state
               - draw_deviate(state) -> FloatArray: sample from pseudo-prior for state
               Note: Must be normalized over respective state spaces.
           log_proposal : ProposableMultiStateDensity
               Object with methods:
               - propose(x_current, state) -> FloatArray: propose new x in state
               - __call__(x, state) -> float: log proposal probability (for MH ratio)
           prob_state : float, optional
               Probability of proposing a state change per MCMC step. Otherwise,
               a parameter change within the current state is proposed. Default is 0.1.
           seed : int, optional
               Random number seed for reproducible results. Default is 61254557.
           parallel : bool, optional
               Whether to use multiprocessing to parallelize over walkers. Default is False.
           n_processors : int, optional
               Number of processors to use if parallel=True. Default is 1.
           progress : bool, optional
        
        
    Returns
    -------
    MultiWalkerStateJumpChain
        Chain results containing state sequences, model parameters, proposal
        acceptance rates, and diagnostics for all walkers.

    Notes
    -----
    The algorithm uses a Metropolis-Hastings sampler with two types of moves:

    1. **Between-state moves** (probability `prob_state`):
       - Propose a new state uniformly at random
       - Generate new parameters from the pseudo-prior of the proposed state
       - Accept/reject based on posterior and pseudo-prior ratios

    2. **Within-state moves** (probability `1 - prob_state`):
       - Use the user-defined proposal function to generate new parameters
       - Accept/reject using standard Metropolis-Hastings criterion

    The pseudo-prior must be normalized for the between-state acceptance
    criterion to satisfy detailed balance.
    
    """

## Example

```python
import numpy as np
from pytransc.samplers import run_state_jump_sampler
```
Detailed examples of showing implementation of all three samplers can be found in

[`examples/Gaussians`](./examples/Gaussians/) - Sampling across unnormalised Mulit-dimensional Gaussians with all three samplers.

[`examples/AirborneEM`](./examples/AirborneEM) - Ensemble Sampler applied to Airborne EM data.

## Licensing
`pytransc` is released as BSD-2-Clause licence.

## Citations and Acknowledgments

> *Sambridge, M., Valentine, A. & Hauser, J., 2025. Trans-Conceptual Sampling: Bayesian Inference With Competing Assumptions, JGR Solid Earth, Volume 130, Issue 8, 17 August 2025, e2024JB030470.*





