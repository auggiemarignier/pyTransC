"""A helper module for the emcee sampler."""

import inspect
from collections.abc import Callable
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from typing import Any

from emcee import EnsembleSampler

from ..utils.types import FloatArray


def perform_sampling_with_emcee(
    log_prob_func: Callable[[FloatArray], float],
    n_walkers: int,
    n_steps: int,
    initial_state: FloatArray,
    **kwargs,
) -> EnsembleSampler:
    """Perform MCMC sampling using emcee.

    Creates, runs and returns an emcee EnsembleSampler instance.
    """

    run_kwargs, initialisation_kwargs, remaining_kwargs = _extract_run_kwargs(kwargs)

    if remaining_kwargs.get("parallel", False):
        n_processors = remaining_kwargs.get("n_processors", 1)
        if n_processors == 1:
            n_processors = cpu_count()
        initialisation_kwargs["pool"] = Pool(processes=n_processors)

    sampler = EnsembleSampler(
        nwalkers=n_walkers,
        ndim=initial_state.shape[1],
        log_prob_fn=log_prob_func,
        **initialisation_kwargs,
    )
    sampler.run_mcmc(initial_state, n_steps, **run_kwargs)
    return sampler


def _extract_run_kwargs(
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Separate kwargs to be passed to the EnsembleSampler initialisation and EnsembleSampler.run_mcmc.

    kwargs for run_mcmc are passed straight to the EnsembleSampler.sample method.
    """
    _run_kwargs = _kwargs_from_instance_method(
        inspect.signature(EnsembleSampler.sample)
    )

    _init_kwargs = _kwargs_from_instance_method(
        inspect.signature(EnsembleSampler.__init__)
    )

    run_kwargs = {k: kwargs.pop(k) for k in _run_kwargs if k in kwargs}
    init_kwargs = {k: kwargs.pop(k) for k in _init_kwargs if k in kwargs}

    return run_kwargs, init_kwargs, kwargs


def _kwargs_from_instance_method(sig: inspect.Signature) -> list[str]:
    """Extract kwargs from an instance method signature.

    Explicitly excludes `self` just in case.
    """
    return [
        name
        for name, param in sig.parameters.items()
        if param.kind
        in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and name != "self"
    ]
