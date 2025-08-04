"""Build an automatic pseudo-prior."""

from enum import StrEnum, auto
from typing import Any, Protocol

import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture

from ..samplers.per_state import run_mcmc_per_state
from .exceptions import InputError
from .types import MultiStateDensity, SampleableMultiStateDensity


class PseudoPriorBuilders(StrEnum):
    """Enum for available pseudo-prior builders."""

    GAUSSIAN_MIXTURE = auto()
    MEAN_COVARIANCE = auto()


class PseudoPriorBuilder(Protocol):
    """Protocol for pseudo-prior builder function."""

    def __call__(
        self,
        ensemble_per_state: list[np.ndarray],
        *args: Any,
        **kwargs: Any,
    ) -> SampleableMultiStateDensity:
        """
        Build a pseudo-prior function based on the provided parameters.

        Args:
            ensemble_per_state (list[np.ndarray]): List of ensembles for each state.  Each ensemble should be appropriately distributed.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns a callable pseudo-prior function.
        """
        ...


class GaussianMixturePseudoPrior:
    """Class for Gaussian mixture pseudo-prior."""

    def __init__(self, ensemble_per_state: list[np.ndarray], **kwargs):
        """
        Initialize the Gaussian mixture pseudo-prior.

        Parameters
        ----------
        ensemble_per_state : list of np.ndarray
            List of ensembles for each state.
        kwargs : dict
            Additional arguments for Gaussian mixture fitting.
        """
        self.gaussian_mixtures = [
            GaussianMixture(**kwargs).fit(ens) for ens in ensemble_per_state
        ]

    def __call__(self, x: np.ndarray, state: int) -> float:
        """Evaluate the log pseudo-prior density."""
        gmm = self.gaussian_mixtures[state]
        return float(gmm.score([x]))

    def draw_deviate(self, state: int) -> np.ndarray:
        """Draw a random deviate from the pseudo-prior for a given state."""
        gmm = self.gaussian_mixtures[state]
        return gmm.sample()[0][0]


class MeanCovariancePseudoPrior:
    """Class for mean and covariance pseudo-prior."""

    def __init__(self, ensemble_per_state: list[np.ndarray]):
        """
        Initialize the mean and covariance pseudo-prior.

        Parameters
        ----------
        ensemble_per_state : list of np.ndarray
            List of ensembles for each state.
        """
        self.rv_list = []
        for state_ensemble in ensemble_per_state:
            pseudo_covariances_ = np.cov(state_ensemble.T)
            pseudo_means_ = np.mean(state_ensemble.T, axis=1)
            rv = stats.multivariate_normal(mean=pseudo_means_, cov=pseudo_covariances_)
            self.rv_list.append(rv)

    def __call__(self, x: np.ndarray, state: int) -> float:
        """Evaluate the log pseudo-prior density."""
        rv = self.rv_list[state]
        return rv.logpdf(x)

    def draw_deviate(self, state: int) -> np.ndarray:
        """Draw a random deviate from the pseudo-prior for a given state."""
        rv = self.rv_list[state]
        return rv.rvs(size=1)[0]


pseudo_prior_factories: dict[PseudoPriorBuilders, PseudoPriorBuilder] = {
    PseudoPriorBuilders.GAUSSIAN_MIXTURE: GaussianMixturePseudoPrior,
    PseudoPriorBuilders.MEAN_COVARIANCE: MeanCovariancePseudoPrior,
}


def build_auto_pseudo_prior(
    pseudo_prior_type: PseudoPriorBuilders = PseudoPriorBuilders.GAUSSIAN_MIXTURE,
    *,
    ensemble_per_state: list[np.ndarray] | None = None,
    log_posterior: MultiStateDensity | None = None,
    sampling_args: dict[str, Any] = {},
    **builder_kwargs,
):
    """
    Build an automatic pseudo-prior function using a specified builder.

    Parameters
    ----------
    pseudo_prior_type : PseudoPriorBuilders, optional
        Type of pseudo-prior builder to use. Default is GAUSSIAN_MIXTURE.
    ensemble_per_state : list of np.ndarray, optional
        List of posterior samples for each state. If not provided, samples will be generated using MCMC.
    log_posterior : MultiStateDensity, optional
        Function evaluating the log-posterior for each state. Required if ensemble_per_state is not provided.
    sampling_args : dict, optional
        Arguments for MCMC sampling if ensemble_per_state is not provided.
    **builder_kwargs : dict
        Additional arguments passed to the pseudo-prior builder.

    Returns
    -------
    log_pseudo_prior : PseudoPrior
        Callable function to evaluate the log pseudo-prior at a given point and state.
    """

    if ensemble_per_state is None:  # generate samples for fitting
        if log_posterior is None:
            raise InputError(
                "log_posterior must be provided if ensemble_per_state is not supplied."
            )
        if any(
            k not in sampling_args
            for k in ["n_states", "n_dims", "n_walkers", "n_steps", "pos"]
        ):
            raise InputError(
                "sampling_args must contain 'n_states', 'n_dims', 'n_walkers', 'n_steps', and 'pos' keys."
            )

        n_states = sampling_args.pop("n_states")
        n_dims = sampling_args.pop("n_dims")
        n_walkers = sampling_args.pop("n_walkers")
        n_steps = sampling_args.pop("n_steps")
        pos = sampling_args.pop("pos")
        ensemble_per_state, _ = run_mcmc_per_state(
            n_states=n_states,
            n_dims=n_dims,
            n_walkers=n_walkers,
            n_steps=n_steps,
            pos=pos,
            log_posterior=log_posterior,
            **sampling_args,
        )

    log_pseudo_prior = pseudo_prior_factories[pseudo_prior_type](
        ensemble_per_state, **builder_kwargs
    )

    return log_pseudo_prior
