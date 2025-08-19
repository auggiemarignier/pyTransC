"""Analysis tools for trans-dimensional MCMC results.

This module provides utilities for analysing the output of trans-dimensional
MCMC samplers, including:

- Marginal likelihood estimation via Laplace approximation
- State visit analysis and acceptance rate calculations
- Sample extraction and post-processing
- Integration methods for evidence calculation

The analysis tools are designed to work with the output from any of the
sampling algorithms in the pytransc.samplers module.
"""

from .laplace import run_laplace_evidence_approximation

__all__ = [
    "run_laplace_evidence_approximation",
]
