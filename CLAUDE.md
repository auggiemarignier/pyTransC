# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyTransC is a Python library implementing Trans-Conceptual MCMC sampling algorithms for Bayesian inference across competing model assumptions. The library provides three main sampling approaches for handling uncertainty across different conceptual states/models.

## Development Commands

### Testing
```bash
python -m pytest tests/
python -m pytest tests/test_pytransc/test_samplers/test_product_space.py  # Run specific test file
```

### Linting and Code Quality
```bash
python -m ruff check                    # Check code style and lint issues
python -m ruff check --fix             # Auto-fix linting issues where possible
python -m ruff format                   # Format code (if configured)
```

### Installation for Development
```bash
pip install -e .                       # Install in editable mode
pip install -e ".[dev]"                # Install with dev dependencies
pip install -e ".[notebooks]"          # Install with Jupyter dependencies
```

## Architecture

### Core Module Structure
- `src/pyTransC/samplers/` - Main sampling algorithms
  - `product_space.py` - Fixed-dimensional sampling over product space of states
  - `state_jump.py` - RJ-MCMC style algorithm with pseudo-prior proposals
  - `ensemble_resampler.py` - Single parameter Metropolis over state indicator
  - `per_state.py` - Independent MCMC sampling within each state
  
- `src/pyTransC/analysis/` - Post-processing and analysis tools
  - `laplace.py` - Marginal likelihood estimation via Laplace approximation
  - `samples.py` - Sample extraction and processing
  - `visits.py` - State visit analysis and acceptance rates
  - `integration.py` - Integration methods for evidence calculation

- `src/pyTransC/utils/` - Utility functions and types
  - `auto_pseudo.py` - Automatic pseudo-prior construction using mixture models
  - `autocorr.py` - Autocorrelation analysis utilities
  - `types.py` - Type definitions for multi-dimensional arrays
  - `exceptions.py` - Custom exception classes

### Key Concepts
- **Trans-Conceptual Sampling**: Bayesian inference across models with different assumptions/states
- **Pseudo-priors**: Bridge distributions enabling transitions between conceptual states
- **Product Space**: Joint parameter space across all model states
- **State Indicator**: Variable tracking which conceptual model is active

### Main Entry Points
All samplers are functions (not classes) that can be imported directly:
```python
from pytransc.samplers import (
    run_product_space_sampler,
    run_state_jump_sampler, 
    run_ensemble_resampler,
    run_mcmc_per_state
)
```

### Coding Standards
- Uses Ruff for linting with NumPy docstring convention
- Type hints required for function parameters and returns
- Maximum complexity limit of 25 (McCabe)
- Python >=3.11 required
- Dependencies: NumPy, SciPy, scikit-learn, emcee, matplotlib, arviz

### Example Structure
The `examples/` directory contains complete workflows:
- `Gaussians/` - Basic examples with multi-dimensional Gaussians for all three samplers
- `AirborneEM/` - Real-world geophysical application with airborne electromagnetic data
- Each example includes Jupyter notebooks demonstrating the full analysis pipeline