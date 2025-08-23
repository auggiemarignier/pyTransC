# Parallelism in pyTransC

This document provides comprehensive guidance on using parallelism in pyTransC, covering the two-level parallelism architecture, pool compatibility, and best practices for high-performance computing environments.

## Overview

pyTransC implements a **two-level parallelism architecture** to enable efficient parallel execution across multiple sampling approaches:

### `run_mcmc_per_state` Parallelization:
1. **State-level parallelism**: Distributes independent states across processes
2. **Walker-level parallelism**: Distributes emcee walkers within each state

### `run_ensemble_resampler` Parallelization:
1. **Walker-level parallelism**: Distributes ensemble walkers across processes
2. **State-level parallelism**: Reserved for future enhancement

This architecture solves the multiprocessing daemon process limitation that prevents nested parallelism and provides consistent pool interfaces across different sampling methods.

## Daemon Process Problem & Solution

### The Problem
Standard `multiprocessing.Pool` creates daemon processes that **cannot spawn child processes**. This prevents combining state-level and walker-level parallelism:

```python
# This FAILS with daemon process error:
with multiprocessing.Pool() as state_pool:  # Creates daemon processes
    # Each state process tries to create its own pool
    with multiprocessing.Pool() as walker_pool:  # FAILS - daemon can't spawn children
        # ... parallel walker execution
```

### The Solution
pyTransC uses **non-daemon processes** and **pool configuration serialization** to enable true two-level parallelism:

```python
# This WORKS:
with ProcessPoolExecutor() as state_pool:  # Creates non-daemon processes  
    # Pool configuration is serialized and recreated in each state process
    with ProcessPoolExecutor() as walker_pool:  # WORKS - no pickling issues
        # ... parallel walker execution
```

### Implementation Details

**Key Innovation**: Instead of passing pool objects between processes (which causes pickling issues), pyTransC:
1. **Serializes pool configuration** (type and parameters) instead of pool objects
2. **Recreates pools** within each state process using the configuration
3. **Uses fork start method** to minimize pickling requirements
4. **Automatically manages pool lifecycle** with proper cleanup

**Multiprocessing Start Method**:
```python
# Set at module import to avoid pickling issues
import multiprocessing
try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    # Already set, ignore
    pass
```

## Pool Compatibility Matrix

### State-Level Pools

| Pool Type | Compatibility | Notes |
|-----------|---------------|-------|
| `ProcessPoolExecutor` | ✅ **Recommended** | Non-daemon, full multiprocessing |
| `multiprocessing.Process` | ✅ Good | Manual process management |
| `schwimmbad.MPIPool` | ✅ Excellent | Best for HPC/cluster environments |
| `schwimmbad.SerialPool` | ✅ Good | Sequential fallback |
| `schwimmbad.MultiPool` | ✅ Good | Automatic pool selection |
| `multiprocessing.Pool` | ❌ **Fails** | Creates daemon processes |
| Custom pools | ✅ Depends | Must implement `map()` method |

### Walker-Level Pools (within each state / ensemble resampler)

| Pool Type | Compatibility | Notes |
|-----------|---------------|-------|
| `ProcessPoolExecutor` | ✅ **Recommended** | Non-daemon, consistent architecture |
| `ThreadPoolExecutor` | ✅ Good | Lower overhead, shared memory |
| `multiprocessing.Pool` | ✅ Good | Works when parent is non-daemon |
| `schwimmbad` pools | ✅ Excellent | Full compatibility |
| Custom pools | ✅ Depends | Must implement `map()` method |

**Note**: All walker-level pools work for both `run_mcmc_per_state` (emcee_pool) and `run_ensemble_resampler` (walker_pool).

### External Library Requirements

| Library | Minimum Version | Purpose |
|---------|----------------|---------|
| `schwimmbad` | ≥0.3.0 | Advanced pool types, MPI support |
| `mpi4py` | ≥3.0.0 | MPI communication (required for MPIPool) |
| `concurrent.futures` | Built-in | Standard ProcessPoolExecutor/ThreadPoolExecutor |

## Configuration Examples

### 1. State-Level Parallelism Only

```python
from concurrent.futures import ProcessPoolExecutor
from pytransc.samplers import run_mcmc_per_state

# Parallel states, sequential walkers within each state
with ProcessPoolExecutor(max_workers=4) as state_pool:
    ensembles, log_probs = run_mcmc_per_state(
        n_states=8,
        n_dims=[3, 2, 4, 1, 2, 3, 2, 1],
        n_walkers=32,
        n_steps=1000,
        pos=initial_positions,
        log_posterior=my_log_posterior,
        state_pool=state_pool
    )
```

### 2. Walker-Level Parallelism Only

```python
from concurrent.futures import ProcessPoolExecutor

# Sequential states, parallel walkers within each state
with ProcessPoolExecutor(max_workers=8) as walker_pool:
    ensembles, log_probs = run_mcmc_per_state(
        n_states=4,
        n_dims=[3, 2, 4, 1],
        n_walkers=32,
        n_steps=1000,
        pos=initial_positions,
        log_posterior=my_log_posterior,
        emcee_pool=walker_pool
    )
```

### 3. Two-Level Parallelism

```python
from concurrent.futures import ProcessPoolExecutor

# Both state and walker parallelism
with ProcessPoolExecutor(max_workers=4) as state_pool, \
     ProcessPoolExecutor(max_workers=4) as walker_pool:
    
    ensembles, log_probs = run_mcmc_per_state(
        n_states=8,
        n_dims=[3, 2, 4, 1, 2, 3, 2, 1],
        n_walkers=32,
        n_steps=1000,
        pos=initial_positions,
        log_posterior=my_log_posterior,
        state_pool=state_pool,      # 4 processes for states
        emcee_pool=walker_pool      # 4 processes per state for walkers
    )
    # Total processes: 4 (states) × 4 (walkers) = 16 processes
```

### 4. HPC/MPI Configuration

```python
from schwimmbad import MPIPool
from concurrent.futures import ProcessPoolExecutor

# MPI across nodes, multiprocessing within nodes
with MPIPool() as mpi_pool:
    if not mpi_pool.is_master():
        mpi_pool.wait()
        sys.exit(0)
        
    # Use local multiprocessing for walker parallelism
    with ProcessPoolExecutor(max_workers=8) as walker_pool:
        ensembles, log_probs = run_mcmc_per_state(
            n_states=16,  # Distributed across MPI processes
            n_dims=dims_list,
            n_walkers=32,
            n_steps=1000,
            pos=initial_positions,
            log_posterior=my_log_posterior,
            state_pool=mpi_pool,        # MPI across nodes
            emcee_pool=walker_pool      # Local multiprocessing
        )
```

### 5. Automatic Internal Parallelism

```python
# Let pyTransC create internal ProcessPoolExecutors
ensembles, log_probs = run_mcmc_per_state(
    n_states=8,
    n_dims=[3, 2, 4, 1, 2, 3, 2, 1],
    n_walkers=32,
    n_steps=1000,
    pos=initial_positions,
    log_posterior=my_log_posterior,
    n_state_processors=4,    # Creates internal ProcessPoolExecutor
    parallel=True,           # Creates internal emcee pools
    n_processors=4
)
```

## Ensemble Resampler Parallelization

The `run_ensemble_resampler` function provides walker-level parallelization through the `walker_pool` parameter, enabling efficient distribution of ensemble walker execution across processes or threads.

### Basic Ensemble Resampler Configuration

```python
from concurrent.futures import ProcessPoolExecutor
from pytransc.samplers import run_ensemble_resampler

# Sequential execution (baseline)
results = run_ensemble_resampler(
    n_walkers=16,
    n_steps=2000,
    n_states=4,
    n_dims=[2, 3, 4, 5],
    log_posterior_ens=posterior_ensembles,
    log_pseudo_prior_ens=pseudo_prior_ensembles,
    parallel=False
)
```

### Walker-Level Parallelism with ProcessPoolExecutor

```python
# Parallel walker execution
with ProcessPoolExecutor(max_workers=4) as walker_pool:
    results = run_ensemble_resampler(
        n_walkers=16,
        n_steps=2000,
        n_states=4,
        n_dims=[2, 3, 4, 5],
        log_posterior_ens=posterior_ensembles,
        log_pseudo_prior_ens=pseudo_prior_ensembles,
        walker_pool=walker_pool  # Distribute walkers across 4 processes
    )
```

### Walker-Level Parallelism with ThreadPoolExecutor

```python
from concurrent.futures import ThreadPoolExecutor

# Thread-based walker parallelism (lower overhead)
with ThreadPoolExecutor(max_workers=4) as walker_pool:
    results = run_ensemble_resampler(
        n_walkers=16,
        n_steps=2000,
        n_states=4,
        n_dims=[2, 3, 4, 5],
        log_posterior_ens=posterior_ensembles,
        log_pseudo_prior_ens=pseudo_prior_ensembles,
        walker_pool=walker_pool  # Thread-based parallelism
    )
```

### schwimmbad Integration

```python
from schwimmbad import MultiPool, SerialPool

# Advanced pool types for HPC environments
with MultiPool(processes=4) as walker_pool:
    results = run_ensemble_resampler(
        n_walkers=16,
        n_steps=2000,
        n_states=4,
        n_dims=[2, 3, 4, 5],
        log_posterior_ens=posterior_ensembles,
        log_pseudo_prior_ens=pseudo_prior_ensembles,
        walker_pool=walker_pool
    )
```

### Pool Reuse Pattern for Multiple Runs

```python
# Efficient pool reuse for multiple ensemble resampler runs
with ProcessPoolExecutor(max_workers=4) as walker_pool:
    results_list = []
    
    for run_id in range(10):
        results = run_ensemble_resampler(
            n_walkers=16,
            n_steps=2000,
            n_states=4,
            n_dims=[2, 3, 4, 5],
            log_posterior_ens=posterior_ensembles,
            log_pseudo_prior_ens=pseudo_prior_ensembles,
            walker_pool=walker_pool,  # Reuse same pool
            seed=42 + run_id
        )
        results_list.append(results)

# Pool reuse provides 20-40% speedup over creating new pools each time
```

### Backward Compatibility

```python
# Legacy parallel=True approach still works
results = run_ensemble_resampler(
    n_walkers=16,
    n_steps=2000,
    n_states=4,
    n_dims=[2, 3, 4, 5],
    log_posterior_ens=posterior_ensembles,
    log_pseudo_prior_ens=pseudo_prior_ensembles,
    parallel=True,       # Creates internal ProcessPoolExecutor
    n_processors=4       # Number of worker processes
)
```

## Performance Guidelines

### CPU-bound vs I/O-bound Workloads

- **CPU-bound posterior evaluations**: Use `ProcessPoolExecutor` at both levels
- **I/O-bound workloads**: Consider `ThreadPoolExecutor` for walker level to reduce overhead
- **Mixed workloads**: Start with `ProcessPoolExecutor`, profile and adjust

### Memory Considerations

| Configuration | Memory Usage | Best For |
|---------------|--------------|----------|
| State-only parallelism | Moderate | Many states, lightweight posteriors |
| Walker-only parallelism | Higher | Few states, expensive posteriors |
| Two-level parallelism | Highest | Many states, expensive posteriors |
| Ensemble resampler walker parallelism | Moderate | High walker counts, long chains |
| MPI + local multiprocessing | Distributed | Very large problems |

### Scaling Recommendations

1. **Start simple**: Begin with single-level parallelism
2. **Profile first**: Measure actual bottlenecks before optimizing
3. **Consider overhead**: Small problems may not benefit from parallelism
4. **NUMA awareness**: Pin processes to cores on multi-socket systems

### Rule of Thumb for Process Counts

```python
import os
n_cores = os.cpu_count()

# Conservative approach (avoid oversubscription)
n_state_processes = min(n_states, n_cores // 2)
n_walker_processes = min(n_walkers, n_cores // n_state_processes)

# Aggressive approach (for CPU-bound workloads)
n_state_processes = min(n_states, n_cores)
n_walker_processes = 2  # Minimal walker parallelism
```

### Performance Insights from Testing

**Observed Performance Patterns** (32-core system, 4 states, 16 walkers, 50 steps):

| Configuration | Execution Time | Speedup | Best Use Case |
|---------------|----------------|---------|---------------|
| Sequential | 0.13s | 1.00x | Baseline, small problems |
| State parallel (2 proc) | 0.10s | 1.31x | Many states, simple posteriors |
| Walker parallel (Thread) | 0.13s | 1.02x | I/O-bound posteriors |
| Walker parallel (Process) | 0.15s | 0.88x | CPU-bound posteriors |
| Two-level (Process+Thread) | 0.27s | 0.50x | Large problems only |
| Two-level (Process+Process) | 0.92s | 0.15x | Very large problems only |

**Key Insights**:
1. **State-level parallelism** shows consistent speedup for multi-state problems
2. **ThreadPoolExecutor** often outperforms ProcessPoolExecutor for walker-level parallelism due to lower overhead
3. **Two-level parallelism** has significant overhead - only beneficial for large problems
4. **Internal parallelism** (n_state_processors, parallel=True) performs similarly to explicit pools
5. **Ensemble resampler** benefits most from parallelization with >16 walkers and >1000 steps
6. **Pool reuse** for ensemble resampler provides 20-40% speedup for multiple runs

## SLURM Integration Examples

### Single-Node Multi-Core

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00

python my_transc_script.py --state-workers=4 --walker-workers=8
```

### Multi-Node MPI

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

mpiexec -n 16 python my_transc_mpi_script.py
```

## Troubleshooting

### Common Errors and Solutions

1. **"daemonic processes are not allowed to have children"**
   ```python
   # Wrong:
   with multiprocessing.Pool() as pool:  # daemon processes
   
   # Right:
   with ProcessPoolExecutor() as pool:   # non-daemon processes
   ```

2. **"Pool objects cannot be passed between processes"**
   - Solution: Create pools within each process, don't pass them
   - Use our two-level architecture instead

3. **Memory exhaustion with many processes**
   ```python
   # Monitor memory usage and reduce process counts
   import psutil
   memory_gb = psutil.virtual_memory().total / (1024**3)
   max_processes = int(memory_gb / estimated_memory_per_process)
   ```

4. **MPI initialization errors**
   ```python
   # Ensure proper MPI initialization order
   from schwimmbad import MPIPool
   
   with MPIPool() as pool:
       if not pool.is_master():
           pool.wait()
           sys.exit(0)
       # ... main computation
   ```

### Performance Debugging

1. **Profile single-threaded first**:
   ```python
   import cProfile
   cProfile.run('run_mcmc_per_state(...)', 'profile_output.prof')
   ```

2. **Monitor resource usage**:
   ```bash
   htop  # CPU and memory usage
   nvidia-smi  # GPU usage (if applicable)
   ```

3. **Time different configurations**:
   ```python
   import time
   
   configs = [
       {'state_pool': None, 'emcee_pool': None},
       {'state_pool': state_pool, 'emcee_pool': None},
       {'state_pool': None, 'emcee_pool': walker_pool},
       {'state_pool': state_pool, 'emcee_pool': walker_pool}
   ]
   
   for config in configs:
       start = time.time()
       run_mcmc_per_state(..., **config)
       print(f"Config {config}: {time.time() - start:.2f}s")
   ```

## Resource Monitoring and System Guidelines

### System Resource Analysis

Before implementing parallelization, analyze your system resources:

```python
import os
import psutil
import multiprocessing

# CPU information
print(f"Physical cores: {psutil.cpu_count(logical=False)}")
print(f"Logical cores: {psutil.cpu_count(logical=True)}")

# Memory information
memory = psutil.virtual_memory()
print(f"Total memory: {memory.total / (1024**3):.1f} GB")
print(f"Available memory: {memory.available / (1024**3):.1f} GB")

# Process count recommendations
n_physical_cores = psutil.cpu_count(logical=False)
n_logical_cores = psutil.cpu_count(logical=True)

# Conservative approach (avoid oversubscription)
print(f"Conservative state processes: {min(n_states, n_physical_cores // 2)}")
print(f"Conservative walker processes: {min(n_walkers, n_physical_cores // 2)}")

# Aggressive approach (utilize all cores)
print(f"Aggressive state processes: {min(n_states, n_logical_cores)}")
print("Aggressive walker processes: 2-4 (minimal walker parallelism)")
```

### Memory Estimation and Safety

```python
def estimate_safe_process_count(base_memory_mb=50, problem_memory_mb=20):
    """Estimate maximum safe process count based on available memory."""
    memory = psutil.virtual_memory()
    total_memory_per_process = (base_memory_mb + problem_memory_mb) * 1024**2
    max_safe = int((memory.available * 0.8) / total_memory_per_process)
    return max_safe

# Monitor memory during execution
def monitor_memory_usage():
    """Monitor memory usage during parallel execution."""
    memory = psutil.virtual_memory()
    return {
        'percent_used': memory.percent,
        'available_gb': memory.available / (1024**3)
    }
```

## Advanced Pool Management

### Pool Reuse Patterns

Pool creation has significant overhead. Reusing pools across multiple operations can provide 20-40% speedup:

```python
# Efficient pool reuse pattern
with ProcessPoolExecutor(max_workers=4) as walker_pool:
    results_list = []
    
    for run_id in range(10):
        # Reuse the same pool for multiple runs
        results = run_ensemble_resampler(
            walker_pool=walker_pool,  # Reuse existing pool
            seed=42 + run_id,         # Different seeds for each run
            **other_params
        )
        results_list.append(results)

# vs inefficient pool creation each time:
results_list = []
for run_id in range(10):
    with ProcessPoolExecutor(max_workers=4) as walker_pool:  # Creates new pool each time
        results = run_ensemble_resampler(walker_pool=walker_pool, **params)
        results_list.append(results)
```

### Context Manager Best Practices

```python
# Always use context managers for automatic cleanup
with ProcessPoolExecutor(max_workers=4) as pool:
    results = run_ensemble_resampler(walker_pool=pool, **params)
# Pool automatically cleaned up here

# Manual control when needed
pool = ProcessPoolExecutor(max_workers=4)
try:
    results = run_ensemble_resampler(walker_pool=pool, **params)
finally:
    pool.shutdown(wait=True)  # Ensure proper cleanup
```

## Common Pitfalls and Troubleshooting

### Memory Exhaustion

**Problem**: Too many processes consume all available memory
**Solution**: Monitor and adjust process counts

```python
import psutil

def check_memory_before_parallel():
    memory = psutil.virtual_memory()
    if memory.percent > 80:
        print("⚠️ Warning: High memory usage detected!")
        print(f"   Current usage: {memory.percent:.1f}%")
        print("   Consider reducing process count")
        return False
    return True

# Monitor during execution
initial_memory = psutil.virtual_memory().percent
# ... run parallel code ...
final_memory = psutil.virtual_memory().percent
print(f"Memory usage increased by {final_memory - initial_memory:.1f}%")
```

### Overhead vs Benefit Trade-offs

**Problem**: Parallelism slower than sequential for small problems
**Solution**: Profile first, parallelize only when beneficial

```python
import time

def should_parallelize(problem_size_estimate):
    """Test if parallelization is beneficial for your problem size."""
    # Quick test with small problem
    small_test_time = time_sequential_execution(small_problem=True)
    parallel_test_time = time_parallel_execution(small_problem=True)
    
    overhead_ratio = parallel_test_time / small_test_time
    
    if overhead_ratio > 1.2:
        print("⚠️ Parallel overhead detected!")
        print(f"   Sequential: {small_test_time:.3f}s")
        print(f"   Parallel:   {parallel_test_time:.3f}s")
        print(f"   Overhead:   {overhead_ratio:.2f}x")
        return False
    
    return True
```

### Process Count Guidelines

**Problem**: Oversubscription degrades performance
**Solution**: Follow systematic guidelines

```python
import psutil

def recommend_process_counts(n_states, n_walkers):
    n_physical = psutil.cpu_count(logical=False)
    n_logical = psutil.cpu_count(logical=True)
    
    recommendations = {
        'conservative_state': min(n_states, n_physical // 2),
        'conservative_walker': min(n_walkers, n_physical // 2),
        'aggressive_state': min(n_states, n_logical),
        'aggressive_walker': 2  # Minimal walker parallelism
    }
    
    print("Process count recommendations:")
    for config, count in recommendations.items():
        print(f"  {config}: {count}")
    
    return recommendations
```

### Debugging Common Errors

```python
# Common error patterns and solutions

def debug_parallelization_issues():
    print("Common parallelization errors and solutions:")
    print()
    print("❌ 'daemonic processes are not allowed to have children'")
    print("   → Use ProcessPoolExecutor instead of multiprocessing.Pool")
    print()
    print("❌ 'Pool objects cannot be passed between processes'")
    print("   → Use pyTransC's two-level architecture")
    print()
    print("❌ Memory usage grows unexpectedly")
    print("   → Check for memory leaks in posterior functions")
    print()
    print("❌ Parallel runs give different results")
    print("   → Set random seeds consistently across processes")
    print()
    print("❌ Performance degrades with more processes")
    print("   → Check for I/O bottlenecks or memory pressure")

def troubleshooting_checklist():
    print("Troubleshooting checklist:")
    print("✅ Sequential version works correctly")
    print("✅ Posterior function is CPU-bound")
    print("✅ Problem size justifies parallelization overhead")
    print("✅ Sufficient memory for multiple processes")
    print("✅ Random seeds handled consistently")
    print("✅ No file I/O or shared state conflicts")
```

### When NOT to Use Parallelism

```python
def should_avoid_parallelism(problem_characteristics):
    """Determine if parallelism should be avoided."""
    avoid_reasons = []
    
    if problem_characteristics['execution_time'] < 1.0:  # seconds
        avoid_reasons.append("Problem completes quickly (<1s)")
    
    if problem_characteristics['memory_gb'] > 0.8 * psutil.virtual_memory().total / (1024**3):
        avoid_reasons.append("Problem uses most available memory")
    
    if problem_characteristics['io_bound']:
        avoid_reasons.append("Posterior function is I/O bound")
    
    if psutil.cpu_count() == 1:
        avoid_reasons.append("Single-core system")
    
    if avoid_reasons:
        print("Consider avoiding parallelism:")
        for reason in avoid_reasons:
            print(f"  ❌ {reason}")
        return True
    
    return False
```

## Best Practices Summary

### General Guidelines
1. **Use ProcessPoolExecutor** as default for both levels
2. **Start with single-level parallelism** and profile before adding complexity  
3. **Monitor system resources** before and during parallel execution
4. **Avoid daemon processes** (use our provided architecture)
5. **Consider MPI for clusters** with `schwimmbad.MPIPool`
6. **Profile before optimizing** to identify actual bottlenecks
7. **Test pool compatibility** in small examples first

### `run_mcmc_per_state` Specific
8. **Use appropriate chunk sizes** for load balancing
9. **Consider overhead vs benefit** - parallelism isn't always faster for small problems
10. **Prefer ThreadPoolExecutor** for walker-level parallelism unless CPU-bound
11. **Monitor memory usage** with large problems and many states

### `run_ensemble_resampler` Specific
12. **Use walker parallelism** for >16 walkers and >1000 steps per walker
13. **Reuse pools** across multiple runs for 20-40% speedup
14. **ProcessPoolExecutor recommended** for CPU-intensive ensemble resampling
15. **ThreadPoolExecutor acceptable** for I/O-bound or memory-limited scenarios
16. **Monitor state visitation frequencies** to verify parallel results consistency

### Resource Management
17. **Estimate memory requirements** before scaling up
18. **Use pool reuse patterns** for efficiency gains
19. **Implement proper error handling** in parallel contexts
20. **Monitor system performance** during execution

### Integration Patterns
21. **Combine approaches**: Use state-level parallelism for ensemble generation, walker-level for ensemble resampling
22. **Consistent pool types** across workflow stages for simplicity
23. **Profile your specific posterior** characteristics before choosing pool types
24. **Consider workload balance** across different states