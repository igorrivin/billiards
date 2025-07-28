# Periodic Billiard Orbits in Lp Balls

A computational toolkit for finding and verifying periodic billiard orbits in Lp balls using Newton's method and Smale's α-criterion.

## Overview

This repository provides tools to:
- Find periodic billiard orbits in Lp superellipses using variational methods
- Verify orbit existence using Smale's α-criterion
- Visualize orbits with signature and rotation number information
- Analyze patterns in orbit types across different parameters

## Quick Start

```bash
# Find periodic orbits with 5 bounces in L³ ball
python orbit_finder.py 3.0 5 1000

# Verify a specific orbit
python verify_orbit.py 3.0 "[0.0657, 0.375, 0.6843]"

# Plot orbits from results
python plot_orbit.py --csv p3.0_N5_orbits.csv --row 1
```

## Mathematical Background

Periodic billiard orbits in the Lp ball (|x|^p + |y|^p ≤ 1) correspond to critical points of the perimeter functional on the space of inscribed N-gons. We find these using:

1. **Newton's Method**: Applied to the gradient of the perimeter functional
2. **Smale's α-Criterion**: Provides rigorous verification that a numerical critical point corresponds to a genuine orbit
3. **Canonical Forms**: Handle symmetries by computing canonical representatives

## Main Scripts

### `orbit_finder.py` - Orbit Finder

The main computational engine using JAX for efficient batched computations.

```bash
python orbit_finder.py P N N_SEEDS [--N_STEPS STEPS]
```

- `P`: Shape parameter (p > 2, p ≠ 2)
- `N`: Number of bounce points
- `N_SEEDS`: Number of random initial conditions
- `--N_STEPS`: Newton iterations (default: 30)

**Output files:**
- `p{P}_N{N}_orbits.csv`: Found orbits with signatures and rotation numbers
- `p{P}_N{N}_metadata.json`: Statistics and summary

### `verify_orbit.py` - Orbit Verifier

Standalone verification tool (pure NumPy, no JAX required).

```bash
# JSON array format
python verify_orbit.py 3.0 "[0.1, 0.3, 0.5, 0.7, 0.9]"

# Comma-separated format
python verify_orbit.py 3.0 0.1,0.3,0.5,0.7,0.9

# Verbose output
python verify_orbit.py 3.0 "[0.125, 0.625]" --verbose
```

### `plot_orbit.py` - Orbit Visualizer

Flexible plotting tool with multiple input options.

```bash
# From CSV file (most common workflow)
python plot_orbit.py --csv p3.0_N5_orbits.csv --row 5

# Multiple orbits in grid
python plot_orbit.py --csv p3.0_N5_orbits.csv --rows 1,10,20,30 --grid

# Direct input
python plot_orbit.py --p 3.0 --theta "0.0657,0.375,0.6843"
```

## Key Concepts

### Morse Signature (pos, neg, zero)
- Number of positive, negative, and zero eigenvalues of the Hessian
- (0,N,0): Local maximum of perimeter
- (1,N-1,0): Saddle point with one unstable direction
- (2,N-2,0): Saddle point with two unstable directions

### Rotation Number (r/s)
- Topological invariant describing how the orbit winds around
- 1/N: Convex N-gon
- 2/N: Star polygon (skipping one vertex)
- k/N: General star pattern (when gcd(k,N) = 1, k>1)
- 0/1: Back-and-forth oscillatory orbit (only for even N)

### Smale's α-Criterion
- α = β × γ where:
  - β: Newton method residual
  - γ: Condition number factor
- α < 0.732: Theoretical guarantee
- α < 0.15767: Conservative threshold used in practice

## Examples

### Finding Triangular Orbits (N=3)
```bash
python orbit_finder.py 3.0 3 1000
# Typical result: All orbits have signature (0,3,0) and rotation 1/3
```

### Finding Star Orbits (N=5)
```bash
python orbit_finder.py 3.0 5 1000
# Results show mix of signatures: (0,5,0), (1,4,0), rare (2,3,0)
# All have rotation number 1/5
```

### Oscillatory Orbits (N=2)
```bash
python orbit_finder.py 3.0 2 1000
# Shows both rotation 0/1 (oscillatory) and 1/2 patterns
```

## Discovered Patterns

Our experiments reveal interesting patterns:

1. **N=3 (prime)**: Only rotation 1/3 (triangles)
2. **N=5 (prime)**: Only rotation 2/5 (star pentagons)
3. **N=7 (prime)**: Mixed rotations (1/7, 2/7, 3/7)
4. **Even N**: Allows oscillatory orbits (rotation 0/1)
5. **Saddle orbits often have longer perimeters than local maxima**

## Requirements

```bash
# For orbit_finder.py
pip install jax jaxlib numpy pandas

# For plotting
pip install matplotlib

# For verification only
pip install numpy
```

## Paper

See `paper/billiard_orbits.tex` for a detailed mathematical exposition of the method and results.

## Contributing

This is research code. Feel free to:
- Report bugs or numerical issues
- Suggest improvements to the algorithms
- Share interesting patterns you discover
- Extend to 3D or other domains

## Citation

If you use this code in research, please cite:
```
@software{billiard_orbits_2024,
  title = {Periodic Billiard Orbits in Lp Balls},
  author = {Generated with Claude Code},
  year = {2024},
  url = {https://github.com/yourusername/billiard-orbits}
}
```

## License

MIT License - see LICENSE file for details.