# Tearing Eigenmodes

Utilities for analyzing tearing instability eigenmodes using the `psecas` pseudospectral solver.

## Overview

This package implements a linear stability analysis of the tearing instability within the framework of linearized incompressible magnetohydrodynamics (MHD). It supports two physical models:
1. **Classical MHD** (`TearingClassicalMHD`): Incorporates resistive and viscous effects.
2. **Gyrotropic MHD** (`TearingGyrotropicMHD`): Incorporates gyrotropic pressuring effects (CGL double-adiabatic equations), including parallel/perpendicular plasma-β anisotropy, transverse magnetic guide fields, and Hall effects.

The linearized equations are solved using the Chebyshev rational pseudospectral solver provided by the `psecas` library.

## Installation

To install the package in editable mode:
```bash
pip install -e .
```

Ensure that the `psecas` library is installed and available in your Python environment.

## Usage

The package includes CLI scripts for performing calculations and visualizing results:

* **`scripts/eigenmodes-compute.py`**: Calculates the dispersion relation (growth rate $\sigma$ vs. wavenumber $k$) for a given set of physical parameters.
* **`scripts/eigenmodes-maxima.py`**: Performs sweeps over a specified parameter (e.g. $S$, $Pr$, $\beta$, $\Delta\beta$) to find the maximum growth rate and its corresponding wavenumber.
* **`scripts/eigenmodes-profiles.py`**: Visualizes the 1D profiles of the eigenfunctions (such as $\delta u_z$, $\delta B_z$) from saved state files.
* **`scripts/eigenmodes-maps.py`**: Reconstructs and plots 2D color maps of the eigenfunctions in the $X-Z$ plane.
* **`scripts/eigenmodes-validate.py`**: Validates and backfills the metadata of stored `.npz` eigenmode state files.

All scripts accept `--help` or `-h` to show the list of available command-line arguments.

### Grid Resolution and Multi-Scale Analysis Configuration

Grid resolution ($N$) and rational Chebyshev scaling factor ($C$) are automatically determined via physics-based multi-scale analysis balancing inner-layer resolution against outer-domain asymptotic decay:

* **`--inner-scale`** (`-δin`): Explicit target inner scale to resolve. If specified, overrides automatic estimation directly without applying safety factor. If omitted, it is automatically computed using per-term eigenmode scale crossings or analytic FKR/Coppi asymptotic inner-scale formulas across Classical and Gyrotropic regimes.
* **`--resistive-scale`** (`-δres`): *(Deprecated)* Legacy alias for `--inner-scale`.
* **`--thickness-tolerance`** (`-δtol`): *(Deprecated)* Legacy tolerance parameter; local-bracket interpolation on the Chebyshev grid is directly grid-resolved.
* **`--inner-resolution-safety`**: Numerical safety factor ($\ge 1.0$, default: 1.01) scaling the target grid inner scale finer than the physical prediction ($L_{\mathrm{inner}} = \ell_{\min} / s$). Safety 1.0 is valid but provides no margin against solve-to-solve physical-scale drift.
* **`--n-inner-scale`** (`-nin`): Number of collocation points allocated to resolve the inner scale (default: 5).
* **`--n-equilibrium`** (`-neq`): Number of collocation points allocated to resolve the equilibrium current sheet $a + w$ (default: 5).
* **`--n-anisotropy`** (`-naniso`): Number of collocation points allocated to resolve the Gyrotropic pressure-anisotropy scale $\delta_q$ (default: 5).

### Multi-Scale Diagnostics
The solver evaluates equation-level balance diagnostics across resistive, viscous, shear, and guide-field terms post-convergence:
* **Classical MHD**: Computes 10 standardized dominance scales across induction and vorticity equations using equation-local activity floors ($\epsilon_q$). Net-ideal balance diagnostics (`eta_vs_ideal` and `nu_vs_ideal`) compare non-ideal diffusion with the magnitude of the net signed complex ideal contribution ($|T_F + T_G + T_\xi|$), preserving relative phase, reinforcement, and cancellation among active terms. Separate envelope diagnostics (`eta_vs_ideal_envelope` and `nu_vs_ideal_envelope`) compare diffusion against the pointwise maximum of individual ideal term magnitudes ($\max(|T_F|, |T_G|, |T_\xi|)$).
* **CGL Gyrotropic MHD**: Computes 4 standardized induction dominance scales ($\epsilon = 0$).
* **Persistence**: Stores complete versioned physical-scale dictionaries in `.npz` files (`mode_scale_schema_version = 2`) alongside the canonical minimum physical scale ($\delta_{\mathrm{in}}$). Schema 1 records remap stored ideal comparisons to envelope diagnostics and treat net-ideal values as unavailable (`NaN`).
* **Refinement**: Interpolates active candidate physical scales independently within contiguous valid segments. Unconverged readable records act as barriers and split interpolation segments. Fallback precedence (`explicit --inner-scale -> per-term minimum / safety -> scalar minimum / safety -> legacy resistive / safety -> analytic estimator`) is evaluated independently at every requested coordinate. Mixed legacy and new caches remain fully supported.

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.
