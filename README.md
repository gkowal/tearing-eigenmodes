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

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.
