#!/usr/bin/env python3
#
import os
import sys
import glob
from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
import logging

from tearing_eigenmodes import build_params, build_dpath, setup_logging, SimulationParams

def process_file(sname: str, params: SimulationParams) -> None:
    """
    Process and plot a single .npz file.
    """
    if not os.path.exists(sname):
        logging.error(f"Error: State file not found: {sname}")
        return

    logging.info(f"Loading state from {sname}...")

    # Load the state
    with np.load(sname) as state:
        z = state['grid']

        # Determine variables to plot based on what is present in the state file
        possible_variables = ['duz', 'dbz', 'duy', 'dby', 'ddp', 'dpi']
        variables = [v for v in possible_variables if v in state]

        if not variables:
            logging.error(f"Error: No eigenfunctions found in state file: {sname}")
            return


        # Plotting
        n_vars = len(variables)
        fig, axes = plt.subplots(n_vars, 1, sharex=True, figsize=(10, 3 * n_vars), constrained_layout=True)
        if n_vars == 1:
            axes = [axes]

        zmin = params.zmin
        zmax = params.zmax

        # Filter indices if zmin/zmax are provided
        mask = np.ones_like(z, dtype=bool)
        if zmin is not None or zmax is not None:
            if zmin is not None:
                mask &= (z >= zmin)
            if zmax is not None:
                mask &= (z <= zmax)

            z_plot = z[mask]
        else:
            z_plot = z

        # Mapping for LaTeX labels
        label_map = {
            'duz': r'$\delta u_z$',
            'dbz': r'$\delta B_z$',
            'duy': r'$\delta u_y$',
            'dby': r'$\delta B_y$',
            'ddp': r'$\delta \Delta p$',
            'dpi': r'$\delta \pi$'
        }

        for ax, var in zip(axes, variables):
            data = state[var][mask]

            ax.plot(z_plot, data.real, label='Real', color='blue', linewidth=1.5)
            ax.plot(z_plot, data.imag, label='Imag', color='orange', linewidth=1.5)
            ax.plot(z_plot, np.abs(data), label='Abs', color='black', linewidth=1.5)

            label = label_map.get(var, f'${var}$')
            ax.set_ylabel(label, fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(loc='upper right')

        axes[-1].set_xlabel('$z$', fontsize=12)

        # Determine title
        title = "Eigenmode Solutions"
        dep = state.get('scan_parameter', state.get('dependence', None))
        if dep is not None:
            dep = str(dep)
            val = float(state.get('scan_parameter_value', state.get('value', 0.0)))
            title += f" (${dep} = {val:.4e}$)"
            if 'wavenumber' in state:
                title += f", $\\alpha = {float(state['wavenumber']):.4e}$"
        elif 'wavenumber' in state:
            title += f" ($\\alpha = {float(state['wavenumber']):.4e}$)"

        fig.suptitle(title, fontsize=14)

        # Save or show the plot
        out_name = params.output_plot
        if not out_name or len(glob.glob((params.dir_plot if params.dir_plot is not None else '') + '/*.npz')) > 1:
            out_name = os.path.splitext(sname)[0] + '.png'

        plt.savefig(out_name, dpi=300)
        plt.close(fig)  # Close to free memory
        logging.info(f"Plot saved to {out_name}")

def main() -> None:
    """
    Reads an eigenmode state from a .npz file and plots the solutions.
    """
    # Parse command-line arguments using the 'plot' type
    params = build_params(parser_type='plot')

    # Configure logging
    setup_logging(verbose=bool(params.verbose), log_file=params.log_file)

    # Build data path
    dpath = build_dpath(params)
    params.data_path = dpath

    # Check if directory plotting is requested
    dir_name = params.dir_plot
    if dir_name:
        if not os.path.exists(dir_name):
            logging.error(f"Error: Directory not found: {dir_name}")
            sys.exit(1)

        files = sorted(glob.glob(os.path.join(dir_name, "*.npz")))
        if not files:
            logging.error(f"Error: No .npz files found in {dir_name}")
            sys.exit(1)

        logging.info(f"Processing {len(files)} files in {dir_name}...")
        for f in files:
            process_file(f, params)
        return

    # Identify a single state file
    sname = params.file_plot
    if not sname:
        if params.alpha_plot is not None:
            α = params.alpha_plot
            sname = os.path.join(dpath, f'state_α{α:.6e}.npz')
        elif params.value_plot is not None and params.dependence is not None:
            dep = params.dependence
            val = params.value_plot
            sname = os.path.join(dpath, f'state_{dep}{val:+.6e}.npz')
        else:
            logging.error("Error: Must provide either --file, --dir, --alpha, or (--value and --dependence) to identify the state file(s).")
            sys.exit(1)

    process_file(sname, params)

if __name__ == "__main__":
    main()
