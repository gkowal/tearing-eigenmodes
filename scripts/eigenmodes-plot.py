#!/usr/bin/env python3
#
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging

from tearing_eigenmodes import build_params, build_dpath, setup_logging

def main():
    """
    Reads an eigenmode state from a .npz file and plots the solutions.
    """
    # Parse command-line arguments using the 'plot' type
    params = build_params(parser_type='plot')

    # Configure logging
    setup_logging(verbose=params.get('verbose'), log_file=params.get('log_file'))

    # Build data path
    dpath = build_dpath(params)
    params['data_path'] = dpath

    # Identify the state file
    sname = params.get('file_plot')
    if not sname:
        if params.get('alpha_plot') is not None:
            α = params['alpha_plot']
            sname = os.path.join(dpath, f'state_α{α:.6e}.npz')
        elif params.get('value_plot') is not None and params.get('dependence') is not None:
            dep = params['dependence']
            val = params['value_plot']
            sname = os.path.join(dpath, f'state_{dep}{val:+.6e}.npz')
        else:
            logging.error("Error: Must provide either --file, --alpha, or (--value and --dependence) to identify the state file.")
            sys.exit(1)

    if not os.path.exists(sname):
        logging.error(f"Error: State file not found: {sname}")
        sys.exit(1)

    logging.info(f"Loading state from {sname}...")

    # Load the state
    with np.load(sname) as state:
        z = state['grid']
        
        variables = ['duz', 'dbz']
        if params.get('CGL'):
            variables += ['duy', 'dby', 'ddp']

        # Check if all variables exist in the state
        missing = [v for v in variables if v not in state]
        if missing:
            logging.error(f"Error: Missing variables in state file: {', '.join(missing)}")
            sys.exit(1)

        # Plotting
        n_vars = len(variables)
        fig, axes = plt.subplots(n_vars, 1, sharex=True, figsize=(10, 3 * n_vars), constrained_layout=True)
        if n_vars == 1:
            axes = [axes]

        zmin = params.get('zmin')
        zmax = params.get('zmax')

        # Filter indices if zmin/zmax are provided
        if zmin is not None or zmax is not None:
            mask = np.ones_like(z, dtype=bool)
            if zmin is not None:
                mask &= (z >= zmin)
            if zmax is not None:
                mask &= (z <= zmax)
            
            z_plot = z[mask]
        else:
            z_plot = z
            mask = slice(None)

        for ax, var in zip(axes, variables):
            data = state[var][mask]
            
            ax.plot(z_plot, data.real, label='Real', linewidth=1.5)
            ax.plot(z_plot, data.imag, '--', label='Imag', linewidth=1.5)
            
            ax.set_ylabel(f'${var}$', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(loc='upper right')

        axes[-1].set_xlabel('$z$', fontsize=12)
        
        title = f"Eigenmode Solutions"
        if params.get('alpha_plot') is not None:
            title += f" ($\\alpha = {params['alpha_plot']:.4e}$)"
        elif params.get('value_plot') is not None:
            title += f" ({params['dependence']} = {params['value_plot']:.4e})"
        
        fig.suptitle(title, fontsize=14)

        # Save or show the plot
        out_name = params.get('output_plot')
        if not out_name:
            out_name = os.path.splitext(sname)[0] + '.png'
        
        plt.savefig(out_name, dpi=300)
        logging.info(f"Plot saved to {out_name}")

if __name__ == "__main__":
    main()
