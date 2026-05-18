#!/usr/bin/env python3
#
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import logging

from tearing_eigenmodes import build_params, build_dpath, setup_logging

def process_file(sname, params):
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
        
        variables = ['duz', 'dbz']
        if params.get('CGL'):
            variables += ['duy', 'dby', 'ddp']

        # Check if all variables exist in the state
        missing = [v for v in variables if v not in state]
        if missing:
            logging.error(f"Error: Missing variables in state file: {', '.join(missing)}")
            return

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
            
            ax.plot(z_plot, data.real, label='Real', color='blue', linewidth=1.5)
            ax.plot(z_plot, data.imag, label='Imag', color='orange', linewidth=1.5)
            ax.plot(z_plot, np.abs(data), label='Abs', color='black', linewidth=1.5)
            
            ax.set_ylabel(f'${var}$', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(loc='upper right')

        axes[-1].set_xlabel('$z$', fontsize=12)
        
        # Determine title
        title = "Eigenmode Solutions"
        if 'value' in state:
            val = float(state['value'])
            dep = params.get('dependence')
            
            # If dependence not provided, try to find it in metadata or filename
            if not dep:
                if 'dependence' in state:
                    dep = str(state['dependence'])
                else:
                    # Try to parse from filename: state_{dependence}{value:+.6e}.npz
                    import re
                    fname = os.path.basename(sname)
                    # This regex matches 'state_', then captures everything up to the first '+' or '-' 
                    # followed by a digit (the start of the value).
                    match = re.match(r'state_(.+?)[+-][0-9]', fname)
                    if match:
                        dep = match.group(1)
                    else:
                        # Fallback heuristic (prone to ambiguity if value is 0.0)
                        dep_map = {
                            'S': 'S', 'Pr': 'Pr', 'β': 'plasma_beta', 'Δβ': 'plasma_beta_difference',
                            'ξ': 'xi', 'ϵ': 'Hall', 'w': 'w', 'a': 'a'
                        }
                        for symbol, key in dep_map.items():
                            if key in state and np.isclose(float(state[key]), val, rtol=1e-8):
                                dep = symbol
                                break
            
            if dep:
                title += f" (${dep} = {val:.4e}$)"
            else:
                title += f" (value = {val:.4e})"
            
            if 'wavenumber' in state:
                title += f", $\\alpha = {float(state['wavenumber']):.4e}$"
        elif 'wavenumber' in state:
            title += f" ($\\alpha = {float(state['wavenumber']):.4e}$)"
        
        fig.suptitle(title, fontsize=14)

        # Save or show the plot
        out_name = params.get('output_plot')
        if not out_name or len(glob.glob(params.get('dir_plot', '') + '/*.npz')) > 1:
            out_name = os.path.splitext(sname)[0] + '.png'
        
        plt.savefig(out_name, dpi=300)
        plt.close(fig)  # Close to free memory
        logging.info(f"Plot saved to {out_name}")

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

    # Check if directory plotting is requested
    dir_name = params.get('dir_plot')
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
            logging.error("Error: Must provide either --file, --dir, --alpha, or (--value and --dependence) to identify the state file(s).")
            sys.exit(1)

    process_file(sname, params)

if __name__ == "__main__":
    main()
