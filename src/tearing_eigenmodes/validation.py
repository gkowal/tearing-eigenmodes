import os
import logging
import numpy as np
from tearing_eigenmodes.io import save_eigenmode

logger = logging.getLogger(__name__)


def validate_and_fix_file(filepath: str, dry_run: bool = False, verbose: bool = False) -> tuple[bool, bool]:
    """
    Validate an eigenmode .npz file, rename older keys to modern ones, and backfill
    any missing required fields. If modified and dry_run is False, updates the file atomically.

    Returns:
        tuple[bool, bool]: (isValid, isModified). True if the file is valid (or successfully updated to be valid), False otherwise.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return False, False

    try:
        with np.load(filepath, allow_pickle=True) as state:
            # np.load holds a file handle open, so we load into a mutable dictionary
            data = {k: state[k] for k in state.files}
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return False, False

    modified = False

    # 1. Parse filename to guess run type and values if missing
    import re
    filename = os.path.basename(filepath)
    m = re.match(r"^state(?:s)?_([a-zA-ZβΔξϵα]+)([+-]?[0-9].*)\.npz$", filename)
    parsed_param = None
    parsed_val = None
    if m:
        parsed_param = m.group(1)
        try:
            parsed_val = float(m.group(2))
        except ValueError:
            pass

    # Check if it's a dependence sweep file or dispersion run
    is_dependence = 'dependence' in data or 'scan_parameter' in data
    if not is_dependence and parsed_param and parsed_param != 'α':
        is_dependence = True

    # 2. Key Mapping & Renaming

    # 2a. Growth rate / eigenvalue rename
    if 'growth_rate' in data and 'eigenvalue' not in data:
        data['eigenvalue'] = data.pop('growth_rate')
        modified = True
    if 'growth_rate_error' in data and 'eigenvalue_error' not in data:
        # Standardize error naming first if growth_rate_error exists
        data['eigenvalue_error'] = data.pop('growth_rate_error')
        modified = True

    if 'eigenvalues' in data and 'eigenvalue' not in data:
        # Handle old array-based format
        eigenvalues = data.pop('eigenvalues')
        data['eigenvalue'] = np.array(eigenvalues[0]) if eigenvalues.size > 0 else np.array(0.0j)
        modified = True

    # 2b. Inner scale rename (legacy files only: current files store
    # 'inner_scale' as the run-config grid input next to the measured
    # 'resistive_layer_thickness', which must not be overwritten)
    if 'inner_scale' in data and 'resistive_layer_thickness' not in data:
        data['resistive_layer_thickness'] = data.pop('inner_scale')
        modified = True

    # 2c. Scaling factor rename
    if 'scaling_factor' in data:
        data['grid_scaling_factor'] = data.pop('scaling_factor')
        modified = True

    # 2d. Node counts rename
    if 'n_inner' in data and 'resistive_layer_nodes' not in data:
        data['resistive_layer_nodes'] = data.pop('n_inner')
        modified = True
    if 'n_wa' in data:
        data['current_sheet_nodes'] = data.pop('n_wa')
        modified = True

    # 2e. Scan parameter renames for dependence sweeps
    if is_dependence:
        if 'dependence' in data:
            data['scan_parameter'] = data.pop('dependence')
            modified = True
        if 'value' in data:
            data['scan_parameter_value'] = data.pop('value')
            modified = True
    else:
        # For dispersion runs, remove the redundant 'value' field if present
        if 'value' in data:
            data.pop('value')
            modified = True

    # 2f. Backfill wavenumber from filename if missing for dispersion runs
    if not is_dependence and 'wavenumber' not in data and parsed_param == 'α' and parsed_val is not None:
        data['wavenumber'] = np.array(parsed_val)
        modified = True

    # 3. Validate presence of core keys
    core_keys = [
        'wavenumber',
        'eigenvalue',
        'tolerance',
        'resistive_layer_thickness',
        'grid_scaling_factor',
        'resolution',
        'grid'
    ]
    missing_cores = [k for k in core_keys if k not in data]
    if missing_cores:
        logger.warning(f"File {filepath} is invalid: missing core fields {missing_cores}")
        return False, False

    # We require the core eigenfunctions 'duz' and 'dbz' to be present
    required_core = ['duz', 'dbz']
    missing_core = [k for k in required_core if k not in data]
    if missing_core:
        logger.warning(f"File {filepath} is invalid: missing core eigenfunctions {missing_core}")
        return False, False


    # 4. Fill in metadata defaults if missing
    metadata_defaults = {
        'S': 1e4, 'Pr': 0.0, 'plasma_beta': 0.0, 'plasma_beta_difference': 0.0,
        'xi': 0.0, 'Hall': 0.0, 'zeta': 1.0, 'a': 1.0, 'w': 0.0, 'parallel_index': 3.0,
        'perpendicular_index': 2.0, 'eos': 'adiabatic', 'CGL': False, 'noshear': False,
        'Nmin': 64, 'Nmax': 2048, 'Ninc': 32, 'atol': 1e-10, 'rtol': 1e-5,
        'gtol': 1e-2, 'dtol': 1e-3, 'n_inner_req': 0, 'f_outer': 1e-4, 'mode': 0
    }
    for k, default in metadata_defaults.items():
        if k not in data:
            data[k] = np.array(default)
            modified = True
            if verbose:
                logger.info(f"[{filepath}] Filled missing metadata field {k} with default {default}")

    # 5. Recalculate and backfill missing secondary fields
    grid = data['grid']

    # 5a. resistive_layer_nodes
    if 'resistive_layer_nodes' not in data:
        thick = float(data['resistive_layer_thickness'])
        nodes = max(1, np.where(np.abs(grid) <= thick)[0].size)
        data['resistive_layer_nodes'] = np.array(nodes)
        modified = True
        if verbose:
            logger.info(f"[{filepath}] Recalculated 'resistive_layer_nodes': {nodes}")

    # 5b. current_sheet_nodes
    if 'current_sheet_nodes' not in data:
        a = float(data.get('a', 1.0))
        w = float(data.get('w', 0.0))
        nodes = np.where(np.abs(grid) <= (w + a))[0].size
        data['current_sheet_nodes'] = np.array(nodes)
        modified = True
        if verbose:
            logger.info(f"[{filepath}] Recalculated 'current_sheet_nodes': {nodes}")

    # 5c. Dependence run specific fields
    if is_dependence:
        # scan_parameter (ensure not empty)
        if 'scan_parameter' not in data:
            param_fallback = 'S'
            if parsed_param and parsed_param != 'α':
                param_fallback = parsed_param
            data['scan_parameter'] = np.array(param_fallback)
            modified = True
            if verbose:
                logger.info(f"[{filepath}] Set missing 'scan_parameter' to guessed value: {param_fallback}")

        # scan_parameter_value
        if 'scan_parameter_value' not in data:
            p_name = str(data['scan_parameter'])
            val_fallback = parsed_val if parsed_val is not None else float(data.get(p_name, 1e4))
            data['scan_parameter_value'] = np.array(val_fallback)
            modified = True
            if verbose:
                logger.info(f"[{filepath}] Set missing 'scan_parameter_value' to: {val_fallback}")

        # niter
        if 'niter' not in data:
            data['niter'] = np.array(1)
            modified = True
            if verbose:
                logger.info(f"[{filepath}] Filled missing 'niter' with default 1")

        # wavenumber_error
        if 'wavenumber_error' not in data:
            wtol = 1e-3
            val = float(data['wavenumber'])
            err = wtol * val
            data['wavenumber_error'] = np.array(err)
            modified = True
            if verbose:
                logger.info(f"[{filepath}] Recalculated 'wavenumber_error': {err:.4e}")

        # eigenvalue_error
        if 'eigenvalue_error' not in data:
            rtol = float(data.get('rtol', 1e-5))
            val = float(np.real(data['eigenvalue']))
            tol = float(data['tolerance'])
            err = rtol * val * tol
            data['eigenvalue_error'] = np.array(err)
            modified = True
            if verbose:
                logger.info(f"[{filepath}] Recalculated 'eigenvalue_error': {err:.4e}")

    # 6. Save back to file if modified and not dry-run
    if modified:
        if dry_run:
            logger.info(f"[{filepath}] [DRY-RUN] File would be updated.")
        else:
            try:
                save_eigenmode(filepath, **data)
                if verbose:
                    logger.info(f"[{filepath}] Successfully validated and updated file.")
            except Exception as e:
                logger.error(f"[{filepath}] Failed to save updated file: {e}")
                return False, False
    else:
        if verbose:
            logger.info(f"[{filepath}] File is fully valid.")

    return True, modified
