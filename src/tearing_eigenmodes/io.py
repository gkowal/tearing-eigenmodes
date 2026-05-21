import os
import glob
import logging
import numpy as np
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def build_dpath(params: Dict[str, Any]) -> str:
    """
    Construct the directory name that will hold the results for a given run.
    """
    a, w = params['a'], params['w']
    S, Pr = params['S'], params['Pr']
    ξ, ϵ = params['xi'], params['Hall']
    β, Δβ = params['plasma_beta'], params['plasma_beta_difference']

    dpath  = './RESULTS/'
    dpath += '' if S  is None else f'S{S:.3e}'
    dpath += '' if Pr is None else f'Pr{Pr:.3e}'
    if params['CGL']:
        dpath += '' if β  is None else f'β{β:.3e}'
        dpath += '' if Δβ is None else f'Δβ{Δβ:+.2f}'
    dpath += '' if ξ is None else f'ξ{ξ:.3e}'
    dpath += '' if ϵ is None else f'ϵ{ϵ:.3e}'
    dpath += '' if a is None else f'a{a:.3e}'
    dpath += '' if w is None else f'w{w:.3e}'

    return dpath + params['suffix']


def load_config(file_path: str) -> Dict[str, Any]:
    """
    Read a simple key-value configuration file.
    """
    config: Dict[str, Any] = {}
    if not os.path.exists(file_path):
        return config

    with open(file_path, 'r') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            if '=' in line:
                key, val = line.split('=', 1)
            elif ':' in line:
                key, val = line.split(':', 1)
            else:
                continue
            config[key.strip()] = val.strip()
    return config


def save_eigenmode(file_path: str, **kwargs: Any) -> None:
    """
    Save eigenmode data to a .npz file atomically.
    """
    import tempfile
    dir_name = os.path.dirname(file_path)
    # Ensure directory exists
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **kwargs)
        os.replace(tmp_path, file_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e


def load_eigenmodes(path: str, pattern: str = "*.npz") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load eigenmode data from .npz files using glob and os.
    """
    search_path = os.path.join(path, pattern)
    files = sorted(glob.glob(search_path))

    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found in {path}")

    rows = []
    for f in files:
        with np.load(f) as state:
            if 'scan_parameter' in state:
                val = float(state['scan_parameter_value'])
            elif 'dependence' in state:
                val = float(state['value'])
            else:
                val = float(state['wavenumber'])

            growth = state['eigenvalue'] if 'eigenvalue' in state else state['growth_rate']
            if isinstance(growth, np.ndarray) and growth.ndim > 0:
                growth = growth[0]

            wavenumber = float(state['wavenumber'])
            dlt = float(state['resistive_layer_thickness']) if 'resistive_layer_thickness' in state else float(state['inner_scale'])
            tol = float(state['tolerance'])
            scaling = float(state['grid_scaling_factor']) if 'grid_scaling_factor' in state else float(state['scaling_factor'])
            res = int(state['resolution'])
            nin_val = int(state['resistive_layer_nodes']) if 'resistive_layer_nodes' in state else int(state['n_inner'])
            nwa_val = int(state['current_sheet_nodes']) if 'current_sheet_nodes' in state else int(state['n_wa'])

            rows.append([
                val,
                wavenumber,
                growth,
                tol,
                dlt,
                nin_val,
                nwa_val,
                scaling,
                res
            ])

    rows.sort()

    v = np.array([x[0] for x in rows])
    α = np.array([x[1] for x in rows])
    σ = np.array([x[2] for x in rows])
    e = np.array([x[3] for x in rows])
    δ = np.array([x[4] for x in rows])
    c = np.array([x[7] for x in rows])
    nin_arr = np.array([x[5] for x in rows])
    nwa_arr = np.array([x[6] for x in rows])
    N = np.array([x[8] for x in rows])

    return v, α, σ, e, δ, c, nin_arr, nwa_arr, N


def load_eig_scales(path: str, pattern: str = "*.npz") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load eigenmode data from .npz files focusing on scales.
    """
    search_path = os.path.join(path, pattern)
    files = sorted(glob.glob(search_path))

    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found in {path}")

    rows = []
    for f in files:
        with np.load(f) as state:
            if 'scan_parameter' in state:
                val = float(state['scan_parameter_value'])
            elif 'dependence' in state:
                val = float(state['value'])
            else:
                val = float(state['wavenumber'])
            nwa_val = int(state['current_sheet_nodes']) if 'current_sheet_nodes' in state else int(state['n_wa'])
            rows.append([val, nwa_val])

    rows.sort()
    v = np.array([x[0] for x in rows])
    nwa_arr = np.array([x[1] for x in rows])

    return v, nwa_arr


def write_results(params: Dict[str, Any], delta_time: float) -> None:
    """
    Unified result writer for simulation output.
    """
    dpath = params['data_path']
    if not dpath or not os.path.exists(dpath):
        raise FileNotFoundError(f"Data path {dpath!r} does not exist")

    try:
        v, α, σ, e, δ, c, nin, nwa, N = load_eigenmodes(dpath)
    except FileNotFoundError:
        logger.warning(f"No results found in {dpath}. Skipping .dat file creation.")
        return

    fname = f"{dpath}.dat"
    dep_key = params.get('scan_parameter', params.get('dependence'))

    with open(fname, 'w') as io:
        io.write(f"#\n# Tearing Instability - mode {params['mode']}\n#\n")
        io.write("# Plasma parameters:\n")
        eqns = "Gyrotropic" if params.get('CGL') else "Classical"
        io.write(f"#   Equations                            =   {eqns} MHD\n")
        if params.get('CGL'):
            io.write(f"#   Equation of State                    =   {params.get('eos')}\n")

        def write_head(label, key, fmt="10.3e"):
            val = params.get(key)
            if val is not None:
                io.write(f"#   {label:<36} =  {val:{fmt}}\n")

        write_head("Lundquist number (S)", 'S')
        write_head("Prandtl number (Pr)", 'Pr')
        if params.get('CGL'):
            write_head("Plasma-β (β)", 'plasma_beta')
            write_head("Plasma-β difference (Δβ)", 'plasma_beta_difference')
        write_head("Magnetic transverse field (ξ)", 'xi')
        write_head("Hall current term strength (ϵ)", 'Hall')
        if params.get('CGL'):
            write_head("  Parallel adiabatic index (γpar)", 'parallel_index')
            write_head("  Perpendicular adiabatic index (γper)", 'perpendicular_index')

        io.write("# Equilibrium parameters:\n")
        write_head("Current sheet thickness (a)", 'a')
        write_head("Current sheet half-width (w)", 'w')

        io.write("# Geometry/convergence parameters:\n")
        io.write(f"#   {'Resolution range (N)':<36} =  [{params['Nmin']}, {params['Nmax']}] increment {params['Ninc']}\n")
        io.write(f"#   {'Real part range':<36} =  [{params['sigma_real_lower']}, {params['sigma_real_upper']}]\n")
        io.write(f"#   {'Imaginary part range':<36} =  [{params['sigma_imag_lower']}, {params['sigma_imag_upper']}]\n")

        if params.get('sigma_imag') is not None:
            io.write(f"#   {'Imaginary amplitude limit':<36} =  {params['sigma_imag']:10.3e}\n")

        write_head("Growth rate absolute tolerance", 'atol')
        write_head("Growth rate relative tolerance", 'rtol')
        write_head("Growth rate guess tolerance", 'gtol')
        if 'ktol' in params:
            write_head("Wavenumber relative tolerance", 'ktol')

        io.write(f"#   {'Selection order':<36} =   {params['orderby']}\n")
        io.write(f"#   {'Converge the mode':<36} =   {params['mode']}\n")
        write_head("Inner-layer thickness tolerance", 'dtol')
        io.write(f"#   {'Number of inner collocation points':<36} =   {params['n_inner']}\n")
        io.write(f"#   {'Amplitude fraction at zmax':<36} =   {params['f_outer']}\n")
        io.write(f"#   {'Decay e-folds at zmax':<36} =  {params['decay_efolds']:10.3e}\n")

        io.write(f"#\n# Calculation done in {delta_time:.2f} seconds.\n#\n")

        if dep_key:
            io.write(f"#    {dep_key:<2s}               α_max            Re(σ_max)        Im(σ_max)        δ_in             tolerance        C              n_in    n_wa    N\n")
            io.write("#" + " --------------- "*7 + " ------  ------  ------" + "\n")
        else:
            io.write("#    α                Re(σ)            Im(σ)            δ_in             tolerance        C              n_in    n_wa    N\n")
            io.write("#" + " --------------- "*6 + " ------  ------  ------" + "\n")

        for i in range(len(α)):
            if dep_key:
                io.write(f"  {v[i]:15.8e}  {α[i]:15.8e}  {σ[i].real:15.8e}  {σ[i].imag:15.8e}  {δ[i]:15.8e}  {e[i]:15.8e}  {c[i]:15.8e}  {nin[i]:>6d}  {nwa[i]:>6d}  {N[i]:>6d}\n")
            else:
                io.write(f"  {α[i]:15.8e}  {σ[i].real:15.8e}  {σ[i].imag:15.8e}  {δ[i]:15.8e}  {e[i]:15.8e}  {c[i]:15.8e}  {nin[i]:>6d}  {nwa[i]:>6d}  {N[i]:>6d}\n")

        io.flush()
