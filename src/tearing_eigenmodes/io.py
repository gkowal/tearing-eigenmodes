import os
import glob
import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


from .params import SimulationParams

def build_dpath(params: SimulationParams) -> str:
    """
    Construct the directory name that will hold the results for a given run.
    """
    a, w = params.a, params.w
    S, Pr = params.S, params.Pr
    ξ, ϵ = params.xi, params.Hall
    β, Δβ = params.plasma_beta, params.plasma_beta_difference

    dpath  = './RESULTS/'
    dpath += '' if S  is None else f'S{S:.3e}'
    dpath += '' if Pr is None else f'Pr{Pr:.3e}'
    if params.CGL:
        dpath += '' if β  is None else f'β{β:.3e}'
        dpath += '' if Δβ is None else f'Δβ{Δβ:+.2f}'
    dpath += '' if ξ is None else f'ξ{ξ:.3e}'
    dpath += '' if ϵ is None else f'ϵ{ϵ:.3e}'
    dpath += '' if a is None else f'a{a:.3e}'
    dpath += '' if w is None else f'w{w:.3e}'

    return dpath + params.suffix


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


from .analysis import MODE_SCALE_SCHEMA_VERSION


def save_eigenmode(file_path: str, **kwargs: Any) -> None:
    """
    Save eigenmode data to a .npz file atomically without pickle or object arrays.
    """
    import tempfile
    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Encode mode_scales dictionary as parallel NumPy arrays if passed
    if "mode_scales" in kwargs:
        mode_scales = kwargs.pop("mode_scales")
        if isinstance(mode_scales, dict):
            sorted_keys = sorted(mode_scales.keys())
            kwargs["mode_scale_schema_version"] = np.int32(MODE_SCALE_SCHEMA_VERSION)
            kwargs["mode_scale_keys"] = np.array(sorted_keys, dtype=str)
            kwargs["mode_scale_values"] = np.array([float(mode_scales[k]) for k in sorted_keys], dtype=np.float64)

    # Ensure scalar fields have proper numpy types avoiding object serialization
    if "minimum_physical_scale" in kwargs and kwargs["minimum_physical_scale"] is not None:
        kwargs["minimum_physical_scale"] = np.float64(kwargs["minimum_physical_scale"])
    if "minimum_physical_scale_key" in kwargs and kwargs["minimum_physical_scale_key"] is not None:
        kwargs["minimum_physical_scale_key"] = np.str_(kwargs["minimum_physical_scale_key"])
    if "minimum_scale_nodes" in kwargs and kwargs["minimum_scale_nodes"] is not None:
        kwargs["minimum_scale_nodes"] = np.int32(kwargs["minimum_scale_nodes"])
    if "resistive_layer_thickness" in kwargs and kwargs["resistive_layer_thickness"] is not None:
        kwargs["resistive_layer_thickness"] = np.float64(kwargs["resistive_layer_thickness"])
    if "resistive_layer_nodes" in kwargs and kwargs["resistive_layer_nodes"] is not None:
        kwargs["resistive_layer_nodes"] = np.int32(kwargs["resistive_layer_nodes"])

    fd, tmp_path = tempfile.mkstemp(dir=dir_name if dir_name else ".", suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **kwargs)
        os.replace(tmp_path, file_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e


def load_state_data(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Read and decode all state variables and reconstructed scale metadata from a .npz file.
    Does not evaluate convergence or reuse policy.
    """
    if not os.path.exists(file_path):
        return None

    try:
        with np.load(file_path, allow_pickle=True) as state:
            data = {}
            for key in state.files:
                val = state[key]
                if isinstance(val, np.ndarray) and val.ndim == 0:
                    data[key] = val.item()
                else:
                    data[key] = val

            # Reconstruct standardized mode_scales dictionary if present
            if 'mode_scale_keys' in data and 'mode_scale_values' in data:
                keys = data['mode_scale_keys']
                values = data['mode_scale_values']
                if len(keys) != len(values):
                    logger.warning(f"Malformed scale data in {file_path}: mismatched key/value lengths.")
                else:
                    schema_ver = int(data.get('mode_scale_schema_version', 1))
                    if schema_ver > MODE_SCALE_SCHEMA_VERSION:
                        logger.warning(f"State file {file_path} uses newer schema version {schema_ver}.")
                    data['mode_scales'] = {str(k): float(v) for k, v in zip(keys, values)}

            # Fallback for minimum physical scale and node counts on old states
            if 'minimum_physical_scale' not in data:
                if 'resistive_layer_thickness' in data:
                    data['minimum_physical_scale'] = float(data['resistive_layer_thickness'])
                elif 'inner_scale' in data:
                    data['minimum_physical_scale'] = float(data['inner_scale'])
                else:
                    data['minimum_physical_scale'] = float("nan")

            if 'minimum_physical_scale_key' not in data:
                data['minimum_physical_scale_key'] = ""

            if 'minimum_scale_nodes' not in data:
                if 'resistive_layer_nodes' in data:
                    data['minimum_scale_nodes'] = int(data['resistive_layer_nodes'])
                elif 'n_inner' in data:
                    data['minimum_scale_nodes'] = int(data['n_inner'])
                else:
                    data['minimum_scale_nodes'] = 0

            return data
    except Exception as e_err:
        logger.warning(f"Could not load state file {file_path}: {e_err}")
        return None


def check_state(file_path: str, force: bool = False, Nmax: int = 2048) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Check if a state file exists and contains a completed/converged calculation.

    Parameters
    ----------
    file_path : str
        Path to the state .npz file.
    force : bool
        If True, force recalculation regardless of state file.
    Nmax : int
        Maximum resolution limit.

    Returns
    -------
    tuple
        (status, state_data) where status is True if we can reuse the results,
        and state_data is a dictionary containing the loaded variables from the file.
    """
    if force or not os.path.exists(file_path):
        return False, None

    data = load_state_data(file_path)
    if data is None:
        return False, None

    e = data.get('tolerance')
    N = data.get('resolution')
    if e is not None and N is not None:
        e_val = np.atleast_1d(e)[0]
        status = not (float(e_val) > 1.0 and int(N) < Nmax)
        return status, data

    return False, None


def compile_metadata(params: SimulationParams) -> Dict[str, Any]:
    """
    Compile physical and numerical parameters for saving in the state file.
    """
    metadata_keys = [
        'S', 'Pr', 'plasma_beta', 'plasma_beta_difference', 'xi', 'Hall',
        'a', 'w', 'parallel_index', 'perpendicular_index', 'eos', 'CGL',
        'noshear', 'Nmin', 'Nmax', 'Ninc', 'atol', 'rtol', 'gtol', 'dtol',
        'f_outer', 'mode', 'dynamic_C', 'n_anisotropy', 'inner_scale',
        'inner_resolution_safety', 'n_equilibrium', 'n_inner_scale'
    ]
    metadata = {k: getattr(params, k) for k in metadata_keys}
    metadata['n_inner_req'] = params.n_equilibrium
    metadata['n_resistivity_req'] = params.n_inner_scale

    # Optional dependence parameter
    dep_key = params.scan_parameter or params.dependence
    if dep_key:
        metadata['scan_parameter'] = dep_key

    return metadata


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

            if 'minimum_physical_scale' in state:
                dlt = float(state['minimum_physical_scale'])
            elif 'resistive_layer_thickness' in state:
                dlt = float(state['resistive_layer_thickness'])
            else:
                dlt = float(state['inner_scale'])

            tol = float(state['tolerance'])
            scaling = float(state['grid_scaling_factor']) if 'grid_scaling_factor' in state else float(state['scaling_factor'])
            res = int(state['resolution'])

            if 'minimum_scale_nodes' in state:
                nin_val = int(state['minimum_scale_nodes'])
            elif 'resistive_layer_nodes' in state:
                nin_val = int(state['resistive_layer_nodes'])
            else:
                nin_val = int(state['n_inner'])

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


def write_results(params: SimulationParams, delta_time: float) -> None:
    """
    Unified result writer for simulation output.
    """
    dpath = params.data_path
    if not dpath or not os.path.exists(dpath):
        raise FileNotFoundError(f"Data path {dpath!r} does not exist")

    try:
        v, α, σ, e, δ, c, nin, nwa, N = load_eigenmodes(dpath)
    except FileNotFoundError:
        logger.warning(f"No results found in {dpath}. Skipping .dat file creation.")
        return

    fname = f"{dpath}.dat"
    dep_key = params.scan_parameter or params.dependence

    with open(fname, 'w') as io:
        io.write(f"#\n# Tearing Instability - mode {params.mode}\n#\n")
        io.write("# Plasma parameters:\n")
        eqns = "Gyrotropic" if params.CGL else "Classical"
        io.write(f"#   Equations                            =   {eqns} MHD\n")
        if params.CGL:
            io.write(f"#   Equation of State                    =   {params.eos}\n")

        def write_head(label, key, fmt="10.3e"):
            val = getattr(params, key)
            if val is not None:
                io.write(f"#   {label:<36} =  {val:{fmt}}\n")

        write_head("Lundquist number (S)", 'S')
        write_head("Prandtl number (Pr)", 'Pr')
        if params.CGL:
            write_head("Plasma-β (β)", 'plasma_beta')
            write_head("Plasma-β difference (Δβ)", 'plasma_beta_difference')
        write_head("Magnetic transverse field (ξ)", 'xi')
        write_head("Hall current term strength (ϵ)", 'Hall')
        if params.CGL:
            write_head("  Parallel adiabatic index (γpar)", 'parallel_index')
            write_head("  Perpendicular adiabatic index (γper)", 'perpendicular_index')

        io.write("# Equilibrium parameters:\n")
        write_head("Current sheet thickness (a)", 'a')
        write_head("Current sheet half-width (w)", 'w')

        io.write("# Geometry/convergence parameters:\n")
        io.write(f"#   {'Resolution range (N)':<36} =  [{params.Nmin}, {params.Nmax}] increment {params.Ninc}\n")
        io.write(f"#   {'Real part range':<36} =  [{params.sigma_real_lower}, {params.sigma_real_upper}]\n")
        io.write(f"#   {'Imaginary part range':<36} =  [{params.sigma_imag_lower}, {params.sigma_imag_upper}]\n")

        write_head("Growth rate absolute tolerance", 'atol')
        write_head("Growth rate relative tolerance", 'rtol')
        write_head("Growth rate guess tolerance", 'gtol')
        if params.ktol is not None:
            write_head("Wavenumber relative tolerance", 'ktol')

        io.write(f"#   {'Selection order':<36} =   {params.orderby}\n")
        io.write(f"#   {'Converge the mode':<36} =   {params.mode}\n")
        write_head("Inner scale (δin)", 'inner_scale')
        write_head("Inner resolution safety", 'inner_resolution_safety')
        write_head("Inner-layer thickness tolerance", 'dtol')
        io.write(f"#   {'Equilibrium collocation points':<36} =   {params.n_equilibrium}\n")
        io.write(f"#   {'Inner scale collocation points':<36} =   {params.n_inner_scale}\n")
        io.write(f"#   {'Anisotropy scale points':<36} =   {params.n_anisotropy}\n")
        io.write(f"#   {'Dynamic C grid':<36} =   {'on' if params.dynamic_C else 'off'}\n")
        io.write(f"#   {'Amplitude fraction at zmax':<36} =   {params.f_outer}\n")
        io.write(f"#   {'Decay e-folds at zmax':<36} =  {params.decay_efolds:10.3e}\n")

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
