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
        dpath += '' if Δβ is None else f'Δβ{Δβ:+.3e}'
        dpath += '' if params.eos is None else params.eos
        parallel, perpendicular = params.parallel_index, params.perpendicular_index
        dpath += '' if parallel is None else f'ɣpar{parallel:.3e}'
        dpath += '' if perpendicular is None else f'ɣper{perpendicular:.3e}'
    else:
        dpath += '' if params.zeta is None else f'ζ{params.zeta:.3e}'
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
    Save eigenmode data to a .npz file atomically.
    Non-None values use typed NumPy scalars/arrays (no pickled object arrays);
    None-valued optional metadata is stored as pickled object arrays, so loaders
    keep allow_pickle=True for legacy states and None-valued current metadata.
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

    fd, tmp_path = tempfile.mkstemp(dir=dir_name if dir_name else ".", suffix=".npz.tmp")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as tmp_file:
            np.savez_compressed(tmp_file, **kwargs)
        os.replace(tmp_path, file_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e


def _positive_integer_scalar(value: Any) -> Optional[int]:
    """
    Return one finite, positive, exactly integer-valued scalar as Python int.
    Safely returns None for bool, NaN, inf, <= 0, fractional values (e.g. 1.5, '2048.9'),
    empty/multi-element arrays, None, strings with fractional parts, or arbitrary objects.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return None

    if isinstance(value, (str, bytes)):
        try:
            s = value.decode("utf-8") if isinstance(value, bytes) else value
            s = s.strip()
            try:
                val_int = int(s)
                return val_int if val_int > 0 else None
            except ValueError:
                f = float(s)
                if np.isfinite(f) and f > 0.0 and f.is_integer():
                    return int(f)
                return None
        except Exception:
            return None

    try:
        arr = np.asanyarray(value)
        if arr.ndim != 0 and arr.size != 1:
            return None
        elem = arr.item() if arr.ndim == 0 else arr.flat[0]
        if isinstance(elem, (bool, np.bool_)):
            return None
        if isinstance(elem, (int, np.integer)):
            val = int(elem)
            return val if val > 0 else None
        if isinstance(elem, (float, np.floating)):
            if np.isfinite(elem) and elem > 0.0:
                f_elem = float(elem)
                if f_elem.is_integer():
                    return int(f_elem)
            return None
        if isinstance(elem, (str, bytes)):
            return _positive_integer_scalar(elem)
    except Exception:
        return None

    return None


def load_state_data(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Read and decode all state variables and reconstructed scale metadata from a .npz file.
    Does not evaluate convergence or reuse policy.
    Uses allow_pickle=True intentionally to support legacy object-array states.
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
    except Exception as e_err:
        logger.warning(f"Could not load state file {file_path}: {e_err}")
        return None

    # Reconstruct standardized mode_scales dictionary if present
    if 'mode_scale_keys' in data and 'mode_scale_values' in data:
        keys = data['mode_scale_keys']
        values = data['mode_scale_values']

        keys_1d = isinstance(keys, (np.ndarray, list, tuple)) and np.ndim(keys) == 1
        values_1d = isinstance(values, (np.ndarray, list, tuple)) and np.ndim(values) == 1

        if not keys_1d or not values_1d or len(keys) != len(values):
            logger.warning(
                f"Malformed scale data in {file_path}: mode_scale_keys and mode_scale_values "
                "must be one-dimensional sequences of equal length."
            )
            data['mode_scales'] = {}
        else:
            schema_valid = True
            schema_ver = 1
            if 'mode_scale_schema_version' in data:
                raw_schema = data['mode_scale_schema_version']
                parsed_ver = _positive_integer_scalar(raw_schema)
                if parsed_ver is None:
                    logger.warning(f"Malformed mode_scale_schema_version in {file_path}: {raw_schema}")
                    schema_valid = False
                elif parsed_ver > MODE_SCALE_SCHEMA_VERSION:
                    logger.warning(f"State file {file_path} uses newer schema version {parsed_ver}.")
                    schema_valid = False
                else:
                    schema_ver = parsed_ver

            if not schema_valid:
                data['mode_scales'] = {}
            elif schema_ver == 1:
                # Schema version 1 stored envelope values under eta_vs_ideal / nu_vs_ideal.
                # Remap them to *_envelope and do not treat them as net-ideal scales.
                mode_scales: Dict[str, float] = {}
                for k, v in zip(keys, values):
                    key_str = str(k)
                    try:
                        if isinstance(v, np.ndarray) and v.ndim == 0:
                            v = v.item()
                        val_float = float(v)
                    except Exception:
                        val_float = float("nan")

                    if key_str == "classical.bz_induction.eta_vs_ideal":
                        mode_scales["classical.bz_induction.eta_vs_ideal_envelope"] = val_float
                        mode_scales["classical.bz_induction.eta_vs_ideal"] = float("nan")
                    elif key_str == "classical.uz_vorticity.nu_vs_ideal":
                        mode_scales["classical.uz_vorticity.nu_vs_ideal_envelope"] = val_float
                        mode_scales["classical.uz_vorticity.nu_vs_ideal"] = float("nan")
                    else:
                        mode_scales[key_str] = val_float
                data['mode_scales'] = mode_scales
            else:
                mode_scales = {}
                for k, v in zip(keys, values):
                    key_str = str(k)
                    try:
                        if isinstance(v, np.ndarray) and v.ndim == 0:
                            v = v.item()
                        val_float = float(v)
                        mode_scales[key_str] = val_float
                    except Exception:
                        logger.warning(f"Malformed scale value for key '{key_str}' in {file_path}: {v}")
                        mode_scales[key_str] = float("nan")
                data['mode_scales'] = mode_scales

    # Normalize scalar fallbacks independently
    if 'minimum_physical_scale' not in data:
        if 'resistive_layer_thickness' in data:
            try:
                v = data['resistive_layer_thickness']
                if isinstance(v, np.ndarray) and v.ndim == 0:
                    v = v.item()
                data['minimum_physical_scale'] = float(v)
            except Exception:
                data['minimum_physical_scale'] = float("nan")
        elif 'inner_scale' in data:
            try:
                v = data['inner_scale']
                if isinstance(v, np.ndarray) and v.ndim == 0:
                    v = v.item()
                data['minimum_physical_scale'] = float(v)
            except Exception:
                data['minimum_physical_scale'] = float("nan")
        else:
            data['minimum_physical_scale'] = float("nan")
    else:
        try:
            v = data['minimum_physical_scale']
            if isinstance(v, np.ndarray) and v.ndim == 0:
                v = v.item()
            data['minimum_physical_scale'] = float(v)
        except Exception:
            data['minimum_physical_scale'] = float("nan")

    if 'minimum_physical_scale_key' not in data:
        data['minimum_physical_scale_key'] = ""
    else:
        try:
            data['minimum_physical_scale_key'] = str(data['minimum_physical_scale_key'])
        except Exception:
            data['minimum_physical_scale_key'] = ""

    if 'minimum_scale_nodes' not in data:
        if 'resistive_layer_nodes' in data:
            try:
                v = data['resistive_layer_nodes']
                if isinstance(v, np.ndarray) and v.ndim == 0:
                    v = v.item()
                data['minimum_scale_nodes'] = int(v)
            except Exception:
                data['minimum_scale_nodes'] = 0
        elif 'n_inner' in data:
            try:
                v = data['n_inner']
                if isinstance(v, np.ndarray) and v.ndim == 0:
                    v = v.item()
                data['minimum_scale_nodes'] = int(v)
            except Exception:
                data['minimum_scale_nodes'] = 0
        else:
            data['minimum_scale_nodes'] = 0
    else:
        try:
            v = data['minimum_scale_nodes']
            if isinstance(v, np.ndarray) and v.ndim == 0:
                v = v.item()
            data['minimum_scale_nodes'] = int(v)
        except Exception:
            data['minimum_scale_nodes'] = 0

    if 'resistive_layer_thickness' in data:
        try:
            v = data['resistive_layer_thickness']
            if isinstance(v, np.ndarray) and v.ndim == 0:
                v = v.item()
            data['resistive_layer_thickness'] = float(v)
        except Exception:
            data['resistive_layer_thickness'] = float("nan")

    if 'resistive_layer_nodes' in data:
        try:
            v = data['resistive_layer_nodes']
            if isinstance(v, np.ndarray) and v.ndim == 0:
                v = v.item()
            data['resistive_layer_nodes'] = int(v)
        except Exception:
            data['resistive_layer_nodes'] = 0

    return data


_METADATA_MISSING: Any = object()


# Physics/problem/mode identity keys used when deciding state reuse in
# ``check_state``. Discretization, search, and acceptance knobs (Nmin/Nmax/
# Ninc, atol/rtol/gtol/dtol, dynamic_C, n_anisotropy, inner_scale,
# inner_resolution_safety, n_equilibrium, n_inner_scale, n_inner_req,
# n_resistivity_req) never enter the equations, so a converged (e<=1)
# stored result stays valid when only those knobs change. ``compile_metadata``
# itself stays complete (files remain self-describing); only the comparison
# loop in ``check_state`` is narrowed to this allowlist.
_REUSE_COMPARE_KEYS = (
    'S', 'Pr', 'plasma_beta', 'plasma_beta_difference', 'xi', 'Hall',
    'zeta', 'a', 'w', 'parallel_index', 'perpendicular_index', 'eos',
    'CGL', 'noshear', 'f_outer', 'mode', 'scan_parameter',
)


def _unwrap_metadata_scalar(value: Any) -> Any:
    """Unwrap 0-d ndarrays to plain Python scalars for metadata comparison."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        try:
            return value.item()
        except Exception:
            return value
    return value


def _metadata_values_equal(first: Any, second: Any) -> bool:
    """
    Conservative equality check for run-config metadata values.

    Numbers compare numerically (NaN never equals, forcing a recompute);
    None equals only None; strings and bools compare with ==; anything
    else compares with == and any failure (including exceptions) counts
    as a mismatch.
    """
    first = _unwrap_metadata_scalar(first)
    second = _unwrap_metadata_scalar(second)
    if first is None and second is None:
        return True
    if first is None or second is None:
        return False
    if isinstance(first, str) or isinstance(second, str):
        try:
            return bool(first == second)
        except Exception:
            return False
    if isinstance(first, (bool, np.bool_)) or isinstance(second, (bool, np.bool_)):
        try:
            return bool(first == second)
        except Exception:
            return False
    if isinstance(first, (int, float, np.integer, np.floating)) and isinstance(
        second, (int, float, np.integer, np.floating)
    ):
        try:
            first_float = float(first)
            second_float = float(second)
        except Exception:
            return False
        if np.isnan(first_float) or np.isnan(second_float):
            return False
        return first_float == second_float
    try:
        result = first == second
        if isinstance(result, np.ndarray):
            return bool(np.all(result))
        return bool(result)
    except Exception:
        return False


def check_state(file_path: str, force: bool = False, Nmax: int = 2048,
                params: Optional[SimulationParams] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
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
    params : SimulationParams, optional
        Current run configuration. When given, the stored run-config
        metadata is compared against ``compile_metadata(params)`` over the
        physics/problem/mode identity keys in ``_REUSE_COMPARE_KEYS`` only
        (discretization/search/acceptance knobs never enter the equations,
        so changes to them alone reuse a converged result); any mismatch
        (or any missing key, e.g. in legacy files) forces a
        recalculation returning (False, None). A stored resolution below
        ``params.Nmin`` also forces recalculation. When None (default),
        only the historical tolerance/resolution policy applies.

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
    if e is None or N is None:
        logger.warning(f"State file {file_path} missing tolerance or resolution reuse metadata.")
        return False, None

    N_val = _positive_integer_scalar(N)
    if N_val is None:
        logger.warning(f"Invalid resolution in state file {file_path}: {N}")
        return False, None

    if isinstance(e, (bool, np.bool_)):
        logger.warning(f"Invalid boolean tolerance in state file {file_path}: {e}")
        return False, None

    try:
        e_arr = np.asanyarray(e)
        if e_arr.ndim != 0 and e_arr.size != 1:
            logger.warning(f"Invalid tolerance array in state file {file_path}: {e}")
            return False, None
        elem = e_arr.item() if e_arr.ndim == 0 else e_arr.flat[0]
        if isinstance(elem, (bool, np.bool_)):
            logger.warning(f"Invalid boolean tolerance in state file {file_path}: {e}")
            return False, None
        e_val = float(elem)
        if not np.isfinite(e_val) or e_val < 0.0:
            logger.warning(f"Non-finite or negative tolerance in state file {file_path}: {e_val}")
            return False, None

        status = not (e_val > 1.0 and N_val < Nmax)
        if status and params is not None:
            nmin_val = params.Nmin
            if nmin_val is not None and N_val < nmin_val:
                logger.debug(
                    f"State file {file_path} resolution N={N_val} below "
                    f"requested Nmin={nmin_val}: forcing recalculation."
                )
                return False, None
            fresh = compile_metadata(params)
            for key in _REUSE_COMPARE_KEYS:
                fresh_val = fresh.get(key)
                stored_val = data.get(key, _METADATA_MISSING)
                if stored_val is _METADATA_MISSING:
                    if fresh_val is None:
                        # Absent on both sides (e.g. conditional
                        # scan_parameter or an unset optional).
                        continue
                    logger.debug(
                        f"State file {file_path} missing run-config key "
                        f"'{key}': forcing recalculation."
                    )
                    return False, None
                if not _metadata_values_equal(fresh_val, stored_val):
                    logger.debug(
                        f"State file {file_path} run-config mismatch for "
                        f"'{key}': forcing recalculation."
                    )
                    return False, None
        return status, data
    except Exception as ex:
        logger.warning(f"Malformed reuse metadata in state file {file_path}: {ex}")
        return False, None


def compile_metadata(params: SimulationParams) -> Dict[str, Any]:
    """
    Compile physical and numerical parameters for saving in the state file.
    """
    metadata_keys = [
        'S', 'Pr', 'plasma_beta', 'plasma_beta_difference', 'xi', 'Hall',
        'zeta', 'a', 'w', 'parallel_index', 'perpendicular_index', 'eos', 'CGL',
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


def reuse_identity(params: SimulationParams) -> Dict[str, Any]:
    """
    The set physics/problem/mode identity values of a run configuration
    (``_REUSE_COMPARE_KEYS`` restricted to values that are not None; a
    swept parameter is None in the sweep-level configuration).
    """
    fresh = compile_metadata(params)
    return {k: fresh[k] for k in _REUSE_COMPARE_KEYS if fresh.get(k) is not None}


def state_matches_identity(data: Dict[str, Any], identity: Dict[str, Any]) -> bool:
    """
    Whether a loaded state belongs to the run with the given reuse identity.
    Keys absent from the state (legacy files), and numeric keys whose stored
    value is malformed (not a finite number, or not positive for S and a),
    do not exclude it.
    """
    for key, fresh_val in identity.items():
        stored_val = data.get(key, _METADATA_MISSING)
        if stored_val is _METADATA_MISSING:
            continue
        numeric = isinstance(fresh_val, (int, float, np.integer, np.floating)) and \
            not isinstance(fresh_val, (bool, np.bool_))
        if numeric:
            stored_scalar = _unwrap_metadata_scalar(stored_val)
            if isinstance(stored_scalar, (bool, np.bool_)) or \
                    not isinstance(stored_scalar, (int, float, np.integer, np.floating)) or \
                    not np.isfinite(float(stored_scalar)) or \
                    (key in ('S', 'a') and float(stored_scalar) <= 0.0):
                continue
        if not _metadata_values_equal(fresh_val, stored_val):
            return False
    return True


def load_eigenmodes(
    path: str,
    pattern: str = "*.npz",
    scale_key: Optional[str] = None,
    params: Optional[SimulationParams] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load eigenmode data from .npz files using glob and os.
    Reuses load_state_data for tolerant decoding of optional multiscale fields.

    Parameters
    ----------
    path : str
        Directory containing state files.
    pattern : str, optional
        Glob pattern for state files, default '*.npz'.
    scale_key : str, optional
        Specific physical scale key to extract as delta (e.g. 'classical.bz_induction.eta_vs_ideal').
        If None (default), returns the envelope minimum_physical_scale.
    params : SimulationParams, optional
        Run configuration. When given, states whose stored reuse identity
        (e.g. noshear or mode, which result paths do not encode) differs
        from it are skipped, so one directory can hold several variants.
    """
    search_path = os.path.join(path, pattern)
    files = sorted(glob.glob(search_path))

    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found in {path}")

    identity = reuse_identity(params) if params is not None else None

    rows = []
    for f in files:
        data = load_state_data(f)
        if data is None:
            continue
        if identity is not None and not state_matches_identity(data, identity):
            logger.debug(f"Skipping {f}: run-config identity differs from the current run.")
            continue

        if 'scan_parameter' in data:
            val = float(data['scan_parameter_value'])
        elif 'dependence' in data:
            val = float(data['value'])
        else:
            val = float(data['wavenumber'])

        growth = data.get('eigenvalue', data.get('growth_rate'))
        if isinstance(growth, np.ndarray) and growth.ndim > 0:
            growth = growth[0]

        wavenumber = float(data['wavenumber'])

        if scale_key is not None:
            if 'mode_scales' in data:
                mode_scales = data['mode_scales']
                if isinstance(mode_scales, dict) and scale_key in mode_scales:
                    try:
                        dlt = float(mode_scales[scale_key])
                    except Exception:
                        dlt = float('nan')
                else:
                    dlt = float('nan')
            elif scale_key in ("classical.bz_induction.eta_vs_ideal", "cgl.bz_induction.eta_vs_ideal", "resistive_layer_thickness") and 'resistive_layer_thickness' in data:
                try:
                    dlt = float(data['resistive_layer_thickness'])
                except Exception:
                    dlt = float('nan')
            else:
                dlt = float('nan')
        else:
            dlt = float(data.get('minimum_physical_scale', float('nan')))

        tol_raw = data.get('tolerance', float('nan'))
        if isinstance(tol_raw, np.ndarray) and tol_raw.size > 0:
            tol = float(tol_raw.flat[0])
        elif tol_raw is not None:
            try:
                tol = float(tol_raw)
            except Exception:
                tol = float('nan')
        else:
            tol = float('nan')

        scaling_raw = data.get('grid_scaling_factor', data.get('scaling_factor', float('nan')))
        if isinstance(scaling_raw, np.ndarray) and scaling_raw.size > 0:
            scaling = float(scaling_raw.flat[0])
        elif scaling_raw is not None:
            try:
                scaling = float(scaling_raw)
            except Exception:
                scaling = float('nan')
        else:
            scaling = float('nan')

        res_raw = data.get('resolution', 0)
        res_int = _positive_integer_scalar(res_raw)
        res = res_int if res_int is not None else 0

        nin_val = int(data.get('minimum_scale_nodes', 0))

        nwa_raw = data.get('current_sheet_nodes', data.get('n_wa', 0))
        nwa_int = _positive_integer_scalar(nwa_raw)
        nwa_val = nwa_int if nwa_int is not None else 0

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

    if not rows:
        raise FileNotFoundError(f"No valid state data loaded matching {pattern} in {path}")

    # Sort by coordinates only: comparing complex growth rates on ties raises
    rows.sort(key=lambda row: (row[0], row[1]))

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
        v, α, σ, e, δ, c, nin, nwa, N = load_eigenmodes(dpath, params=params)
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
        if not params.CGL:
            write_head("Magnetic transverse field (ξ)", 'xi')
        write_head("Hall current term strength (ϵ)", 'Hall')
        if params.CGL:
            write_head("  Parallel adiabatic index (γpar)", 'parallel_index')
            write_head("  Perpendicular adiabatic index (γper)", 'perpendicular_index')

        io.write("# Equilibrium parameters:\n")
        write_head("Current sheet thickness (a)", 'a')
        if not params.CGL:
            write_head("Current sheet half-width (w)", 'w')
            write_head("Shear parameter (ζ)", 'zeta')

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
        if params.CGL:
            io.write(f"#   {'Anisotropy scale points':<36} =   {params.n_anisotropy}\n")
        io.write(f"#   {'Dynamic C grid':<36} =   {'on' if params.dynamic_C else 'off'}\n")
        io.write(f"#   {'Amplitude fraction at zmax':<36} =   {params.f_outer}\n")
        io.write(f"#   {'Decay e-folds at zmax':<36} =  {params.decay_efolds:10.3e}\n")

        io.write(f"#\n# Calculation done in {delta_time:.2f} seconds.\n#\n")
        io.write("# Rows with tolerance > 1 did not converge within the resolution range.\n#\n")

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
