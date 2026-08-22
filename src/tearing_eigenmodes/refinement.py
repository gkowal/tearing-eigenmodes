from typing import List, Optional, Dict, Any, Tuple
import os
import glob
import numpy as np
import logging
from scipy.interpolate import make_interp_spline
from .io import load_eigenmodes

logger = logging.getLogger(__name__)

# Cache dictionary to map directory path & pattern to last seen directory state and loaded data.
# The cache key is (path, pattern).
# The cache value is (directory_state, data).
# directory_state is a tuple of (filename, mtime, size) tuples.
_load_eigenmodes_cache: Dict[Tuple[str, str], Tuple[Tuple[Tuple[str, float, int], ...], Tuple]] = {}


def _get_directory_state(path: str, pattern: str) -> Tuple[Tuple[str, float, int], ...]:
    """
    Return a tuple of (filename, mtime, size) for all matching files.
    This uniquely identifies the state of the directory.
    """
    search_path = os.path.join(path, pattern)
    files = sorted(glob.glob(search_path))
    state = []
    for f in files:
        try:
            st = os.stat(f)
            state.append((f, st.st_mtime, st.st_size))
        except FileNotFoundError:
            continue
    return tuple(state)


def _cached_load_eigenmodes(
    path: str, pattern: str = "*.npz"
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Wrapper around load_eigenmodes that caches results based on the files' sizes and modification times.
    """
    key = (path, pattern)
    try:
        current_state = _get_directory_state(path, pattern)
    except Exception:
        # If directory/files access fails, bypass cache
        return load_eigenmodes(path, pattern)

    if key in _load_eigenmodes_cache:
        cached_state, data = _load_eigenmodes_cache[key]
        if cached_state == current_state:
            logger.debug(f"Refinement I/O Cache HIT for {path}")
            return data

    logger.debug(f"Refinement I/O Cache MISS for {path}")
    data = load_eigenmodes(path, pattern)
    _load_eigenmodes_cache[key] = (current_state, data)
    return data


from .params import SimulationParams

def refine_eigenvalues(vs: np.ndarray, params: SimulationParams) -> List[Any]:
    sigma = [params.sigma] * vs.size

    """Update eigenvalues using cached eigenmodes if available."""
    try:
        if params.data_path is None:
            return sigma
        v, _, σ, _, _, _, _, _, _ = _cached_load_eigenmodes(params.data_path)
    except FileNotFoundError:
        return sigma

    if v.size < 2:
        return sigma

    degree = min(3, v.size - 1)
    spline = make_interp_spline(v, σ, k=degree)
    sigma = spline(vs)
    if np.real(sigma).min() <= 0.0:
        spline = make_interp_spline(v, σ, k=0)
        sigma = spline(vs)

    vmn, vmx = v.min(), v.max()

    sigma = sigma.tolist()
    for i, x in enumerate(vs):
        if x < vmn or x > vmx:
            sigma[i] = params.sigma

    return sigma


def refine_wavenumber_bracket(vs: np.ndarray, params: SimulationParams) -> List[Optional[List[float]]]:
    """Update wavenumber brackets using cached eigenmodes if available."""
    # Initialize with default bracket from params
    kbracket = [params.kbracket] * vs.size

    try:
        if params.data_path is None:
            return kbracket
        # Assuming load_eigenmodes returns arrays
        v, α, *_ = _cached_load_eigenmodes(params.data_path)
    except (FileNotFoundError, KeyError, TypeError):
        return kbracket

    if v.size < 2:
        return kbracket

    logger.info("Improved wavenumber bounds:")

    vmn, vmx = v.min(), v.max()
    atol = 1e-12 * max(abs(vmn), abs(vmx))

    def _alpha_at_last_leq(x):
        idx = np.searchsorted(v, x + atol, side="right") - 1
        return α[idx] if idx >= 0 else None

    def _alpha_at_first_geq(x):
        idx = np.searchsorted(v, x - atol, side="left")
        return α[idx] if idx < v.size else None

    for n, x in enumerate(vs):
        kbr = params.kbracket
        kl, ku = (kbr[0], kbr[1]) if (kbr is not None and len(kbr) >= 2) else (None, None)
        within_bounds = (vmn <= x <= vmx) or np.isclose(x, vmn, atol=atol) or np.isclose(x, vmx, atol=atol)

        if within_bounds:
            αl = _alpha_at_last_leq(x)
            αu = _alpha_at_first_geq(x)

            if αl is not None and αu is not None:
                inner_l = min(αl, αu)
                inner_u = max(αl, αu)
                kl = inner_l if kl is None else max(kl, inner_l)
                ku = inner_u if ku is None else min(ku, inner_u)

        if kl is None or ku is None:
            kbracket[n] = None
            continue

        if np.isclose(kl, ku, atol=atol):
            ktol = params.ktol if params.ktol is not None else 1e-3
            tol_factor = 5.0 * ktol
            kl *= (1.0 - tol_factor)
            ku *= (1.0 + tol_factor)

        kbracket[n] = [float(kl), float(ku)]

        logger.debug(
            f"\t{params.dependence or 'v'} = {x:+.3e}: "
            f"α-bracket = [{kl:.4e}, {ku:.4e}]"
        )

    return kbracket


from .analysis import CLASSICAL_GRID_SCALE_KEYS, CGL_GRID_SCALE_KEYS
from .io import check_state
from .physics import estimate_inner_scale


def _load_cached_state_records(path: str, params: SimulationParams, pattern: str = "*.npz") -> List[Dict[str, Any]]:
    """
    Load all valid cached state files from the given directory.
    """
    search_path = os.path.join(path, pattern)
    files = sorted(glob.glob(search_path))
    records: List[Dict[str, Any]] = []
    is_dependence = params.dependence is not None

    for f in files:
        status, data = check_state(f)
        if status and data is not None:
            if is_dependence:
                if 'scan_parameter' in data:
                    v = float(data['scan_parameter_value'])
                elif 'dependence' in data:
                    v = float(data['value'])
                else:
                    v = float(data.get('wavenumber', 0.0))
            else:
                v = float(data['wavenumber'])

            records.append({
                "v": v,
                "wavenumber": float(data.get("wavenumber", v)),
                "mode_scales": data.get("mode_scales"),
                "minimum_physical_scale": data.get("minimum_physical_scale"),
                "resistive_layer_thickness": data.get("resistive_layer_thickness"),
            })
    records.sort(key=lambda r: r["v"])
    return records


def refine_inner_scale(vs: np.ndarray, params: SimulationParams) -> List[Optional[float]]:
    """
    Update inner scale using cached per-term eigenmode scales and physics-based estimators.

    Parameters
    ----------
    vs : np.ndarray
        Array of sweep values (wavenumbers for dispersion runs or parameter values for maxima runs).
    params : SimulationParams
        Simulation parameters object.

    Returns
    -------
    List[Optional[float]]
        List of resolved inner grid scales (with safety factor applied) for each value in vs.
    """
    safety = float(getattr(params, "inner_resolution_safety", 1.0) or 1.0)
    explicit_scale = params.inner_scale if params.inner_scale is not None else params.delta
    if explicit_scale is not None and explicit_scale > 0.0:
        return [explicit_scale / safety] * vs.size

    # 1. Default baseline: physics-based estimator for all values (estimate_inner_scale already divides by safety)
    inner_scales: List[Optional[float]] = [None] * vs.size
    for i, x in enumerate(vs):
        try:
            if params.dependence is None:
                # Dispersion run: vs are k values, alpha = k * a
                a = params.a if params.a is not None else 1.0
                alpha_val = float(x * a)
                inner_scales[i] = estimate_inner_scale(params, alpha=alpha_val)
            else:
                # Maxima sweep run
                import copy
                p_temp = copy.copy(params)
                dep_map = {
                    'a': 'a', 'w': 'w', 'S': 'S', 'Pr': 'Pr',
                    'ξ': 'xi', 'ϵ': 'Hall', 'β': 'plasma_beta',
                    'Δβ': 'plasma_beta_difference'
                }
                field = dep_map.get(params.dependence)
                if field:
                    setattr(p_temp, field, float(x))
                inner_scales[i] = estimate_inner_scale(p_temp)
        except Exception as ex:
            logger.debug(f"Analytic inner scale fallback for v={x:+.3e}: {ex}")
            inner_scales[i] = None

    if params.data_path is None or not os.path.exists(params.data_path):
        return inner_scales

    try:
        records = _load_cached_state_records(params.data_path, params)
    except Exception as ex:
        logger.debug(f"Failed to load cached state records for scale refinement: {ex}")
        return inner_scales

    if not records:
        return inner_scales

    candidate_keys = CGL_GRID_SCALE_KEYS if params.CGL else CLASSICAL_GRID_SCALE_KEYS
    is_positive_dispersion = (params.dependence is None) and np.all(vs > 0)

    # 2. Try per-term physical-scale interpolation
    interpolated_terms: Dict[str, List[Optional[float]]] = {k: [None] * vs.size for k in candidate_keys}
    has_any_per_term_data = False

    for k in candidate_keys:
        pts = [
            (r["v"], float(r["mode_scales"][k]))
            for r in records
            if r.get("mode_scales") is not None
            and k in r["mode_scales"]
            and r["mode_scales"][k] is not None
            and not np.isnan(r["mode_scales"][k])
            and not np.isinf(r["mode_scales"][k])
            and float(r["mode_scales"][k]) > 0.0
        ]
        if not pts:
            continue

        has_any_per_term_data = True
        v_pts = np.array([p[0] for p in pts])
        scale_pts = np.array([p[1] for p in pts])
        vmn, vmx = v_pts.min(), v_pts.max()

        if v_pts.size == 1:
            for i, x in enumerate(vs):
                if np.isclose(x, vmn) or (vmn <= x <= vmx):
                    interpolated_terms[k][i] = float(scale_pts[0])
        else:
            if is_positive_dispersion and np.all(v_pts > 0):
                log_v = np.log(v_pts)
                log_s = np.log(scale_pts)
                for i, x in enumerate(vs):
                    if (vmn <= x <= vmx) or np.isclose(x, vmn) or np.isclose(x, vmx):
                        interpolated_terms[k][i] = float(np.exp(np.interp(np.log(x), log_v, log_s)))
            else:
                log_s = np.log(scale_pts)
                for i, x in enumerate(vs):
                    if (vmn <= x <= vmx) or np.isclose(x, vmn) or np.isclose(x, vmx):
                        interpolated_terms[k][i] = float(np.exp(np.interp(x, v_pts, log_s)))

    if has_any_per_term_data:
        for i, x in enumerate(vs):
            active_vals: List[Tuple[str, float]] = []
            for k in candidate_keys:
                term_val = interpolated_terms[k][i]
                if term_val is not None:
                    active_vals.append((k, term_val))
            if active_vals:
                min_key, min_scale = min(active_vals, key=lambda item: item[1])
                inner_scales[i] = min_scale / safety
                logger.debug(
                    f"Refined grid scale at v={x:+.3e}: {inner_scales[i]:.4e} "
                    f"(limiting: {min_key}, safety: {safety})"
                )
        return inner_scales

    # 3. Fallback to scalar minimum_physical_scale or legacy resistive_layer_thickness
    scalar_pts = []
    for r in records:
        val = r.get("minimum_physical_scale")
        if val is None or np.isnan(val) or np.isinf(val) or val <= 0.0:
            val = r.get("resistive_layer_thickness")
        if val is not None and not np.isnan(val) and not np.isinf(val) and float(val) > 0.0:
            scalar_pts.append((r["v"], float(val)))

    if not scalar_pts:
        return inner_scales

    v_pts = np.array([p[0] for p in scalar_pts])
    scale_pts = np.array([p[1] for p in scalar_pts])
    vmn, vmx = v_pts.min(), v_pts.max()

    if v_pts.size == 1:
        for i, x in enumerate(vs):
            if np.isclose(x, vmn) or (vmn <= x <= vmx):
                inner_scales[i] = float(scale_pts[0]) / safety
    else:
        if is_positive_dispersion and np.all(v_pts > 0):
            log_v = np.log(v_pts)
            log_s = np.log(scale_pts)
            for i, x in enumerate(vs):
                if (vmn <= x <= vmx) or np.isclose(x, vmn) or np.isclose(x, vmx):
                    interp_val = float(np.exp(np.interp(np.log(x), log_v, log_s)))
                    inner_scales[i] = interp_val / safety
        else:
            log_s = np.log(scale_pts)
            for i, x in enumerate(vs):
                if (vmn <= x <= vmx) or np.isclose(x, vmn) or np.isclose(x, vmx):
                    interp_val = float(np.exp(np.interp(x, v_pts, log_s)))
                    inner_scales[i] = interp_val / safety

    return inner_scales


refine_resistive_scale = refine_inner_scale
