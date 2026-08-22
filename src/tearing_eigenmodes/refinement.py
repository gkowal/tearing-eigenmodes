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


from .physics import estimate_inner_scale


def refine_inner_scale(vs: np.ndarray, params: SimulationParams) -> List[Optional[float]]:
    """
    Update inner scale using cached eigenmodes and physics-based estimator.

    Parameters
    ----------
    vs : np.ndarray
        Array of sweep values (wavenumbers for dispersion runs or parameter values for maxima runs).
    params : SimulationParams
        Simulation parameters object.

    Returns
    -------
    List[Optional[float]]
        List of resolved inner scales for each value in vs.
    """
    explicit_scale = params.inner_scale if params.inner_scale is not None else params.delta
    if explicit_scale is not None and explicit_scale > 0.0:
        return [explicit_scale] * vs.size

    # Initialize with physics-based estimator for all values
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

    try:
        if params.data_path is None:
            return inner_scales
        v, _, _, _, δ, _, _, _, _ = _cached_load_eigenmodes(params.data_path)
    except Exception:
        return inner_scales

    if v.size < 2:
        return inner_scales

    # Filter out invalid / zero / negative / NaN delta values
    valid_indices = np.where((δ is not None) & (δ > 0.0) & (~np.isnan(δ)))[0]
    if valid_indices.size < 2:
        return inner_scales

    v_valid = v[valid_indices]
    δ_valid = δ[valid_indices]

    # Interpolate in logarithmic scale coordinates
    try:
        log_δ = np.log(δ_valid)
        degree = min(3, v_valid.size - 1)
        spline = make_interp_spline(v_valid, log_δ, k=degree)
        δ_interp = np.exp(spline(vs))

        # Fallback to nearest neighbor (k=0) if any interpolated value is <= 0 or nan
        if np.any(δ_interp <= 0.0) or np.any(np.isnan(δ_interp)):
            spline = make_interp_spline(v_valid, log_δ, k=0)
            δ_interp = np.exp(spline(vs))
    except Exception:
        try:
            log_δ = np.log(δ_valid)
            spline = make_interp_spline(v_valid, log_δ, k=0)
            δ_interp = np.exp(spline(vs))
        except Exception:
            return inner_scales

    vmn, vmx = v_valid.min(), v_valid.max()
    for i, x in enumerate(vs):
        within_bounds = (vmn <= x <= vmx) or np.isclose(x, vmn) or np.isclose(x, vmx)
        if within_bounds:
            val = float(δ_interp[i])
            if val > 0.0 and not np.isnan(val):
                inner_scales[i] = val

    return inner_scales


refine_resistive_scale = refine_inner_scale
