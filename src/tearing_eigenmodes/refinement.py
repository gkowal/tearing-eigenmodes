from typing import List, Optional, Dict, Any
import numpy as np
import logging
from scipy.interpolate import make_interp_spline
from .io import load_eigenmodes, load_eig_scales

logger = logging.getLogger(__name__)

def refine_inner_scale(vs: np.ndarray, params: Dict[str, Any]) -> List[float]:
    linner = [params['l_inner']] * vs.size

    """Update thickness using cached eigenmodes if available."""
    try:
        v, l = load_eig_scales(params['data_path'])
    except FileNotFoundError:
        return linner

    if v.size < 2:
        return linner

    degree = min(3, v.size - 1)
    spline = make_interp_spline(v, l, k=degree)
    linner = spline(vs)
    if linner.min() <= 0.0:
        spline = make_interp_spline(v, l, k=0)
        linner = spline(vs)

    vmn, vmx = v.min(), v.max()

    linner = linner.tolist()
    for i, x in enumerate(vs):
        if x < vmn or x > vmx:
            linner[i] = params['l_inner']

    return linner


def refine_growth_rate(vs: np.ndarray, params: Dict[str, Any]) -> List[float]:
    sigma = [params['sigma']] * vs.size

    """Update thickness using cached eigenmodes if available."""
    try:
        v, _, σ, _, _, _, _, _, _ = load_eigenmodes(params['data_path'])
    except FileNotFoundError:
        return sigma

    if v.size < 2:
        return sigma

    degree = min(3, v.size - 1)
    spline = make_interp_spline(v, σ, k=degree)
    sigma = spline(vs)
    if sigma.min() <= 0.0:
        spline = make_interp_spline(v, σ, k=0)
        sigma = spline(vs)

    vmn, vmx = v.min(), v.max()

    sigma = sigma.tolist()
    for i, x in enumerate(vs):
        if x < vmn or x > vmx:
            sigma[i] = params['sigma']

    return sigma


def refine_thickness(vs: np.ndarray, params: Dict[str, Any]) -> List[float]:
    δinner = [params['delta']] * vs.size

    """Update thickness using cached eigenmodes if available."""
    try:
        v, _, _, _, δ, _, _, _, _ = load_eigenmodes(params['data_path'])
    except FileNotFoundError:
        return δinner

    if v.size < 2:
        return δinner

    degree = min(3, v.size - 1)
    spline = make_interp_spline(v, δ, k=degree)
    δinner = spline(vs)
    if δinner.min() <= 0.0:
        spline = make_interp_spline(v, δ, k=0)
        δinner = spline(vs)

    vmn, vmx = v.min(), v.max()

    δinner = δinner.tolist()
    for i, x in enumerate(vs):
        if x < vmn or x > vmx:
            δinner[i] = params['delta']

    return δinner


def refine_wavenumber_bracket(vs: np.ndarray, params: Dict[str, Any]) -> List[Optional[List[float]]]:
    """Update wavenumber brackets using cached eigenmodes if available."""
    # Initialize with default bracket from params
    kbracket = [params.get('kbracket')] * vs.size

    try:
        # Assuming load_eigenmodes returns arrays
        v, α, *_ = load_eigenmodes(params['data_path'])
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
        kl, ku = params.get('kbracket') or (None, None)
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

        todo = True
        if np.isclose(kl, ku, atol=atol):
            tol_factor = 5.0 * params.get('ktol', 1e-3)
            kl *= (1.0 - tol_factor)
            ku *= (1.0 + tol_factor)
            todo = False

        kbracket[n] = [float(kl), float(ku)]

        logger.debug(
            f"\t{params.get('dependence', 'v')} = {x:+.3e}: "
            f"α-bracket = [{kl:.4e}, {ku:.4e}]"
        )

    return kbracket
