from typing import Any, Optional, Dict, List
from .params import SimulationParams
import logging

logger = logging.getLogger(__name__)


def print_info(params: SimulationParams) -> None:
    """
    Print the physical and numerical parameters used for the calculation.
    """

    def log_head(label, key, fmt="10.3e"):
        val = getattr(params, key)
        if val is not None:
            logger.info(f"  {label:<34} =  {val:{fmt}}")

    logger.info("Plasma parameters:")
    eq_type = "Gyrotropic" if params.CGL else "Classical"
    logger.info(f"  {'Equations':<34} =  {eq_type} MHD")
    if params.CGL:
        logger.info(f"  {'Equation of State':<34} =  {params.eos}")

    log_head("Lundquist number (S)", 'S')
    log_head("Prandtl number (Pr)", 'Pr')
    if params.CGL:
        log_head("Plasma-β (β)", 'plasma_beta')
        log_head("Plasma-β difference (Δβ)", 'plasma_beta_difference')
    if not params.CGL:
        log_head("Magnetic transverse field (ξ)", 'xi')
    log_head("Hall current term strength (ϵ)", 'Hall')

    if params.CGL:
        logger.info(f"  {'Parallel adiabatic index':<34} =  {params.parallel_index}")
        logger.info(f"  {'Perpendicular adiabatic index':<34} =  {params.perpendicular_index}")

    logger.info("Equilibrium parameters:")
    log_head("Current sheet thickness (a)", 'a')
    if not params.CGL:
        log_head("Current sheet half-width (w)", 'w')
        log_head("Shear parameter (ζ)", 'zeta')

    if not params.CGL:
        logger.info(f"  {'Velocity shear     ':<34} =  {not params.noshear}")

    vmin, vmax, vinc = params.vmin, params.vmax, params.vinc
    is_log = "log10" if params.logarithmic else ""

    if params.dependence:
        logger.info("Dependence parameters:")
        logger.info(f"  {'Calculating dependence on':<34} =  {params.dependence}")
        logger.info(f"  {'Parameter value range (R)':<34} =  {is_log}[{vmin}, {vmax}] with increment {vinc}")

        if params.kbracket:
            logger.info(f"  {'Wavenumber bracket':<34} =  {params.kbracket}")

    else:
        logger.info("Wavenumbers:")
        kmin, kmax, kinc = params.kmin, params.kmax, params.kinc
        logger.info(f"  {'Wavenumber range (k)':<34} =  {is_log}[{kmin}, {kmax}] with increment {kinc}")

    logger.info("Geometry/convergence parameters:")
    Nmin, Nmax, Ninc = params.Nmin, params.Nmax, params.Ninc
    logger.info(f"  {'Resolution range (N)':<34} =  [{Nmin}, {Nmax}] with increment {Ninc}")

    log_head("Growth rate absolute tolerance", 'atol')
    log_head("Growth rate relative tolerance", 'rtol')
    log_head("Guess tolerance", 'gtol')
    logger.info(f"  {'Real part range':<34} =  [{params.sigma_real_lower}, {params.sigma_real_upper}]")
    logger.info(f"  {'Imaginary part range':<34} =  [{params.sigma_imag_lower}, {params.sigma_imag_upper}]")

    logger.info(f"  {'Equilibrium collocation points':<34} =  {params.n_equilibrium}")
    logger.info(f"  {'Inner scale collocation points':<34} =  {params.n_inner_scale}")
    if params.CGL:
        logger.info(f"  {'Anisotropy scale points':<34} =  {params.n_anisotropy}")
    logger.info(f"  {'Inner resolution safety factor':<34} =  {params.inner_resolution_safety}")
    logger.info(f"  {'Amplitude fraction at zmax':<34} =  {params.f_outer}")
    logger.info(f"  {'Dynamic C grid':<34} =  {'on' if params.dynamic_C else 'off'}")

    log_head("Inner scale (δin)", 'inner_scale')
    log_head("Inner-layer thickness tolerance", 'dtol')

    logger.info("Miscellaneous/Runtime:")

    logger.info(f"  {'Eigenmode order':<34} =  {params.orderby}")

    mode_val = params.mode
    logger.info(f"  {'Converge the mode':<34} =  {mode_val if mode_val is not None else 'N/A'}")

    logger.info(f"  {'Return all modes':<34} =  {params.allmodes}")
    logger.info(f"  {'Force recalculation':<34} =  {params.force}")

    if params.suffix:
        logger.info(f"  {'Suffix':<34} =  {params.suffix}")


def format_mode_scale_summary(scales: Any) -> Optional[str]:
    """
    Format a deterministic mode scales summary string from a mode_scales mapping.
    Safely handles non-mapping inputs, mixed key types, canonical key collisions, NaN, inf, <= 0, strings, and malformed entries.
    Returns 'mode scales: key=value, ...' if valid entries exist, else None.
    """
    from collections.abc import Mapping
    from collections import defaultdict
    import numpy as np

    if not isinstance(scales, Mapping):
        return None

    try:
        items = list(scales.items())
    except Exception:
        return None

    grouped_entries: Dict[str, List[str]] = defaultdict(list)
    for k, v in items:
        try:
            k_str = str(k)
        except Exception:
            continue

        if v is None:
            continue
        if isinstance(v, (bool, np.bool_)):
            continue
        if isinstance(v, (str, bytes)):
            continue
        try:
            arr = np.asanyarray(v)
            if arr.ndim != 0 and arr.size != 1:
                continue
            elem = arr.item() if arr.ndim == 0 else arr.flat[0]
            if isinstance(elem, (bool, np.bool_)):
                continue
            if isinstance(elem, (str, bytes)):
                continue
            if isinstance(elem, complex) or np.iscomplexobj(elem):
                if elem.imag != 0.0:
                    continue
                elem = elem.real
            f_val = float(elem)
            if np.isnan(f_val) or f_val <= 0.0:
                continue
            if np.isinf(f_val):
                if f_val > 0.0:
                    rendered = f"{k_str}=inf"
                    if rendered not in grouped_entries[k_str]:
                        grouped_entries[k_str].append(rendered)
            else:
                rendered = f"{k_str}={f_val:.4e}"
                if rendered not in grouped_entries[k_str]:
                    grouped_entries[k_str].append(rendered)
        except Exception:
            continue

    # Resolve canonical duplicate collisions:
    # If a canonical key has exactly one distinct rendered value, emit it.
    # If multiple distinct rendered values exist (conflict), omit that ambiguous key.
    scale_strings = []
    for k_str in sorted(grouped_entries.keys()):
        rendered_list = grouped_entries[k_str]
        if len(rendered_list) == 1:
            scale_strings.append(rendered_list[0])

    if not scale_strings:
        return None

    return f"mode scales: {', '.join(scale_strings)}"
