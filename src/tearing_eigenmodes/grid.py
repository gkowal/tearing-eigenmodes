from .exceptions import DeltaError, ConvergenceError
from .physics import calculate_cgl_factors, calculate_anisotropy_scale, estimate_inner_scale
from .params import SimulationParams
from typing import Dict, Tuple, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def select_C_for_N(
    N: int,
    params: SimulationParams,
    current_sigma: Optional[Any] = None,
    inner_scale_override: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Calculate the inner limit Cinn, outer limit Cout, and mean scaling factor C for a given N.

    Parameters
    ----------
    N : int
        The grid resolution.
    params : SimulationParams
        A SimulationParams object containing physical and numerical parameters.
    current_sigma : complex, optional
        The current growth rate estimate.
    inner_scale_override : float, optional
        Explicit or cached inner scale value.

    Returns
    -------
    tuple
        A tuple (Cinn, Cout, C) representing the inner limit, outer limit, and selected C.
    """
    n_equilibrium = getattr(params, 'n_equilibrium', getattr(params, 'n_inner', 5))
    n_inner_scale = getattr(params, 'n_inner_scale', getattr(params, 'n_resistivity', 5))
    n_anisotropy  = getattr(params, 'n_anisotropy', 5)

    decay_efolds  = params.decay_efolds
    CGL           = params.CGL
    β             = params.plasma_beta
    Δβ            = params.plasma_beta_difference
    ɣpar          = params.parallel_index
    ɣper          = params.perpendicular_index
    a             = params.a
    w             = params.w
    α             = params.alpha
    σ             = current_sigma if current_sigma is not None else params.sigma
    ξ             = params.xi

    if α is None or a is None or w is None:
        raise ValueError("alpha, a, and w must be specified and not None.")

    # Enforce odd point counts for collocation at z = 0
    if n_equilibrium % 2 == 0:
        n_equilibrium += 1
    m_eq = (n_equilibrium - 1) / 2

    if n_inner_scale % 2 == 0:
        n_inner_scale += 1
    m_in = (n_inner_scale - 1) / 2

    if n_anisotropy % 2 == 0:
        n_anisotropy += 1
    m_aniso = (n_anisotropy - 1) / 2

    # --- λ: decaying factor of the outer solution
    if CGL:
        A_cgl, R0_cgl = calculate_cgl_factors(
            beta=β,
            delta_beta=Δβ,
            gamma_par=ɣpar,
            gamma_per=ɣper,
            alpha=α,
            sigma=σ,
        )
        lambda_sq_ratio = A_cgl / R0_cgl
        λ = np.sqrt(lambda_sq_ratio) * α
    else:
        λ = α

    # --- Outer requirement: amplitude reduced by e^{-decay_efolds} at |z|max
    zmin_eq = a + w
    zmax = decay_efolds / λ
    if σ is not None and ξ > 0.0 and σ.real > 0.0:
        lσ = ξ / σ.real
        if lσ <= zmin_eq:
            zmin_eq = np.sqrt(zmin_eq * lσ)
        else:
            zmax = max(zmax, decay_efolds * lσ)

    πh = np.pi / 2
    Np = N + 1

    # 1. Equilibrium current sheet constraint
    Cinn_eq = zmin_eq / np.tan(np.pi * m_eq / Np)
    Cinn = Cinn_eq

    # 2. Inner scale constraint (explicit override -> params.inner_scale / params.delta -> analytic estimator)
    resolved_inner_scale: Optional[float] = None
    if inner_scale_override is not None and inner_scale_override > 0.0:
        resolved_inner_scale = inner_scale_override
    elif params.inner_scale is not None and params.inner_scale > 0.0:
        resolved_inner_scale = params.inner_scale
    elif params.delta is not None and params.delta > 0.0:
        resolved_inner_scale = params.delta
    else:
        try:
            resolved_inner_scale = estimate_inner_scale(params, alpha=α)
        except Exception as ex:
            logger.debug(f"Could not calculate analytic inner scale: {ex}")
            resolved_inner_scale = None

    if resolved_inner_scale is not None and resolved_inner_scale > 0.0:
        Cinn_inner = np.nextafter(
            resolved_inner_scale / np.tan(np.pi * m_in / Np),
            0.0,
        )
        Cinn = min(Cinn, Cinn_inner)

    # 3. Anisotropy pressure scale constraint
    delta_aniso = calculate_anisotropy_scale(params, current_sigma=σ)
    if delta_aniso > 0.0:
        Cinn_aniso = delta_aniso / np.tan(np.pi * m_aniso / Np)
        Cinn = min(Cinn, Cinn_aniso)

    Cout = zmax * np.tan(πh / Np)
    C = Cinn
    return Cinn, Cout, C


def select_NC(params: SimulationParams) -> Tuple[int, float]:
    """
    Determine the Chebyshev–TB grid resolution N and scaling factor C for the linear tearing
    instability eigenproblem under the mapping z = C tan(θ).

    Parameters
    ----------
    params : SimulationParams
        A SimulationParams object containing physical and numerical parameters.

    Returns
    -------
    tuple
        A tuple (N, C) representing the determined resolution and scaling factor.
    """
    Nmin         = params.Nmin
    Nmax         = params.Nmax
    Ninc         = params.Ninc
    decay_efolds = params.decay_efolds
    CGL          = params.CGL
    β            = params.plasma_beta
    Δβ           = params.plasma_beta_difference
    ɣpar         = params.parallel_index
    ɣper         = params.perpendicular_index
    a            = params.a
    w            = params.w
    α            = params.alpha
    σ            = params.sigma
    ξ             = params.xi

    # Input validation checks
    if α is None or α <= 0:
        raise ValueError("Wavenumber alpha must be positive.")
    if decay_efolds <= 0:
        raise ValueError("decay_efolds must be positive.")
    if a is None or a <= 0:
        raise ValueError("Current sheet thickness a must be positive.")
    if w is None or w < 0:
        raise ValueError("Current sheet half-width w must be non-negative.")
    if Nmin <= 0 or Nmax <= 0:
        raise ValueError("Resolution limits Nmin and Nmax must be positive.")
    if Ninc <= 0:
        raise ValueError("Resolution increment Ninc must be positive.")

    # Resolve effective inner scale before the N loop without mutating params
    effective_inner_scale: Optional[float] = None
    if params.inner_scale is not None and params.inner_scale > 0.0:
        effective_inner_scale = params.inner_scale
    elif params.delta is not None and params.delta > 0.0:
        effective_inner_scale = params.delta
    else:
        try:
            effective_inner_scale = estimate_inner_scale(params, alpha=α)
        except Exception as ex:
            logger.debug(f"Analytic inner scale estimation skipped: {ex}")
            effective_inner_scale = None

    δaniso = calculate_anisotropy_scale(params)

    # --- λ: decaying factor of the outer solution
    if CGL:
        A_cgl, R0_cgl = calculate_cgl_factors(
            beta=β,
            delta_beta=Δβ,
            gamma_par=ɣpar,
            gamma_per=ɣper,
            alpha=α,
            sigma=σ,
        )
        lambda_sq_ratio = A_cgl / R0_cgl
        λ = np.sqrt(lambda_sq_ratio) * α
    else:
        λ = α

    zmin_eq = a + w
    zmax = decay_efolds / λ
    lk = ξ / α
    if σ is not None and ξ > 0.0 and σ.real > 0.0:
        lσ = ξ / σ.real
        lk = 2.0 * np.pi * ξ / np.abs(α + σ.imag)
        if lσ <= zmin_eq:
            zmin_eq = np.sqrt(zmin_eq * lσ)
        else:
            zmax = max(zmax, decay_efolds * lσ)
    else:
        lσ = 0.0

    δin_print = effective_inner_scale if effective_inner_scale is not None else 0.0
    logger.debug(
        f"[all scales for α={α:.4e}] a = {a:.4e}, (w+a) = {w+a:.4e}, "
        f"L_inner = {δin_print:.4e}, δaniso = {δaniso:.4e}, "
        f"zmin_eq = {zmin_eq:.4e}, zmax = {zmax:.4e}, Nmin = {Nmin:4d}"
    )

    # Iterate N upward until C_out(N) <= C_in(N); then set C = C_out(N).
    N = Nmin
    while True:
        if N > Nmax:
            raise ConvergenceError(
                f"Insufficient N up to Nmax={Nmax}: cannot satisfy C_out(N) <= C_in(N). "
                f"Try increasing Nmax or relaxing n_equilibrium/n_inner_scale/decay_efolds."
            )
        Cinn, Cout, C = select_C_for_N(N, params, inner_scale_override=effective_inner_scale)
        if Cout <= Cinn:
            break

        if N >= Nmax:
            raise ConvergenceError(
                f"Insufficient N up to Nmax={Nmax}: cannot satisfy C_out(N) <= C_in(N). "
                f"Try increasing Nmax or relaxing n_equilibrium/n_inner_scale/decay_efolds."
            )
        N = min(N + Ninc, Nmax)

    logger.debug(f"[grid scale constrains for α={α:.4e}] Nmin = {N:4d}  C_inner={Cinn:.4e}  C_outer={Cout:.4e} => C = {C:.6e}")

    return N, C
