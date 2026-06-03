from .exceptions import DeltaError, ConvergenceError
from .physics import calculate_cgl_factors, calculate_anisotropy_scale
from .params import SimulationParams
from typing import Dict, Tuple, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def select_C_for_N(N: int, params: SimulationParams, current_sigma: Optional[Any] = None) -> Tuple[float, float, float]:
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

    Returns
    -------
    tuple
        A tuple (Cinn, Cout, C) representing the inner limit, outer limit, and selected C.
    """
    n_inner      = params.n_inner
    n_resistivity = params.n_resistivity
    n_anisotropy = getattr(params, 'n_anisotropy', 5)
    delta        = params.delta
    decay_efolds = params.decay_efolds
    CGL          = params.CGL
    β            = params.plasma_beta
    Δβ           = params.plasma_beta_difference
    ɣpar         = params.parallel_index
    ɣper         = params.perpendicular_index
    a            = params.a
    w            = params.w
    α            = params.alpha
    σ            = current_sigma if current_sigma is not None else params.sigma
    ξ            = params.xi
    Cmean        = params.Cmean

    if α is None or a is None or w is None:
        raise ValueError("alpha, a, and w must be specified and not None.")

    # Enforce odd n_inner (so m is integer and z=0 is a collocation point)
    if n_inner % 2 == 0:
        n_inner += 1
    m = (n_inner - 1) / 2

    # Enforce odd n_resistivity
    if n_resistivity % 2 == 0:
        n_resistivity += 1
    m_res = (n_resistivity - 1) / 2

    # Enforce odd n_anisotropy
    if n_anisotropy % 2 == 0:
        n_anisotropy += 1
    m_aniso = (n_anisotropy - 1) / 2

    # --- λ: decaying factor of the outer solution;
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
    zmin = a + w
    zmax = decay_efolds / λ
    if σ is not None and ξ > 0.0 and σ.real > 0.0:
        lσ = ξ / σ.real
        if lσ <= zmin:
            zmin = np.sqrt(zmin * lσ)
        else:
            zmax = max(zmax, decay_efolds * lσ)

    πh   = np.pi / 2
    πm   = np.pi * m
    Np   = N + 1

    Cinn = zmin / np.tan(πm / Np)
    if delta is not None and delta > 0.0:
        Cinn_res = delta / np.tan(np.pi * m_res / Np)
        Cinn = min(Cinn, Cinn_res)

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
    ξ            = params.xi
    Cmean        = params.Cmean
    delta        = params.delta
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

    # --- λ: decaying factor of the outer solution;
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
    zmin = a + w
    zmax = decay_efolds / λ
    lk = ξ / α
    if σ is not None and ξ > 0.0 and σ.real > 0.0:
        lσ = ξ / σ.real
        lk = 2.0 * np.pi * ξ / np.abs(α + σ.imag)

        if lσ <= zmin:
            zmin = np.sqrt(zmin * lσ)
        else:
            zmax = max(zmax, decay_efolds * lσ)
    else:
        lσ   = 0.0

    δres = delta if delta is not None else 0.0
    δaniso = calculate_anisotropy_scale(params)

    if CGL:
        scales = [a]
        if δres > 0.0:
            scales.append(δres)
        if δaniso > 0.0:
            scales.append(δaniso)
        zmin_print = min(scales)
        logger.debug(
            f"[all scales for α={α:.4e}] a = {a:.4e}, "
            f"δres = {δres:.4e}, δaniso = {δaniso:.4e}, "
            f"zmin = {zmin_print:.4e}, zmax = {zmax:.4e}, Nmin = {Nmin:4d}"
        )
    else:
        logger.debug(
            f"[all scales for α={α:.4e}] (w+a) = {w+a:.4e}, "
            f"lσ = {lσ:.4e}, lk = {lk:.4e}, δres = {δres:.4e}, "
            f"zmin = {zmin:.4e}, zmax = {zmax:.4e}, Nmin = {Nmin:4d}"
        )

    # Iterate N upward until C_out(N) <= C_in(N); then set C = C_out(N).
    Ntop = Nmax - 3 * Ninc

    N  = Nmin
    while True:
        if N > Ntop:
            raise ConvergenceError(
                f"Insufficient N up to Nmax={Nmax}: cannot satisfy C_out(N) <= C_in(N). "
                f"Try increasing Nmax or relaxing n_inner/decay_efolds."
            )
        Cinn, Cout, C = select_C_for_N(N, params)
        if Cout <= Cinn:
            break

        N += Ninc

    logger.debug(f"[grid scale constrains for α={α:.4e}] Nmin = {N:4d}  C_inner={Cinn:.4e}  C_outer={Cout:.4e} => C = {C:.6e} using {Cmean} mean")

    return N, C
