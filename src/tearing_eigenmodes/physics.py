from .exceptions import DeltaError
from .params import SimulationParams
import numpy as np
from typing import Tuple, Dict, Any

def eos_indices(eos: str) -> Tuple[float, float]:
    """
    Return the two numerical indices that correspond to a given equation‑of‑state (EOS).
    """
    if eos == 'adiabatic':
        return 3.0, 2.0
    elif eos == 'polytropic':
        return 0.5, 2.0
    elif eos == 'isothermal':
        return 1.0, 1.0

    raise ValueError(f"Unsupported equation of state: {eos!r}")


def calculate_cgl_factors(
    beta: float,
    delta_beta: float,
    gamma_par: float,
    gamma_per: float,
    R: float = 0.0,
    alpha: float = 0.1,
    context: str = 'grid',
) -> Tuple[float, float]:
    """
    Compute CGL coefficients C and D, validating physical scaling regimes.
    """
    C = 1.0 + R - delta_beta / 2
    D = 1.0 + R + 0.5 * ((gamma_par + gamma_per - 2.0) * beta + gamma_par * delta_beta)

    if context == 'physics':
        denominator = 2 * C
        numerator = 2 * D
        if denominator <= 0:
            raise DeltaError(f"Stable or unphysical regime: Δβ = {delta_beta:+.3e} >= 2 causes division by zero or negative denominator.")
        μ_inside = numerator / denominator
        if μ_inside <= 0:
            raise DeltaError(f"Δ' purely imaginary or stable: decay coefficient μ_inside = {μ_inside:+.3e} <= 0.")
    else:
        if np.isclose(D, 0.0):
            raise DeltaError(f"Infinite decaying factor for (β, Δβ) = ({beta:.3e}, {delta_beta:+.3e}) => stable eigenmode for α = {alpha:.3e}.")
        μ = C / D
        if μ < 0.0:
            raise DeltaError(f"Decaying factor purely imaginary for (β, Δβ) = ({beta:.3e}, {delta_beta:+.3e}) => stable eigenmode for α = {alpha:.3e}.")

    return C, D


def estimate_max(params: SimulationParams) -> Tuple[float, float]:
    """
    Determine the maximum growth rate and corresponding wavenumber based on physics scaling.
    """
    CGL          = params.CGL
    S            = params.S
    Pr           = params.Pr
    β            = params.plasma_beta
    Δβ           = params.plasma_beta_difference
    ɣpar         = params.parallel_index
    ɣper         = params.perpendicular_index

    if S is None or S <= 0:
        raise ValueError("Lundquist number S must be positive.")
    if Pr < 0:
        raise ValueError("Prandtl number Pr cannot be negative.")
    if ɣpar <= 0:
        raise ValueError("Parallel adiabatic index ɣpar must be positive.")
    if ɣper <= 0:
        raise ValueError("Perpendicular adiabatic index ɣper must be positive.")

    C = 1 - Δβ/2
    if C <= 0:
        raise DeltaError(f"Stable or unphysical regime: coefficient C = 1 - Δβ/2 = {C:+.3e} <= 0 (requires Δβ < 2).")

    if CGL:
        C_cgl, D_cgl = calculate_cgl_factors(
            beta=β,
            delta_beta=Δβ,
            gamma_par=ɣpar,
            gamma_per=ɣper,
            R=0.0,
            context='physics',
        )
        μ_inside = D_cgl / C_cgl
        μ = np.sqrt(μ_inside)
    else:
        μ = 1.0

    αm = 1.3583e+00 * (S / (S + 400))**(1/4) * S**(-1/4) * C**(-1/8) * μ**(-3/4) * ((0.05*Pr**2+0.7*Pr+1)/(12*Pr+1))**(1/8)
    Xm = αm * μ
    Δm = 2 * (1 / Xm - Xm)

    return αm, Δm
