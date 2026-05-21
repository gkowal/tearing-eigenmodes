from .exceptions import DeltaError
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


def estimate_max(params: Dict[str, Any]) -> Tuple[float, float]:
    """
    Determine the maximum growth rate and corresponding wavenumber based on physics scaling.
    """
    CGL          = params.get('CGL'                   , False  )
    S            = params.get('S'                     ,    1.0e4 )
    Pr           = params.get('Pr'                    ,    0.0   )
    β            = params.get('plasma_beta'           ,    0.0   )
    Δβ           = params.get('plasma_beta_difference',    0.0   )
    ɣpar         = params.get('parallel_index'        ,    3.0   )
    ɣper         = params.get('perpendicular_index'   ,    2.0   )

    if S <= 0:
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
        numerator = 2 + (ɣpar + ɣper - 2) * β + ɣpar * Δβ
        denominator = 2 - Δβ
        if denominator <= 0:
            raise DeltaError(f"Stable or unphysical regime: Δβ = {Δβ:+.3e} >= 2 causes division by zero or negative denominator.")
        μ_inside = numerator / denominator
        if μ_inside <= 0:
            raise DeltaError(f"Δ' purely imaginary or stable: decay coefficient μ_inside = {μ_inside:+.3e} <= 0.")
        μ = np.sqrt(μ_inside)
    else:
        μ = 1.0

    αm = 1.3583e+00 * (S / (S + 400))**(1/4) * S**(-1/4) * C**(-1/8) * μ**(-3/4) * ((0.05*Pr**2+0.7*Pr+1)/(12*Pr+1))**(1/8)
    Xm = αm * μ
    Δm = 2 * (1 / Xm - Xm)

    return αm, Δm
