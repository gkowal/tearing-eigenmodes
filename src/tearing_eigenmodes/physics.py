from .exceptions import DeltaError
import numpy as np
from typing import Tuple

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


def estimate_max(params):
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

    if CGL and (Δβ <= - (2 + (ɣpar + ɣper - 2) * β) / ɣpar or Δβ >= 2):
        raise DeltaError(f"Δ' purely imaginary for Δβ = {Δβ:+.3e} => stable eigenmode for any α.")

    C = 1 - Δβ/2
    μ = np.sqrt((2 + (ɣpar + ɣper - 2) * β + ɣpar * Δβ) / (2 - Δβ)) if CGL else 1

    αm = 1.3583e+00 * (S / (S + 400))**(1/4) * S**(-1/4) * C**(-1/8) * μ**(-3/4) * ((0.05*Pr**2+0.7*Pr+1)/(12*Pr+1))**(1/8)
    Xm = αm * μ
    Δm = 2 * (1 / Xm - Xm)

    return αm, Δm
