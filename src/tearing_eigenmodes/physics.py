from .exceptions import DeltaError
from .params import SimulationParams
import numpy as np
from typing import Tuple, Dict, Any, Optional

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
    alpha: float = 0.1,
    sigma: Optional[Any] = None,
) -> Tuple[float, float]:
    """
    Compute CGL coefficients A and R0, validating physical scaling regimes.
    """
    if sigma is None:
        chi = 0.0
    else:
        sigma_real = getattr(sigma, 'real', sigma)
        if isinstance(sigma_real, np.ndarray):
            sigma_real = np.atleast_1d(sigma_real)[0]
        chi = float(sigma_real)**2 / alpha**2 if sigma_real is not None else 0.0

    A = 1.0 + chi - delta_beta / 2
    R0 = 1.0 + chi + 0.5 * ((gamma_par + gamma_per - 2.0) * beta + gamma_par * delta_beta)

    if A <= 0:
        raise DeltaError(f"Stable or unphysical regime: coefficient A = 1 - Δβ/2 = {A:+.3e} <= 0 (requires Δβ < 2).")

    if sigma is None:
        lambda_sq_ratio = A / R0
        if lambda_sq_ratio <= 0:
            raise DeltaError(f"Δ' purely imaginary or stable: decay coefficient lambda_sq_ratio = {lambda_sq_ratio:+.3e} <= 0.")
    else:
        if np.isclose(R0, 0.0):
            raise DeltaError(f"Infinite decaying factor for (β, Δβ) = ({beta:.3e}, {delta_beta:+.3e}) => stable eigenmode for α = {alpha:.3e}.")
        lambda_inf_sq_ratio = A / R0
        if lambda_inf_sq_ratio < 0.0:
            raise DeltaError(f"Decaying factor purely imaginary for (β, Δβ) = ({beta:.3e}, {delta_beta:+.3e}) => stable eigenmode for α = {alpha:.3e}.")

    return A, R0


def estimate_max(params: SimulationParams) -> float:
    """
    Determine the maximum growth rate and corresponding wavenumber based on physics scaling.
    """
    CGL          = params.CGL
    S            = params.S
    Pr           = params.Pr

    if S is None or S <= 0:
        raise ValueError("Lundquist number S must be positive.")
    if Pr is None or Pr < 0:
        raise ValueError("Prandtl number Pr cannot be negative.")

    αm = 1.3583e+00 * (S / (S + 400))**0.25 * S**-0.25 * ((0.05*Pr**2+0.7*Pr+1)/(12*Pr+1))**0.125

    if CGL:
        β            = params.plasma_beta
        Δβ           = params.plasma_beta_difference
        ɣpar         = params.parallel_index
        ɣper         = params.perpendicular_index
        if β < 0:
            raise ValueError("Plasma beta must be positive.")
        if ɣpar <= 0:
            raise ValueError("Parallel adiabatic index ɣpar must be positive.")
        if ɣper <= 0:
            raise ValueError("Perpendicular adiabatic index ɣper must be positive.")

        A, R0 = calculate_cgl_factors(
            beta=β,
            delta_beta=Δβ,
            gamma_par=ɣpar,
            gamma_per=ɣper,
            alpha=1.0,
            sigma=None,
        )

        αm *= A**0.25 * R0**-0.375

    return αm


def calculate_anisotropy_scale(params: SimulationParams, current_sigma: Optional[Any] = None) -> float:
    """
    Calculate the anisotropy pressure scale width.
    """
    if not params.CGL:
        return 0.0

    sigma_val = current_sigma if current_sigma is not None else params.sigma
    if sigma_val is None:
        return 0.0

    gamma = getattr(sigma_val, 'real', sigma_val)
    if gamma is None:
        return 0.0
    if isinstance(gamma, np.ndarray):
        gamma = np.atleast_1d(gamma)[0]
    gamma = float(gamma)
    if gamma <= 0.0:
        return 0.0

    beta = params.plasma_beta
    delta_beta = params.plasma_beta_difference
    gamma_par = params.parallel_index
    gamma_per = params.perpendicular_index
    alpha = params.alpha
    a = params.a

    if alpha is None or alpha <= 0.0 or a is None or a <= 0.0:
        return 0.0

    beta_bar = 0.5 * ((gamma_par + gamma_per - 2.0) * beta + (gamma_par - 1.0) * delta_beta)
    if beta_bar <= 0.0:
        return 0.0

    # delta_aniso = gamma * tau_A / (k * a * sqrt(beta_bar))
    # In our dimensionless equations, k * a = alpha.
    delta_aniso = gamma / (alpha * np.sqrt(beta_bar))
    return delta_aniso