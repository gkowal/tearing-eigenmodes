import logging
from .exceptions import DeltaError
from .params import SimulationParams
import numpy as np
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

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


def calculate_inner_factors(
    beta: float = 0.0,
    delta_beta: float = 0.0,
    gamma_par: float = 3.0,
    gamma_per: float = 2.0,
    CGL: bool = False,
) -> Tuple[float, float, float]:
    """
    Compute quasistatic Classical and Gyrotropic inner-layer coefficients A, R0, and q.

    Parameters
    ----------
    beta : float
        Perpendicular plasma beta.
    delta_beta : float
        Difference between parallel and perpendicular plasma beta (Δβ).
    gamma_par : float
        Parallel adiabatic index.
    gamma_per : float
        Perpendicular adiabatic index.
    CGL : bool
        Whether Gyrotropic (CGL) MHD is enabled.

    Returns
    -------
    Tuple[float, float, float]
        (A, R0, q) where q = sqrt(A / R0).
    """
    if not CGL:
        return 1.0, 1.0, 1.0

    A = 1.0 - delta_beta / 2.0
    R0 = 1.0 + 0.5 * ((gamma_par + gamma_per - 2.0) * beta + gamma_par * delta_beta)

    if A <= 0.0 or R0 <= 0.0:
        raise DeltaError(
            f"Gyrotropic inner factors require A > 0 and R0 > 0 for analytic inner-scale estimation "
            f"(got A = {A:+.3e}, R0 = {R0:+.3e})."
        )

    q = float(np.sqrt(A / R0))
    return A, R0, q


def model_delta_prime(alpha: float, q: float = 1.0) -> float:
    """
    Compute the approximate outer tearing stability index Δ'_model.

    Parameters
    ----------
    alpha : float
        Normalized wavenumber k * a.
    q : float
        Quasistatic anisotropy ratio sqrt(A / R0) (q = 1 for Classical MHD).

    Returns
    -------
    float
        Δ'_model = 2 * (q / alpha - alpha / q).
    """
    if alpha <= 0.0:
        raise ValueError("Wavenumber alpha must be strictly positive.")
    if q <= 0.0:
        raise ValueError("Parameter q must be strictly positive.")
    return 2.0 * (q / alpha - alpha / q)


def legacy_fS(S: float) -> float:
    """Legacy empirical Lundquist-number prefactor for Coppi branch."""
    return 0.038288 / (0.047443 + S**(-0.46355)) + 0.079649


def legacy_gS(S: float) -> float:
    """Legacy empirical Lundquist-number prefactor for FKR branch."""
    return 0.91451 - 2.0654 / (S**0.37651 + 0.88448)


def legacy_fPr(Pr: float) -> float:
    """Legacy empirical Prandtl-number prefactor for Coppi branch."""
    return (1.0777 + (0.71554 * Pr) * (5.765 + Pr))**0.079176


def legacy_gPr(Pr: float) -> float:
    """Legacy empirical Prandtl-number prefactor for FKR branch."""
    return 2.4379 * (Pr + 0.0045991)**0.16186


def estimate_growth_rate(params: SimulationParams, alpha: Optional[float] = None) -> float:
    """
    Estimate the tearing instability growth rate gamma * tau_A using smooth asymptotic formulas.

    Notes
    -----
    The theoretical derivation assumes inviscid (Pr = 0) tearing with w = 0, xi = 0.
    """
    a = params.a if params.a is not None else 1.0
    alpha_val = alpha if alpha is not None else params.alpha
    if alpha_val is None or alpha_val <= 0.0:
        raise ValueError("Wavenumber alpha must be positive.")
    S = params.S if params.S is not None else 1e4
    if S <= 0.0:
        raise ValueError("Lundquist number S must be positive.")

    A, R0, q = calculate_inner_factors(
        beta=params.plasma_beta,
        delta_beta=params.plasma_beta_difference,
        gamma_par=params.parallel_index,
        gamma_per=params.perpendicular_index,
        CGL=params.CGL,
    )

    delta_prime = model_delta_prime(alpha_val, q=q)
    gamma_hat_Coppi = (A**(1.0 / 3.0)) * (alpha_val**(2.0 / 3.0)) * (S**(-1.0 / 3.0))

    if delta_prime <= 0.0:
        return float(gamma_hat_Coppi)

    gamma_hat_FKR = (A**(1.0 / 5.0)) * (alpha_val**(2.0 / 5.0)) * (delta_prime**(4.0 / 5.0)) * (S**(-3.0 / 5.0))
    p = 8
    gamma_hat = (gamma_hat_Coppi * gamma_hat_FKR) / ((gamma_hat_Coppi**p + gamma_hat_FKR**p)**(1.0 / p))
    return float(gamma_hat)


def estimate_inner_scale(params: SimulationParams, alpha: Optional[float] = None) -> float:
    """
    Calculate the physics-based initial grid inner scale for the tearing layer.

    Parameters
    ----------
    params : SimulationParams
        Simulation parameters object.
    alpha : float, optional
        Normalized wavenumber k * a. If None, uses params.alpha.

    Returns
    -------
    float
        Dimensional inner scale (inner_scale = a * delta_hat_grid).

    Notes
    -----
    - The underlying FKR and Coppi asymptotic derivations are inviscid (Pr = 0) and
      derived for standard Harris sheet equilibria (w = 0, xi = 0).
    - Factors for Pr > 0 and finite S are legacy empirical order-unity prefactors.
    - An inner_resolution_safety factor >= 1 scales the target grid scale finer than the model.
    """
    alpha_val = alpha if alpha is not None else params.alpha
    if alpha_val is None or alpha_val <= 0.0:
        raise ValueError("Wavenumber alpha must be positive for inner scale estimation.")

    a = params.a if params.a is not None else 1.0
    if a <= 0.0:
        raise ValueError("Current sheet thickness a must be positive.")

    S = params.S if params.S is not None else 1e4
    if S <= 0.0:
        raise ValueError("Lundquist number S must be positive.")

    Pr = params.Pr if params.Pr is not None else 0.0
    if Pr < 0.0:
        raise ValueError("Prandtl number Pr cannot be negative.")

    A, R0, q = calculate_inner_factors(
        beta=params.plasma_beta,
        delta_beta=params.plasma_beta_difference,
        gamma_par=params.parallel_index,
        gamma_per=params.perpendicular_index,
        CGL=params.CGL,
    )

    Delta_prime_model = model_delta_prime(alpha_val, q=q)

    fS = legacy_fS(S)
    gS = legacy_gS(S)
    fPr = legacy_fPr(Pr)
    gPr = legacy_gPr(Pr)

    delta_hat_Coppi = fS * fPr * (A**(-1.0 / 6.0)) * ((alpha_val * S)**(-1.0 / 3.0))

    if Delta_prime_model <= 0.0:
        logger.debug(
            f"Delta_prime_model = {Delta_prime_model:.3e} <= 0 for alpha = {alpha_val:.3e} "
            f"(outside positive-FKR range); using Coppi branch fallback."
        )
        delta_hat_model = delta_hat_Coppi
    else:
        delta_hat_FKR = gS * gPr * (A**(-1.0 / 5.0)) * (((alpha_val * S)**(-2.0) * Delta_prime_model)**(1.0 / 5.0))
        p = 8
        delta_hat_model = (delta_hat_Coppi * delta_hat_FKR) / ((delta_hat_Coppi**p + delta_hat_FKR**p)**(1.0 / p))

    safety = getattr(params, 'inner_resolution_safety', 1.0)
    if safety is None or safety < 1.0:
        safety = 1.0

    delta_hat_grid = delta_hat_model / safety
    inner_scale = a * delta_hat_grid
    return float(inner_scale)


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