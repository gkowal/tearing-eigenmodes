from .exceptions import DeltaError, ConvergenceError
import numpy as np
import logging

logger = logging.getLogger(__name__)


def select_NC(params):
    """
    Determine the Chebyshev–TB grid resolution N and scaling factor C for the linear tearing
    instability eigenproblem under the mapping z = C tan(θ).

    Parameters
    ----------
    params : dict
        A dictionary containing physical and numerical parameters.

    Returns
    -------
    tuple
        A tuple (N, C) representing the determined resolution and scaling factor.
    """
    Nmin         = params.get('Nmin'                  ,  128       )
    Nmax         = params.get('Nmax'                  , 1024       )
    Ninc         = params.get('Ninc'                  ,   32       )
    n_inner      = params.get('n_inner'               ,    5       )
    decay_efolds = params.get('decay_efolds'          ,    3.51    )
    CGL          = params.get('CGL'                   , False      )
    β            = params.get('plasma_beta'           ,    0.0     )
    Δβ           = params.get('plasma_beta_difference',    0.0     )
    ɣpar         = params.get('parallel_index'        ,    3       )
    ɣper         = params.get('perpendicular_index'   ,    2       )
    a            = params.get('a'                     ,    1       )
    w            = params.get('w'                     ,    0.0     )
    α            = params.get('alpha'                 ,    0.1     )
    σ            = params.get('sigma'                 , None       )
    ξ            = params.get('xi'                    ,    0.0     )
    Cmean        = params.get('Cmean'                 , 'geometric')

    # Enforce odd n_inner (so m is integer and z=0 is a collocation point)
    if n_inner % 2 == 0:
        n_inner += 1
    m = (n_inner - 1) / 2

    # --- λ: decaying factor of the outer solution;
    if CGL:
        R = 0.0 if σ is None else σ.real**2 / α**2
        C = 1.0 + R - Δβ/2
        D = 1.0 + R + 0.5 * ((ɣpar + ɣper - 2) * β + ɣpar * Δβ)
        if np.isclose(D, 0.0):
            raise DeltaError(f"Infinite decaying factor for (β, Δβ) = ({β:.3e}, {Δβ:+.3e}) => stable eigenmode for α = {α:.3e}.")

        μ = C / D
        if μ < 0.0:
            raise DeltaError(f"Decaying factor purely imaginary for (β, Δβ) = ({β:.3e}, {Δβ:+.3e}) => stable eigenmode for α = {α:.3e}.")
        λ = np.sqrt(μ) * α
    else:
        λ = α

    # --- Outer requirement: amplitude reduced by e^{-decay_efolds} at |z|max
    zmin = a + w
    zmax = decay_efolds / λ
    lk = ξ / α
    if σ is not None and ξ > 0.0:
        lσ = ξ / σ.real
        lk = 2.0 * np.pi * ξ / np.abs(α + σ.imag)

        if lσ <= zmin:
            zmin = np.sqrt(zmin * lσ)
        else:
            zmax = max(zmax, decay_efolds * lσ)
    else:
        lσ   = 0.0

    logger.debug(f"[all scales for α={α:.4e}] (w+a) = {w+a:.4e}, lσ = {lσ:.4e}, lk = {lk:.4e}, zmin = {zmin:.4e}, zmax = {zmax:.4e}, Nmin = {Nmin:4d}")

    # Iterate N upward until C_out(N) <= C_in(N); then set C = C_out(N).
    πh   = np.pi / 2
    πm   = np.pi * m
    Ntop = Nmax - 3 * Ninc

    N  = Nmin
    while True:
        if N > Ntop:
            raise ConvergenceError(
                f"Insufficient N up to Nmax={Nmax}: cannot satisfy C_out(N) <= C_in(N). "
                f"Try increasing Nmax or relaxing n_inner/decay_efolds."
            )
        Np   = N + 1
        Cinn = zmin / np.tan(πm / Np)
        Cout = zmax * np.tan(πh / Np)
        if Cout <= Cinn:
            break

        N += Ninc

    if Cmean == 'harmonic':
        C = 2.0 * Cinn * Cout / (Cinn + Cout)
    elif Cmean == 'geometric':
        C = np.sqrt(Cinn * Cout)
    elif Cmean == 'average':
        C = 0.5 * (Cinn + Cout)
    elif Cmean in ['lower', 'outer']:
        C = Cout
    elif Cmean in ['upper', 'inner']:
        C = Cinn
    else:
        C = Cinn

    logger.debug(f"[grid scale constrains for α={α:.4e}] Nmin = {N:4d}  C_inner={Cinn:.4e}  C_outer={Cout:.4e} => C = {C:.6e} using {Cmean} mean")

    return N, C
