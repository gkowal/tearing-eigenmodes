from .exceptions import DeltaError, ConvergenceError
import numpy as np


def select_NC(params):
    """
    Determine the Chebyshev–TB grid resolution N and scaling factor C for the linear tearing
    instability eigenproblem under the mapping z = C tan(θ).
    """
    Nmin         = params.get('Nmin'                  ,  128       )
    Nmax         = params.get('Nmax'                  , 1024       )
    Ninc         = params.get('Ninc'                  ,   32       )
    n_inner      = params.get('n_inner'               ,    5       )
    decay_efolds = params.get('decay_efolds'          ,    3.51    )
    CGL          = params.get('CGL'                   , False      )
    S            = params.get('S'                     ,    1.0e4   )
    Pr           = params.get('Pr'                    ,    0.0     )
    β            = params.get('plasma_beta'           ,    0.0     )
    Δβ           = params.get('plasma_beta_difference',    0.0     )
    ɣpar         = params.get('parallel_index'        ,    3       )
    ɣper         = params.get('perpendicular_index'   ,    2       )
    a            = params.get('a'                     ,    1       )
    w            = params.get('w'                     ,    0.0     )
    α            = params.get('alpha'                 ,    0.1     )
    θ            = params.get('theta'                 ,    8.0     )
    δ            = params.get('delta'                 , None       )
    σ            = params.get('sigma'                 , None       )
    ξ            = params.get('xi'                    ,    0.0     )
    verbose      = params.get('verbose'               , False      )
    Cmean        = params.get('Cmean'                 , 'geometric')

    # Enforce odd n_inner (so m is integer and z=0 is a collocation point)
    if n_inner % 2 == 0:
        n_inner += 1
    m = (n_inner - 1) / 2

    # --- Δ'(α); raise if stable at α
    if CGL and (Δβ <= - (2 + (ɣpar + ɣper - 2) * β) / ɣpar or Δβ >= 2):
        raise DeltaError(f"Δ' purely imaginary for Δβ = {Δβ:+.3e} => stable eigenmode for α = {α:.3e}.")

    # --- CGL anisotropy factor and scaled wavenumbers
    C = 1 - Δβ/2
    μ = np.sqrt((2 + (ɣpar + ɣper - 2) * β + ɣpar * Δβ) / (2 - Δβ)) if CGL else 1
    λ = α / μ

    if δ is None:
        # --- Δ'(α); raise if stable at α
        X = α * μ
        Δ = 2 * (1 / X - X)
        if Δ <= 0.0:
            raise DeltaError(f"Δ' <= 0 (Δ = {Δ:.3e}) ⇒ stable eigenmode for α = {α:.3e}.")

        # -- Prefactors from fits to numerical results
        fS   = (3.8288e-2 / (4.7443e-2 + S**-0.46355) + 7.9649e-2)
        gS   = 0.91451 - 2.0654 / (S**0.37651 + 0.88448)
        fPr  = (1.0777 + ((Pr * 0.71554) * (5.765 + Pr)))**0.079176
        gPr  = 2.4379 * ((Pr + 4.5991e-3)**0.16186)

        # --- Inner-layer widths from theoretical scalings modified by prefactors
        δCop = fS * fPr * a * (α * S)**(-1.0/3.0)
        δFKR = gS * gPr * a * ((S * α)**-2 * a * Δ)**(1.0/5.0)

        # --- Smooth the inner-layer thickness from both regimes
        δα = δCop * δFKR / (δCop**θ + δFKR**θ)**(1.0/θ)
        if verbose:
            print(f"[estimated inner scales for α={α:.4e}] a={a:.4e}, δCop={δCop:.4e}, δFKR={δFKR:.4e}, δα={δα:.4e}")
    else:
        δα = δ
        if verbose:
            print(f"[provided inner scales for α={α:.4e}] a={a:.4e}, δα={δα:.4e}")

    πh   = np.pi / 2
    πm   = np.pi * m
    Ntop = Nmax - 3 * Ninc

    # --- Outer requirement: amplitude reduced by e^{-decay_efolds} at |z|max
    zmin = a + w
    zmax = decay_efolds / λ
    if verbose:
        print(f"[estimated for α={α:.4e}] zmin={zmin:.4e}, zmax={zmax:.4e}")
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

    if verbose:
        print(f"[all scales for α={α:.4e}] (w+a) = {w+a:.4e}, lσ = {lσ:.4e}, lk = {lk:.4e}, zmin = {zmin:.4e}, zmax = {zmax:.4e}, Nmin = {Nmin:4d}")

    # Iterate N upward until C_out(N) <= C_in(N); then set C = C_out(N).
    N  = Nmin
    while True:
        if N > Ntop:
            raise ConvergenceError(
                f"Insufficient N up to Nmax={Nmax}: cannot satisfy C_out(N) <= C_in(N). "
                f"Try increasing Nmax or relaxing n_inner/decay_efolds."
            )
        Np   = N + 1
        Cinn  = zmin / np.tan(πm / Np)
        Cout  = zmax * np.tan(πh / Np)
        if verbose:
            print(f"[grid scale constrains for α={α:.4e}] Nmin = {N:4d}  C_inner={Cinn:.4e}  C_outer={Cout:.4e}")
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

    if verbose:
        print(f"[grid scale constrains for α={α:.4e}] Nmin = {N:4d}  C_inner={Cinn:.4e}  C_outer={Cout:.4e} => C = {C:.6e} using {Cmean} mean")

    return N, C, δα
