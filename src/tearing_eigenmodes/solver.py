from .exceptions import DeltaError, ConvergenceError
from .grid import select_NC, inner_layer_thickness, find_peak_location
from psecas import Solver, ChebyshevRationalGrid
from psecas.systems.tearing_instability import TearingClassicalMHD, TearingGyrotropicMHD
import numpy as np
from typing import Tuple


def eos_indices(eos: str) -> Tuple[float, float]:
    """
    Return the two numerical indices that correspond to a given equation‑of‑state (EOS).

    Parameters
    ----------
    eos : str
        One of ``'adiabatic'``, ``'polytropic'`` or ``'isothermal'``.

    Returns
    -------
    Tuple[float, float]
        The pair of indices for the requested EOS.

    Raises
    ------
    ValueError
        If *eos* is not one of the supported values.
    """
    if eos == 'adiabatic':
        return 3.0, 2.0
    elif eos == 'polytropic':
        return 0.5, 2.0
    elif eos == 'isothermal':
        return 1.0, 1.0

    # If we reach this point, the EOS is unknown.
    raise ValueError(f"Unsupported equation of state: {eos!r}")


def estimate_max(params):
    """
    Determine the maximum growth rate and corresponding wavenumber.

    Parameters (kwargs)
    -------------------
    CGL          : bool  , enable CGL anisotropy (default False)
    S            : float , Lundquist number (default 1e4)
    Pr           : float , magnetic Prandtl number (used softly here;) (default 0)
    β, Δβ        : floats, gyrotropic plasma-β and anisotropy (defaults β=0, Δβ=0)
    ɣpar, ɣper   : floats, aidabatic indices (γ∥=3, γ⟂=2)

    Returns
    -------
    α_max : float
    Δ_max : float
        The wavenumber corresponding to maximum eigenmode.
    """
    CGL          = params.get('CGL'                   , False  )
    S            = params.get('S'                     ,    1.0e4 )
    Pr           = params.get('Pr'                    ,    0.0   )
    β            = params.get('plasma_beta'           ,    0.0   )
    Δβ           = params.get('plasma_beta_difference',    0.0   )
    ɣpar         = params.get('parallel_index'        ,    3.0   )
    ɣper         = params.get('perpendicular_index'   ,    2.0   )

    # --- Δ'(α); raise if stable at α
    if CGL and (Δβ <= - (2 + (ɣpar + ɣper - 2) * β) / ɣpar or Δβ >= 2):
        raise DeltaError(f"Δ' purely imaginary for Δβ = {Δβ:+.3e} => stable eigenmode for any α.")

    # --- CGL anisotropy factor and scaled wavenumbers
    C = 1 - Δβ/2
    μ = np.sqrt((2 + (ɣpar + ɣper - 2) * β + ɣpar * Δβ) / (2 - Δβ)) if CGL else 1

    # α_m scaling
    αm = 1.3583e+00 * (S / (S + 400))**(1/4) * S**(-1/4) * C**(-1/8) * μ**(-3/4) * ((0.05*Pr**2+0.7*Pr+1)/(12*Pr+1))**(1/8)
    Xm = αm * μ
    Δm = 2 * (1 / Xm - Xm)

    return αm, Δm


def eigenmodes(params):
    """
    Calculates tearing instability eigenmodes for a given set of parameters.

    Parameters:
    -----------
    params : dict
        Keyword arguments for physical and numerical parameters.
        Keys and default values:
        'CGL'      ( bool, default=False): Flag for Gyrotropic (True) or Stadard MHD (False).
        'Nmin'     (  int, default=128  ): Minimum number of collocation points.
        'Nmax'     (  int, default=1024 ): Maximum number of collocation points.
        'Ninc'     (  int, default=32   ): Increment in collocation points for iteration.
        'C'        (float, default=None ): Scaling factor for the rational Chebyshev grid.
                                           If None, it is auto-determined.
        'a'        (float, default=1.0  ): Half-width of the equilibrium current profile.
        'w'        (float, default=0.0  ): Half-width of the velocity shear region.
        'α'        (float, default=0.1  ): Wave number k multiplied by the half-width a.
        'ζ'        (float, default=1.0  ): Parameter controlling equilibrium magnetic field profile.
        'S'        (float, default=1e4  ): Lundquist number.
        'Pr'       (float, default=0.0  ): Prandtl number (or similar ratio).
        'ξ'        (float, default=0.0  ): The strength of the transversal magnetic field component.
        'ϵ'        (float, default=0.0  ): The strength of the Hall term.
        'β'        (float, default=1.0  ): Plasma beta.
        'Δβ'       (float, default=0.0  ): Plasma beta difference between parallel and perpendicular directions.
        'ɣpar'     (float, default=3.0  ): Parallel adiabatic index.
        'ɣper'     (float, default=2.0  ): Perpendicular adiabatic index.
        'σlower'   (float, default=1e-5 ): Lower bound for real part of eigenvalue search.
        'σupper'   (float, default=1.0  ): Upper bound for real part of eigenvalue search.
        'mode'     (  int, default=0    ): Index of the mode to track (0 is the fastest growing).
        'atol'     (float, default=1e-10): Absolute tolerance for convergence.
        'rtol'     (float, default=1e-5 ): Relative tolerance for convergence.
        'orderby'  (  str, default='amp'): Criterion to order eigenvalues: amplitude or tolerance.
        'allgrids' ( bool, default=False): Flag to iterate over all grids even if the convergence was reached.
        'verbose'  ( bool, default=False): Flag to enable verbose output.

    Returns:
    --------
    tuple: (σ, v, e, C, N_final, success)
        σ       (complex array or None): Eigenvalues (growth rates).
        v       (complex array or None): Eigenvectors (eigenmodes).
        e       (object or None)       : Error estimates.
        C       (float or None)        : Final scaling factor used in the grid.
        N       (int or None): Final number of collocation points.
        success (bool): True if the solver was successful, False otherwise.

    Raises:
    -------
    ValueError: If a required parameter is not positive or within a specified range.
    """
    '''
        Calculates tearing instability eigenmodes for a given set of parameters
    '''
    α       = params.get('alpha'   ,   0.1)
    verbose = params.get('verbose' , False)

    try:
        Nmin         = params.get('Nmin'                  ,  128          )
        Nmax         = params.get('Nmax'                  , 1024          )
        Ninc         = params.get('Ninc'                  ,   32          )
        n_inner      = params.get('n_inner'               ,    5          )
        decay_efolds = params.get('decay_efolds'          , -np.log(0.001))
        C            = params.get('C'                     , None          )
        a            = params.get('a'                     ,   1.0         )
        w            = params.get('w'                     ,   0.0         )
        CGL          = params.get('CGL'                   , False         )
        S            = params.get('S'                     ,   1.0e4       )
        Pr           = params.get('Pr'                    ,   0.0         )
        ζ            = params.get('zeta'                  ,   1.0         )
        ξ            = params.get('xi'                    ,   0.0         )
        ϵ            = params.get('Hall'                  ,   0.0         )
        β            = params.get('plasma_beta'           ,   1.0         )
        Δβ           = params.get('plasma_beta_difference',   0.0         )
        ɣpar         = params.get('parallel_index'        ,   3.0         )
        ɣper         = params.get('perpendicular_index'   ,   2.0         )
        reσlo        = params.get('sigma_real_lower'      ,   1.0e-6      )
        reσup        = params.get('sigma_real_upper'      ,   1.0         )
        imσlo        = params.get('sigma_imag_lower'      , -10.0         )
        imσup        = params.get('sigma_imag_upper'      ,  10.0         )
        mode         = params.get('mode'                  ,   0           )
        atol         = params.get('atol'                  ,   1.0e-10     )
        rtol         = params.get('rtol'                  ,   1.0e-5      )
        gtol         = params.get('gtol'                  ,   1.0e-2      )
        δtol         = params.get('dtol'                  ,   1.0e-3      )
        orderby      = params.get('orderby'               , 'errors'      )
        allmodes     = params.get('allmodes'              , False         )
        allgrids     = params.get('allgrids'              , False         )
        noshear      = params.get('noshear'               , False         )

        if α <= 0:
            raise ValueError("α must be positive.")
        if a <= 0:
            raise ValueError("a must be positive.")
        if not 0 <= ζ <= 1:
            raise ValueError("ζ must be between 0 and 1.")
        if S <= 0:
            raise ValueError("S must be positive.")
        if Pr < 0:
            raise ValueError("Pr cannot be negative.")
        if β < 0:
            raise ValueError("β cannot be negative.")
        if ɣpar <= 0:
            raise ValueError("ɣpar must be positive.")
        if ɣper <= 0:
            raise ValueError("ɣper must be positive.")
        if ϵ < 0:
            raise ValueError("ϵ cannot be negative.")

        if C is None:
            Nlow, C, δin = select_NC(params)
        else:
            Nlow = Nmin

        if C <= 0:
            raise ValueError("C must be positive.")

        kx = α/a

        order = max(4 if Pr > 0 else 2, 3 if abs(ξ) > 0 else 2)
        Ns    = np.arange(Nlow, Nmax+Ninc, Ninc)

        re_range=[reσlo, reσup]
        im_range=[imσlo, imσup]

        grid = ChebyshevRationalGrid(N=Ns[0], C=C, max_derivative_order=4)
        if CGL:
            system  = TearingGyrotropicMHD(grid, periodic=False, kx=kx, \
                                            a=a, S=S, Pr=Pr, β=β, Δβ=Δβ, \
                                            ɣpar=ɣpar, ɣper=ɣper, ϵ=ϵ)
        else:
            system  = TearingClassicalMHD(grid, periodic=False, kx=kx, \
                                          a=a, w=w, ζ=ζ, S=S, Pr=Pr, ξ=ξ, ϵ=ϵ, shear=not noshear)
        solver  = Solver(grid, system)

        σ, v, e = solver.iterate_solve_multimode(Ns, maxmode=mode, allmodes=allmodes, \
                     atol=atol, rtol=rtol, gtol=gtol, \
                     metric="complex", orderby=orderby, \
                     re_range=re_range, im_range=im_range, \
                     useOPinv=True, verbose=verbose)
        N = solver.grid.N

        δin, nin = inner_layer_thickness(system, δtol=δtol)
        l, nin = find_peak_location(system.result['duz'], system.result['dbz'], system.grid, a=a, w=w)

        if allmodes:
            return σ, v, e, δin, l, nin, C, N, system.grid.zg, True

        if np.isclose(σ, system.result['sigma']):
            σ = system.result['sigma']
            e = system.result['error']
            z = system.result['grid']
        else:
            print("[WARNING] Solver returned inconsistent eigenvalue!")
            σ = σ[0]
            e = e[0]
            z = system.grid.zg
        s = {}
        for key in system.variables:
            s[key] = system.result[key]

        if verbose:
            print(f'Calculation done for α = {α:.4e} with C = {C:.3e} ({nin} points over the interval |z| < δin):')
            print(f'  σ₀ = {σ.real:.4e}{σ.imag:+.4e}j (error = {e:.3e}, N = {N})')

        return σ, s, e, δin, l, nin, C, N, z, True

    except DeltaError as ex:
        if verbose:
            print(f"Stable eigenmode: {ex}")
        return None, None, None, None, None, None, None, None, None, False

    except ConvergenceError as ex:
        if verbose:
            print(f"Insufficient resolution: {ex}")
        return None, None, None, None, None, None, None, None, None, False

    except ValueError as ex:
        if verbose:
            print(f"Wrong parameter: {ex}")
        return None, None, None, None, None, None, None, None, None, False

    except Exception as ex:
        if verbose:
            print(f"[WARNING] Solver failed for α = {α:.3e}: {ex}")
        return None, None, None, None, None, None, None, None, None, False
