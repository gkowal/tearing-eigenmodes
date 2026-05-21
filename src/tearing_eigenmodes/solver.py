from .params import SimulationParams
from .exceptions import DeltaError, ConvergenceError
from .grid import select_NC
from .analysis import inner_layer_thickness
from psecas import Solver, ChebyshevRationalGrid
from psecas.systems.tearing_instability import TearingClassicalMHD, TearingGyrotropicMHD
from typing import Tuple, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

EigenmodesReturn = Tuple[
    Optional[np.ndarray],            # σ (eigenvalues)
    Optional[Dict[str, np.ndarray]], # s (eigenfunctions)
    Optional[float],                 # e (error/tolerance)
    Optional[float],                 # δin (inner layer thickness)
    Optional[int],                   # nin (inner layer nodes)
    Optional[int],                   # nwa (current sheet nodes)
    Optional[float],                 # C (grid scaling factor)
    Optional[int],                   # N (resolution)
    Optional[np.ndarray],            # z (grid zg)
    bool                             # success flag
]


def eigenmodes(params: SimulationParams) -> EigenmodesReturn:
    """
    Calculates tearing instability eigenmodes for a given set of parameters.
    """
    α       = params.alpha if params.alpha is not None else 0.1
    verbose = params.verbose

    try:
        Nmin         = params.Nmin
        Nmax         = params.Nmax
        Ninc         = params.Ninc
        C            = params.C
        a            = params.a if params.a is not None else 1.0
        w            = params.w if params.w is not None else 0.0
        CGL          = params.CGL
        S            = params.S if params.S is not None else 1.0e4
        Pr           = params.Pr
        ζ            = params.zeta
        ξ            = params.xi
        ϵ            = params.Hall
        β            = params.plasma_beta
        Δβ           = params.plasma_beta_difference
        ɣpar         = params.parallel_index
        ɣper         = params.perpendicular_index
        reσlo        = params.sigma_real_lower
        reσup        = params.sigma_real_upper
        imσlo        = params.sigma_imag_lower
        imσup        = params.sigma_imag_upper
        mode         = params.mode if params.mode is not None else 0
        atol         = params.atol
        rtol         = params.rtol
        gtol         = params.gtol
        δtol         = params.dtol
        orderby      = params.orderby
        allmodes     = params.allmodes
        noshear      = params.noshear

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
            Nlow, C = select_NC(params)
        else:
            Nlow = Nmin

        if C is None or C <= 0:
            raise ValueError("C must be positive.")

        kx = α/a

        # order = max(4 if Pr > 0 else 2, 3 if abs(ξ) > 0 else 2)
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

        δin, nin, nwa = inner_layer_thickness(system, δtol=δtol)

        if allmodes:
            return σ, v, e, δin, nin, nwa, C, N, system.grid.zg, True

        system_result = getattr(system, 'result')
        if np.isclose(σ, system_result['sigma']):
            σ = system_result['sigma']
            e = system_result['error']
            z = system_result['grid']
        else:
            logger.warning("Solver returned inconsistent eigenvalue!")
            σ = σ[0]
            e = e[0]
            z = system.grid.zg
        s = {}
        system_variables = getattr(system, 'variables')
        for key in system_variables:
            s[key] = system_result[key]

        logger.debug(
            f'Calculation done for α = {α:.4e} with C = {C:.3e} '
            f'({nin} points over the interval |z| < δin, {nwa} points over |z| < w+a):'
        )
        logger.debug(f'  σ₀ = {σ.real:.4e}{σ.imag:+.4e}j (error = {e:.3e}, N = {N})')

        return σ, s, e, δin, nin, nwa, C, N, z, True

    except DeltaError as ex:
        logger.debug(f"Stable eigenmode: {ex}")
        return None, None, None, None, None, None, None, None, None, False

    except ConvergenceError as ex:
        logger.debug(f"Insufficient resolution: {ex}")
        return None, None, None, None, None, None, None, None, None, False

    except ValueError as ex:
        logger.debug(f"Wrong parameter: {ex}")
        return None, None, None, None, None, None, None, None, None, False

    except Exception as ex:
        logger.warning(f"Solver failed for α = {α:.3e}: {ex}")
        return None, None, None, None, None, None, None, None, None, False
