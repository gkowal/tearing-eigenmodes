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


def eigenmodes(params: Dict[str, Any]) -> EigenmodesReturn:
    """
    Calculates tearing instability eigenmodes for a given set of parameters.
    """
    α       = params.get('alpha'   ,   0.1)
    verbose = params.get('verbose' , False)

    try:
        Nmin         = params.get('Nmin'                  ,   64          )
        Nmax         = params.get('Nmax'                  , 2048          )
        Ninc         = params.get('Ninc'                  ,   32          )
        C            = params.get('C'                     , None          )
        a            = params.get('a'                     ,   1.0         )
        w            = params.get('w'                     ,   0.0         )
        CGL          = params.get('CGL'                   , False         )
        S            = params.get('S'                     ,   1.0e4       )
        Pr           = params.get('Pr'                    ,   0.0         )
        ζ            = params.get('zeta'                  ,   1.0         )
        ξ            = params.get('xi'                    ,   0.0         )
        ϵ            = params.get('Hall'                  ,   0.0         )
        β            = params.get('plasma_beta'           ,   0.0         )
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
        orderby      = params.get('orderby'               , 'real'        )
        allmodes     = params.get('allmodes'              , False         )
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
            Nlow, C = select_NC(params)
        else:
            Nlow = Nmin

        if C <= 0:
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
