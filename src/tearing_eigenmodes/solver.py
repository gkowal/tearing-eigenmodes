from .params import SimulationParams
from .exceptions import DeltaError, ConvergenceError
from .grid import select_NC, select_C_for_N
from .analysis import inner_layer_thickness
from psecas import Solver, ChebyshevRationalGrid

class TearingChebyshevRationalGrid(ChebyshevRationalGrid):
    """
    A ChebyshevRationalGrid subclass that dynamically adjusts the scaling factor C
    when the grid resolution N changes to preserve the inner layer resolution constraints.
    """
    def __init__(self, N, C=1, z="z", max_derivative_order=2, params=None, dynamic_C=False):
        self.params = params
        self.dynamic_C = dynamic_C
        self.current_sigma = None
        super().__init__(N, C, z, max_derivative_order)

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, value):
        self._N = value
        if self.dynamic_C and self.params is not None:
            # Prefer the initial high-quality params.sigma guess if provided
            sigma_for_scale = self.params.sigma
            is_zero_or_negative = True
            if sigma_for_scale is not None:
                val = getattr(sigma_for_scale, 'real', sigma_for_scale)
                if isinstance(val, np.ndarray):
                    val = np.atleast_1d(val)[0]
                if float(val) > 0.0:
                    is_zero_or_negative = False

            if is_zero_or_negative:
                sigma_for_scale = self.current_sigma

            _, _, new_C = select_C_for_N(value, self.params, current_sigma=sigma_for_scale)
            self._C = new_C
        self.make_grid()


class TearingSolver(Solver):
    """
    A Solver subclass that updates the current growth rate estimate in the grid
    before each resolution step during the convergence iteration loop.
    """
    def iterate_solve_multimode(self, Ns, maxmode=None, allmodes=False,
                                 atol=1e-10, rtol=1e-5, gtol=1e-2,
                                 metric="complex", orderby="tolerance",
                                 re_range=None, im_range=None,
                                 useOPinv=True, useEVguess=True, verbose=False):
        import numpy as np
        import copy

        def _print_modes(Σ, N, errors=None, case=None, delta=None, error=None):
            n = Σ.size
            fmt = f" {n:2d}" if n < 100 else ">99"
            print(f"N: {N:4d}, C: {self.grid.C:.4e}, {fmt} eigenvalue{'s' if n > 1 else ' '}: ", end='')
            m = min(3, n)
            if case is None:
                for i in range(m):
                    print(f" {Σ[i]:.4e}", end='')
                print(" ..." if n > m else '', ' '*10)
            else:
                if errors is None:
                    for i in range(m):
                        print(f" {Σ[i]:.4e}", end='')
                else:
                    for i in range(m):
                        print(f" {Σ[i]:.4e} ({errors[i]:.2e})", end='')
                print(" ..." if n > m else '', end='')
                if delta is not None:
                    print(f", Δσ/σ : {delta:.2e}", end='')
                if error is not None:
                    print(f", error : {error:.2e}", end='')
                print(f"{case}", ' '*10)

        def _errors(Σ_new, Σ_old, rtol=1e-5, atol=1e-10, metric='complex', orderby='tolerance'):
            errors = []
            deltas = []
            if metric == 'real':
                for i in range(Σ_new.size):
                    ΔΣ  = np.abs(Σ_old.real - Σ_new[i].real)
                    j   = np.argsort(ΔΣ)[0]
                    err = ΔΣ[j] / (atol + rtol * max(np.abs(Σ_new[i].real), np.abs(Σ_old[j].real)))
                    errors.append(err)
            elif metric == 'imag':
                for i in range(Σ_new.size):
                    ΔΣ  = np.abs(Σ_old.imag - Σ_new[i].imag)
                    j   = np.argsort(ΔΣ)[0]
                    err = ΔΣ[j] / (atol + rtol * max(np.abs(Σ_new[i].imag), np.abs(Σ_old[j].imag)))
                    errors.append(err)
            else:
                for i in range(Σ_new.size):
                    ΔΣ  = np.abs(Σ_old - Σ_new[i])
                    j   = np.argsort(ΔΣ)[0]
                    err = ΔΣ[j] / (atol + rtol * max(np.abs(Σ_new[i]), np.abs(Σ_old[j])))
                    errors.append(err)
            for i in range(Σ_new.size):
                ΔΣ  = np.abs(Σ_old - Σ_new[i])
                j   = np.argsort(ΔΣ)[0]
                fac = 1.0 + np.abs(Σ_new[i].imag) / max(atol, Σ_new[i].real)
                dlt = ΔΣ[j] / max(atol, np.abs(Σ_new[i])) * fac
                deltas.append(dlt)
            deltas = np.array(deltas)
            errors = np.array(errors)
            if orderby in ['amplitude', 'magnitude']:
                index = np.argsort(np.abs(Σ_new))[::-1]
            elif orderby in ['real_part', 'real']:
                index = np.argsort(Σ_new.real)[::-1]
            elif orderby in ['imag_part', 'imag', 'imaginary']:
                index = np.argsort(Σ_new.imag)[::-1]
            else:
                index = np.argsort(errors)
            return errors[index], deltas[index], index

        def _select(Nmodes, maxmode, allmodes):
            if Nmodes <= 0:
                return 0, 0
            if maxmode is None:
                m = 0
                k = Nmodes if allmodes else 1
                return m, k
            m = max(0, min(int(maxmode), Nmodes - 1))
            k = (m + 1) if allmodes else 1
            return m, k

        self.grid.N = Ns[0]
        Σ, V = self.solve_full()
        Σ_old, V_old = self.filter_modes(Σ, V, re_range=re_range, im_range=im_range)
        grid_old = copy.deepcopy(self.grid)
        if orderby in ['real_part', 'real']:
            index = np.argsort(Σ_old.real)[::-1]
        elif orderby in ['imag_part', 'imag', 'imaginary']:
            index = np.argsort(Σ_old.imag)[::-1]
        else:
            index = np.argsort(np.abs(Σ_old))[::-1]
        _print_modes(Σ_old[index], self.grid.N)

        mode, modes = _select(Σ_old.size, maxmode, allmodes)

        error = np.inf
        delta = np.inf

        for N in Ns[1:]:
            if hasattr(self.grid, 'current_sigma') and Σ_old.size > 0:
                self.grid.current_sigma = Σ_old[mode]
            self.grid.N = N
            if delta > gtol:
                case = ''
                Σ, V = self.solve_full()
            else:
                case = ' [with guess]'
                Σ = []
                V = []
                for i in range(modes):
                    σ0 = Σ_old[i]
                    if useEVguess:
                        v0 = self.prolongate_eigenvector(V_old[:,i], grid_old)
                    else:
                        v0 = None
                    σ, v = self.solve_mode(σ0, v0=v0, useOPinv=useOPinv, verbose=verbose)
                    Σ.append(σ)
                    V.append(v)
                Σ = np.array(Σ)
                V = np.array(V).T

            try:
                Σ_new, V_new = self.filter_modes(Σ, V, re_range=re_range, im_range=im_range)
            except ValueError:
                case = ' [guess failed → full]'
                Σ, V = self.solve_full()
                Σ_new, V_new = self.filter_modes(Σ, V, re_range=re_range, im_range=im_range)

            errors, deltas, index = _errors(Σ_new, Σ_old, rtol=rtol, atol=atol, metric=metric, orderby=orderby)

            Σ_new = Σ_new[index]
            V_new = V_new[:,index]

            mode, modes = _select(Σ_new.size, maxmode, allmodes)

            error = errors[mode]
            delta = deltas[mode]

            _print_modes(Σ_new, self.grid.N, errors=errors, case=case, delta=delta, error=error)

            if error <= 1.0:
                self.keep_result(Σ_new[mode], V_new[:,mode], mode)
                self.system.result.update({"converged": True})
                self.system.result.update({"error": error})
                self.system.result.update({"grid": self.grid.zg})
                if allmodes:
                    return Σ_new[:modes], V_new[:, :modes], errors[:modes]
                return Σ_new[mode], V_new[:,mode], errors[mode]

            Σ_old = Σ_new.copy()
            V_old = V_new.copy()
            grid_old = copy.deepcopy(self.grid)

        self.keep_result(Σ_old[mode], V_old[:,mode], mode)
        self.system.result.update({"converged": False})
        self.system.result.update({"error": error})
        self.system.result.update({"grid": self.grid.zg})

        if allmodes:
            return Σ_old[:modes], V_old[:, :modes], errors[:modes]
        return Σ_old[mode], V_old[:,mode], errors[mode]
from .systems import TearingClassicalMHD, TearingGyrotropicMHD
from typing import Tuple, Dict, Any, Optional, Union
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
    σ       = params.sigma if params.sigma is not None else 0.0+0.0j
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

        grid = TearingChebyshevRationalGrid(N=Ns[0], C=C, max_derivative_order=4, params=params, dynamic_C=params.dynamic_C)
        grid.current_sigma = σ
        system: Union[TearingClassicalMHD, TearingGyrotropicMHD]
        if CGL:
            system  = TearingGyrotropicMHD(grid, periodic=False, kx=kx, \
                                            a=a, S=S, Pr=Pr, β=β, Δβ=Δβ, \
                                            ɣpar=ɣpar, ɣper=ɣper, ϵ=ϵ, σ=σ.real)
        else:
            system  = TearingClassicalMHD(grid, periodic=False, kx=kx, \
                                          a=a, w=w, ζ=ζ, S=S, Pr=Pr, ξ=ξ, ϵ=ϵ, shear=not noshear)
        solver  = TearingSolver(grid, system)

        σ, v, e = solver.iterate_solve_multimode(Ns, maxmode=mode, allmodes=allmodes, \
                     atol=atol, rtol=rtol, gtol=gtol, \
                     metric="complex", orderby=orderby, \
                     re_range=re_range, im_range=im_range, \
                     useOPinv=True, verbose=verbose)
        N = solver.grid.N
        C = solver.grid.C

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

        if hasattr(σ, 'item'):
            σ = σ.item()
        if hasattr(e, 'item'):
            e = e.item()
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
