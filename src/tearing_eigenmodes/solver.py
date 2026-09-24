from .params import SimulationParams
from .exceptions import DeltaError, ConvergenceError
from .grid import select_NC, select_C_for_N
from .analysis import (
    measure_eigenmode_scales,
    minimum_eigenmode_scale,
    CLASSICAL_GRID_SCALE_KEYS,
    CGL_GRID_SCALE_KEYS,
    CLASSICAL_DIAGNOSTIC_SCALE_KEYS,
    CGL_DIAGNOSTIC_SCALE_KEYS,
)
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
                if val is not None and float(val) > 0.0:
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
                                 rtol=1e-5, atol=1e-10, gtol=1e-2,
                                 orderby="tolerance", metric="complex",
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
        if verbose:
            _print_modes(Σ_old[index], self.grid.N)

        mode, modes = _select(Σ_old.size, maxmode, allmodes)

        error = np.inf
        delta = np.inf
        errors: np.ndarray = np.array([])

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
                    if useEVguess and V_old is not None:
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
            if V_new is not None:
                V_new = V_new[:,index]

            mode, modes = _select(Σ_new.size, maxmode, allmodes)

            error = errors[mode]
            delta = deltas[mode]

            if verbose:
                _print_modes(Σ_new, self.grid.N, errors=errors, case=case, delta=delta, error=error)

            if error <= 1.0:
                v_mode = V_new[:,mode] if V_new is not None else None
                self.keep_result(Σ_new[mode], v_mode, mode)
                self.system.result.update({"converged": True})
                self.system.result.update({"error": error})
                self.system.result.update({"grid": self.grid.zg})
                if allmodes:
                    v_res = V_new[:, :modes] if V_new is not None else None
                    return Σ_new[:modes], v_res, errors[:modes]
                v_res = V_new[:,mode] if V_new is not None else None
                return Σ_new[mode], v_res, errors[mode]

            Σ_old = Σ_new.copy()
            V_old = V_new.copy() if V_new is not None else None
            grid_old = copy.deepcopy(self.grid)

        if errors.size == 0:
            # A single resolution gives no convergence estimate: report the
            # modes of the one full solve as unconverged instead of failing.
            errors = np.full(Σ_old.size, np.inf)

        v_old_mode = V_old[:,mode] if V_old is not None else None
        self.keep_result(Σ_old[mode], v_old_mode, mode)
        self.system.result.update({"converged": False})
        self.system.result.update({"error": error})
        self.system.result.update({"grid": self.grid.zg})

        if allmodes:
            v_old_res = V_old[:, :modes] if V_old is not None else None
            return Σ_old[:modes], v_old_res, errors[:modes]
        v_old_res = V_old[:,mode] if V_old is not None else None
        return Σ_old[mode], v_old_res, errors[mode]
from .systems import TearingClassicalMHD, TearingGyrotropicMHD
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

EigenmodesReturn = Tuple[
    Any,                             # σ (eigenvalues)
    Any,                             # s (eigenfunctions)
    Any,                             # e (error/tolerance)
    Optional[float],                 # δin (inner layer thickness)
    Optional[int],                   # nin (inner layer nodes)
    Optional[int],                   # nwa (current sheet nodes)
    Optional[float],                 # C (grid scaling factor)
    Optional[int],                   # N (resolution)
    Optional[np.ndarray],            # z (grid zg)
    bool                             # success flag
]


def resolution_sequence(Nlow: int, Nmax: int, Ninc: int) -> np.ndarray:
    """
    Resolutions Nlow, Nlow+Ninc, ... capped at Nmax, always ending at Nmax.
    """
    Ns = np.arange(Nlow, Nmax + 1, Ninc)
    if Ns.size == 0 or Ns[-1] != Nmax:
        Ns = np.append(Ns, Nmax)
    return Ns


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
        Ns    = resolution_sequence(Nlow, Nmax, Ninc)

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
        solver  = TearingSolver(grid, system, gevp_method=params.gevp_method or 'qz')

        σ, v, e = solver.iterate_solve_multimode(Ns, maxmode=mode, allmodes=allmodes, \
                     atol=atol, rtol=rtol, gtol=gtol, \
                     metric="complex", orderby=orderby, \
                     re_range=re_range, im_range=im_range, \
                     useOPinv=True, verbose=verbose)
        N = solver.grid.N
        C = solver.grid.C

        system_result = getattr(system, 'result')
        is_converged = bool(system_result.get("converged", True) and float(np.atleast_1d(e)[0]) <= 1.0)

        I_wa = np.where(np.abs(system.grid.zg) <= (w + a))
        nwa = I_wa[0].size

        scales_supported = True
        if is_converged:
            # Multi-scale eigenmode analysis for converged mode
            try:
                scales = measure_eigenmode_scales(system)
            except NotImplementedError as ex:
                # The eigenmode itself is valid (e.g. Hall branches);
                # only the dominance-scale diagnostics are unavailable.
                logger.debug(f"Scale diagnostics unavailable: {ex}")
                scales_supported = False

        if is_converged and scales_supported:
            model_name = "cgl" if CGL else "classical"
            min_scale, min_key = minimum_eigenmode_scale(scales, model=model_name)

            if min_scale is not None and np.isfinite(min_scale) and min_scale > 0.0:
                δin = min_scale
                I = np.where(np.abs(system.grid.zg) <= δin)
                nin = max(1, I[0].size)
                min_scale_val = min_scale
                min_key_str = min_key if min_key is not None else ""
            else:
                δin = float("nan")
                nin = 0
                min_scale_val = float("nan")
                min_key_str = ""

            # Resistive diagnostic scale
            res_key = "cgl.bz_induction.eta_vs_ideal" if CGL else "classical.bz_induction.eta_vs_ideal"
            res_scale = scales.get(res_key, float("nan"))
            if res_scale is not None and np.isfinite(res_scale) and res_scale > 0.0:
                res_nodes = max(1, np.where(np.abs(system.grid.zg) <= res_scale)[0].size)
            else:
                res_scale = float("nan")
                res_nodes = 0
        else:
            # Solve did not converge or diagnostics are unsupported:
            # populate all-nan/zero invalid dictionaries
            candidate_keys = CGL_DIAGNOSTIC_SCALE_KEYS if CGL else CLASSICAL_DIAGNOSTIC_SCALE_KEYS
            scales = {k: float("nan") for k in candidate_keys}
            δin = float("nan")
            nin = 0
            min_scale_val = float("nan")
            min_key_str = ""
            res_scale = float("nan")
            res_nodes = 0

        system_result["mode_scales"] = scales
        system_result["minimum_physical_scale"] = min_scale_val
        system_result["minimum_physical_scale_key"] = min_key_str
        system_result["minimum_scale_nodes"] = nin
        system_result["resistive_layer_thickness"] = res_scale
        system_result["resistive_layer_nodes"] = res_nodes

        # Print verbose deterministic mode scales line once for final mode if converged
        emit_scale = getattr(params, "emit_scale_summary", True)
        if verbose and is_converged and emit_scale:
            from .printing import format_mode_scale_summary
            summary = format_mode_scale_summary(scales)
            if summary:
                logger.info(summary)

        if allmodes:
            return σ, v, e, δin, nin, nwa, C, N, system.grid.zg, True

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

        # Populate multiscale diagnostics for serialization
        s["mode_scales"] = system_result["mode_scales"]
        s["minimum_physical_scale"] = system_result["minimum_physical_scale"]
        s["minimum_physical_scale_key"] = system_result["minimum_physical_scale_key"]
        s["minimum_scale_nodes"] = system_result["minimum_scale_nodes"]
        s["resistive_layer_thickness"] = system_result["resistive_layer_thickness"]
        s["resistive_layer_nodes"] = system_result["resistive_layer_nodes"]

        limiting_str = f" [limiting: {min_key_str}]" if min_key_str else ""
        logger.debug(
            f'Calculation done for α = {α:.4e} with C = {C:.3e} '
            f'({nin} points over interval |z| < δin={δin:.3e}{limiting_str}, {nwa} points over |z| < w+a):'
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
