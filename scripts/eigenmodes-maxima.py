#!/usr/bin/env python3
#
import os, sys, time, logging, signal
from typing import Any, Dict, Tuple, List, Optional, Callable, Deque
import multiprocessing as mp
import numpy as np

from collections import deque
from functools import lru_cache
from tearing_eigenmodes import build_params, build_dpath, \
                             print_info, refine_wavenumber_bracket, \
                             refine_eigenvalues, refine_resistive_scale, \
                             eigenmodes, write_results, DeltaError, \
                             estimate_max, save_eigenmode, setup_logging, \
                             check_state, compile_metadata, SimulationParams

counter: Any = None

class Extrapolator:
    """
    Tracks a history of (x, y) pairs and extrapolates y at a new x
    using polynomial fitting (degree 1=linear, 2=quadratic, 3=cubic).
    Falls back gracefully when insufficient history is available.
    Direction-agnostic: works for both increasing and decreasing x sweeps.
    """
    def __init__(self, maxdeg: int = 2, minpoints: int = 2, maxhistory: int = 6, ymin: Optional[float] = None) -> None:
        self.maxdeg: int = maxdeg
        self.minpoints: int = minpoints
        self.ymin: Optional[float] = ymin        # optional lower clamp on predicted value
        self.xs: Deque[float] = deque(maxlen=maxhistory)
        self.ys: Deque[Any] = deque(maxlen=maxhistory)

    def add(self, x: float, y: Any) -> None:
        self.xs.append(float(x))
        self.ys.append(y)

    def predict(self, x_new: float) -> Optional[float]:
        n = len(self.xs)
        if n < self.minpoints:
            return None                          # not enough history yet
        deg = min(self.maxdeg, n - 1)           # can't exceed n-1
        xs  = np.array(self.xs)
        ys  = np.array(self.ys)
        # centre & scale for numerical stability; ptp() is direction-agnostic
        x0     = xs.mean()
        xscale = (xs.max() - xs.min()) or 1.0
        coeffs = np.polyfit((xs - x0) / xscale, ys, deg)
        y_pred = float(np.polyval(coeffs, (x_new - x0) / xscale))
        if self.ymin is not None:
            y_pred = max(y_pred, self.ymin)
        return y_pred

    def __len__(self) -> int:
        return len(self.xs)

def init_worker(shared_counter: Any) -> None:
    """Assign the shared object to the global variable in this worker."""
    # Worker processes should ignore SIGINT; only the main process will handle it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global counter
    counter = shared_counter

def make_objective(params_base: SimulationParams) -> Callable[[float], float]:
    import copy
    # capture a shallow copy once; treat as immutable thereafter
    params_fixed = copy.copy(params_base)

    @lru_cache(maxsize=1024)
    def objective(αq: float) -> float:
        if αq <= 0:
            return 0.0
        p = copy.copy(params_fixed)
        p.alpha = αq
        σ, _, _, _, _, _, _, _, _, status = eigenmodes(p)
        return float(-σ.real) if (status and σ is not None) else 0.0

    def f(α: float) -> float:
        return objective(round(α, 12))

    return f

def task(value: float, αbracket: Optional[List[float]], sigma: Any, delta: Optional[float], params: SimulationParams) -> Tuple[Optional[float], Optional[np.ndarray], Optional[float], Optional[int], bool]:
    from scipy.optimize import bracket, minimize_scalar

    global counter

    status = False

    import copy
    params_base = copy.copy(params)

    # Initialize variables to satisfy static analysis
    αm: float = 0.0
    σm: Optional[np.ndarray] = None
    e: Optional[float] = None
    δin: Optional[float] = None
    nin: Optional[int] = None
    nwa: Optional[int] = None
    N: Optional[int] = None
    C: Optional[float] = None
    nit: int = 0
    Δα: float = 0.0
    Δσ: Any = 0.0
    z: Optional[np.ndarray] = None
    s: Optional[Dict[str, np.ndarray]] = None

    ntasks     = params_base.ntasks
    dependence = params_base.dependence if params_base.dependence is not None else 'S'
    verbose    = params_base.verbose
    force      = params_base.force
    Nmax       = params_base.Nmax
    rtol       = params_base.rtol
    wtol       = 1e-3
    w          = params_base.w if params_base.w is not None else 0.0
    a          = params_base.a if params_base.a is not None else 1.0

    UP = '\033[F'
    info = f"  {dependence} = {value:+.3e}: "
    progress_line = ''
    bracket_line = ''
    result_line = ''

    dep_map = {
        'a':  'a',
        'w':  'w',
        'S':  'S',
        'Pr': 'Pr',
        'ξ':  'xi',
        'ϵ':  'Hall',
        'β':  'plasma_beta',
        'Δβ': 'plasma_beta_difference'
    }
    setattr(params_base, dep_map[dependence], float(value))

    sname = os.path.join(params_base.data_path if params_base.data_path is not None else './', f'state_{dependence}{value:+.6e}.npz')

    status, state_data = check_state(sname, force=force, Nmax=Nmax)
    if status:
        assert state_data is not None
        αm  = float(state_data['wavenumber'])
        σm  = np.atleast_1d(state_data['eigenvalue'])
        e   = float(state_data['tolerance'])
        δin = float(state_data['resistive_layer_thickness'])
        nin = int(state_data['resistive_layer_nodes'])
        nwa = int(state_data['current_sheet_nodes'])
        N   = int(state_data['resolution'])
        C   = float(state_data['grid_scaling_factor'])
        nit = int(state_data['niter'])
        Δα  = float(state_data['wavenumber_error'])
        Δσ  = float(state_data['eigenvalue_error'])

    if not status:
        status = True
        if αbracket is None or len(αbracket) != 2:
            try:
                αm, _ = estimate_max(params_base)
                αlo = αm * (1 - wtol)
                αup = αm * (1 + wtol)
            except DeltaError as ex:
                status = False
                logging.info(f"Stable eigenmode: {ex}")
        else:
            αlo, αup = αbracket
        if status:
            logging.debug(f"Initial bracket for {dependence}={value:+.3e}: α = {αlo:.4e} … {αup:.4e}")
            αu = (αup + 1.618 * αlo) / 2.618

            try:
                params_base.sigma   = sigma
                params_base.delta   = delta

                f = make_objective(params_base)

                αa, αb, αc, σa, σb, σc, fn = bracket(f, xa=αlo, xb=αu)
                if αa > αc:
                    αa, αc, σa, σc = αc, αa, σc, σa
                if αa <= 0:
                    αa = 1e-3
                bracket_line = info + f"α-bracket = [ {αa:.3e}, {αb:.3e}, {αc:.3e} ],  σ-values = [ {-σa:.3e}, {-σb:.3e}, {-    σc:.3e} ]  after {fn} function calls" + ' '*4
                if verbose:
                    logging.info(f"{bracket_line}")
                else:
                    print(f"\r{bracket_line}\n{result_line}\n\n{progress_line}{UP}{UP}{UP}", end='', flush=True)
                bracket_line = ''

                Δα = wtol * αa
                res = minimize_scalar(f, bracket=(αa, αb, αc), method='brent', options={'xtol': Δα})

                if verbose:
                    logging.debug(res)
                    logging.info('Final refinement:')
                αm  = res.x
                params_final = copy.copy(params_base)
                params_final.alpha = αm
                σ, s, e, δin, nin, nwa, C, N, z, status = eigenmodes(params_final)
                if status:
                    assert σ is not None
                    assert e is not None
                    assert δin is not None
                    assert C is not None
                    assert nin is not None
                    assert nwa is not None
                    assert N is not None
                    assert s is not None
                    σm  = np.atleast_1d(σ)
                    Δσ  = rtol * σm[0].real * e
                    nit = res.nfev

                    # Include physical and numerical parameters for reproducibility
                    metadata = compile_metadata(params_final)

                    save_eigenmode(sname, scan_parameter_value=value, wavenumber=αm, wavenumber_error=Δα, \
                                    eigenvalue=σm, eigenvalue_error=Δσ, tolerance=e, \
                                    resistive_layer_thickness=δin, grid_scaling_factor=C, \
                                    resistive_layer_nodes=nin, current_sheet_nodes=nwa, \
                                    niter=nit, resolution=N, grid=z, **metadata, **s)

            except DeltaError as ex:
                status = False
                logging.info(f"Stable eigenmode: {ex}")

            except ValueError as ex:
                status = False
                logging.info(f"Wrong parameter: {ex}")

            except Exception as ex:
                status = False
                logging.warning(f"Could not find brackets for {dependence} = {value:+.3e}: {ex}")


    n = 0
    if counter is not None:
        with counter.get_lock():
            counter.value += 1
            n = counter.value

    progress   = n / ntasks
    percentage = int(progress * 100)

    fmt = r'[{:0' + str(len(str(ntasks))) + 'd}/' + str(ntasks) + ']'
    progress_line = f"Progress {percentage}% complete {'█' * (percentage // 2)}{' ' * (50 - (percentage // 2))} {fmt.format(n)}"

    if status:
        assert σm is not None
        assert δin is not None
        assert C is not None
        assert nin is not None
        assert nwa is not None
        assert N is not None
        σm_val = σm[0]
        Δσ_val = np.atleast_1d(Δσ)[0]
        result_line = info + f"α={αm:.4e}±{Δα:.1e}  σ={σm_val.real:.4e}±{Δσ_val:.1e}  δin={δin:.3e}  nin={nin}  nwa={nwa}  C={C:.3e}  N={N} after {nit} function calls" + ' '*6
        if verbose:
            logging.info(f"{result_line}")
        else:
            print(f"\r{bracket_line}\n{result_line}\n\n{progress_line}{UP}{UP}{UP}", end='', flush=True)

        return αm, σm, δin, N, status

    else:
        bracket_line = info + "could not find any bracket!" + ' '*80
        result_line  = info + "could not find any maximum!" + ' '*80
        if verbose:
            logging.info(f"{result_line}")
        else:
            print(f"\r{bracket_line}\n{result_line}\n\n{progress_line}{UP}{UP}{UP}", end='', flush=True)

        return None, None, None, None, status

def main() -> None:
    '''
        Given provided options, calculates eigenmodes of the equilibrium field with magnetic and velocity shear.
    '''
    params = build_params(parser_type='maximum')

    # Configure logging
    setup_logging(verbose=bool(params.verbose), log_file=params.log_file)

    dpath = build_dpath(params)

    vl, vu, dv = params.vmin, params.vmax, params.vinc
    assert vl is not None and vu is not None and dv is not None, "Sweep parameters (vmin, vmax, vinc) must be defined."
    n_points = int(np.ceil((vu - vl + 0.5 * dv) / dv))
    if n_points <= 0:
        logging.error(f"Error: Parameter range is empty or invalid (min={vl:.3e}, max={vu:.3e}, inc={dv:.3e}).")
        sys.exit(1)
    vs = np.linspace(vl, vu, n_points)
    if params.logarithmic:
        vs = 10**vs

    ntasks = vs.size
    nprocs = int(os.getenv('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    nprocs = min(ntasks, nprocs)

    params.data_path = dpath
    params.ntasks    = ntasks

    if not os.path.exists(dpath):
        os.makedirs(dpath)

    logging.info("\nCalculation of the maximum growth rate dependence on several parameters for the selected eigenmode.")
    logging.info("Use option '-h' to show all possible arguments.\n")

    print_info(params)

    if ntasks > 1 and os.getenv("OMP_NUM_THREADS") != "1":
        logging.warning("\n\033[1mPlease set OMP_NUM_THREADS=1 to ensure optimal performance!\033[0m")
        nprocs = 1

    k = refine_wavenumber_bracket(vs, params)
    g = refine_eigenvalues(vs, params)
    deltas = refine_resistive_scale(vs, params)

    plural = 'es' if nprocs > 1 else ''
    logging.info(f'\nCalculation initiated with {nprocs} process{plural}'
                 f' for {ntasks} values.\n')

    delta_time = -time.time()

    shared_counter = mp.Value('i', 0)

    try:
        if params.step:
            extrap_deg   = params.extrap_deg if params.extrap_deg is not None else 2
            extrap_guard = params.extrap_guard if params.extrap_guard is not None else 0.01

            k_extrap = Extrapolator(maxdeg=extrap_deg, ymin=1e-6)
            g_extrap = Extrapolator(maxdeg=extrap_deg, ymin=1e-10)

            global counter
            counter = shared_counter

            # Initial values from refinement if any
            gm = params.sigma
            δm = params.delta

            for n, v in enumerate(vs):
                # ── extrapolate wavenumber bracket ────────────────────────────────
                kn: Optional[List[float]] = None
                k_pred = k_extrap.predict(v)
                if k_pred is not None:
                    guard = extrap_guard * k_pred
                    kl_val = max(k_pred - guard, 1e-6)
                    ku_val = k_pred + guard
                    kn    = [kl_val, ku_val]
                    if params.verbose:
                        logging.info(f"  k extrapolated: {k_pred:.4e}  →  bracket [{kl_val:.4e}, {ku_val:.4e}]")
                else:
                    kn = k[n]

                # ── use previous step results as guesses if refinement is default ─
                gn = gm if g[n] == params.sigma else g[n]
                δn = δm if (deltas[n] == params.delta and δm is not None and δm > 0.0) else deltas[n]

                km, gm, δm, N, status = task(v, kn, gn, δn, params)

                if not status:
                    break
                if gm is not None and gm.real < 1e-6:
                    logging.info(f"Growth rate dropped below 1e-6 ({gm.real:.3e}). Stopping sweep.")
                    break

                # ── record for next extrapolation ─────────────────────────────────
                k_extrap.add(v, km)
                if gm is not None:
                    g_extrap.add(v, gm)
        else:
            with mp.Pool(
                processes=nprocs,
                initializer=init_worker,
                initargs=(shared_counter,)
            ) as pool:
                pool.starmap(task, [(v, k[n], g[n], deltas[n], params) for n, v in enumerate(vs)])
    except KeyboardInterrupt:
        logging.info("\n\nCalculation interrupted by user. Exiting cleanly...")
        sys.exit(1)

    delta_time += time.time()

    write_results(params, delta_time)

    if not params.verbose:
        sys.stdout.write("\n\n\n\n")
    logging.info(f"\nCalculation done in {delta_time:.2f} seconds.\n")


if __name__ == "__main__":
    main()
