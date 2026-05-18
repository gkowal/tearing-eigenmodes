#!/usr/bin/env python3
#
import os, sys, time, logging, signal
import multiprocessing as mp
import numpy as np

from collections import deque
from functools import lru_cache
from tearing_eigenmodes import build_params, build_dpath, \
                             print_info, refine_wavenumber_bracket, \
                             refine_thickness, refine_growth_rate, \
                             eigenmodes, write_results, DeltaError, \
                             estimate_max, save_eigenmode, setup_logging

counter = None

class Extrapolator:
    """
    Tracks a history of (x, y) pairs and extrapolates y at a new x
    using polynomial fitting (degree 1=linear, 2=quadratic, 3=cubic).
    Falls back gracefully when insufficient history is available.
    Direction-agnostic: works for both increasing and decreasing x sweeps.
    """
    def __init__(self, maxdeg=2, minpoints=2, maxhistory=6, ymin=None):
        self.maxdeg    = maxdeg
        self.minpoints = minpoints
        self.ymin      = ymin        # optional lower clamp on predicted value
        self.xs        = deque(maxlen=maxhistory)
        self.ys        = deque(maxlen=maxhistory)

    def add(self, x, y):
        self.xs.append(float(x))
        self.ys.append(y)

    def predict(self, x_new):
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

    def __len__(self):
        return len(self.xs)

def init_worker(shared_counter):
    """Assign the shared object to the global variable in this worker."""
    # Worker processes should ignore SIGINT; only the main process will handle it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global counter
    counter = shared_counter

def make_objective(params_base):
    # capture a shallow copy once; treat as immutable thereafter
    params_fixed = dict(params_base)

    @lru_cache(maxsize=1024)
    def objective(αq: float) -> float:
        if αq <= 0:
            return 0.0
        p = dict(params_fixed)
        p["alpha"] = αq
        σ, _, _, _, _, _, _, _, _, status = eigenmodes(p)
        return float(-σ.real) if status else 0.0

    def f(α: float) -> float:
        return objective(round(float(α), 12))

    return f

def task(value, αbracket, sigma, δinner, params):
    from scipy.optimize import bracket, minimize_scalar

    global counter

    status = False

    params_base = dict(params)

    ntasks     = params_base.get('ntasks'  , 1)
    dependence = params_base.get('dependence'  , 'S')
    verbose    = params_base.get('verbose', False)
    force      = params_base.get('force'  , False)
    Nmax       = params_base.get('Nmax'   , 2048)
    rtol       = params_base.get('rtol'   , 1e-5)
    wtol       = params_base.get('wtol'   , 1e-3)
    w          = params_base.get('w'       , 0.0)
    a          = params_base.get('a'       , 1.0)

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
    params_base[dep_map[dependence]] = float(value)

    sname = os.path.join(params_base.get('data_path', './'), f'state_{dependence}{value:+.6e}.npz')

    if os.path.exists(sname):
        with np.load(sname) as state:

            αm  = state['wavenumber']
            σm  = state['growth_rate']
            e   = state['tolerance']
            δin = state['inner_scale']
            nin = state['n_inner']
            if 'n_wa' in state:
                nwa = state['n_wa']
            else:
                z = state['grid']
                I = np.where(np.abs(z) <= (w + a))
                nwa = I[0].size
            N   = state['resolution']
            C   = state['scaling_factor']
            if 'niter' in state.keys():
                nit = state['niter']
            else:
                nit = 1
            if 'wavenumber_error' in state.keys():
                Δα = state['wavenumber_error']
            else:
                Δα = wtol * αm
            if 'eigenvalue_error' in state.keys():
                Δσ = state['eigenvalue_error']
            else:
                Δσ = rtol * σm.real * e

            status = not force and not (e > 1.0 and N < Nmax)

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
                params_base['sigma']   = sigma
                params_base['delta']   = δinner

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
                params_final = dict(params_base)
                params_final["alpha"] = αm
                σ, s, e, δin, nin, nwa, C, N, z, status = eigenmodes(params_final)
                if status:
                    σm  = σ
                    Δσ  = rtol * σm.real * e
                    nit = res.nfev

                    # Include physical and numerical parameters for reproducibility
                    metadata = {
                        'dependence': dependence,
                        'S': params_final.get('S'),
                        'Pr': params_final.get('Pr'),

                        'plasma_beta': params_final.get('plasma_beta'),
                        'plasma_beta_difference': params_final.get('plasma_beta_difference'),
                        'xi': params_final.get('xi'),
                        'Hall': params_final.get('Hall'),
                        'a': params_final.get('a'),
                        'w': params_final.get('w'),
                        'parallel_index': params_final.get('parallel_index'),
                        'perpendicular_index': params_final.get('perpendicular_index'),
                        'eos': params_final.get('eos'),
                        'CGL': params_final.get('CGL'),
                        'noshear': params_final.get('noshear'),
                        'Nmin': params_final.get('Nmin'),
                        'Nmax': params_final.get('Nmax'),
                        'Ninc': params_final.get('Ninc'),
                        'atol': params_final.get('atol'),
                        'rtol': params_final.get('rtol'),
                        'gtol': params_final.get('gtol'),
                        'dtol': params_final.get('dtol'),
                        'n_inner_req': params_final.get('n_inner'),
                        'f_outer': params_final.get('f_outer'),
                        'mode': params_final.get('mode'),
                    }

                    save_eigenmode(sname, value=value, wavenumber=αm, wavenumber_error=Δα, \
                                    growth_rate=σm, growth_rate_error=Δσ, tolerance=e, \
                                    inner_scale=δin, scaling_factor=C, n_inner=nin, n_wa=nwa, \
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
    with counter.get_lock():
        counter.value += 1
        n = counter.value

    progress   = n / ntasks
    percentage = int(progress * 100)

    fmt = r'[{:0' + str(len(str(ntasks))) + 'd}/' + str(ntasks) + ']'
    progress_line = f"Progress {percentage}% complete {'█' * (percentage // 2)}{' ' * (50 - (percentage // 2))} {fmt.format(n)}"

    if status:
        result_line = info + f"α={αm:.4e}±{Δα:.1e}  σ={σm.real:.4e}±{Δσ:.1e}  δin={δin:.3e}  nin={nin}  nwa={nwa}  C={C:.3e}  N={N} after {nit} function calls" + ' '*6
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

def main():
    '''
        Given provided options, calculates eigenmodes of the equilibrium field with magnetic and velocity shear.
    '''
    params = build_params(parser_type='maximum')

    # Configure logging
    setup_logging(verbose=params.get('verbose'), log_file=params.get('log_file'))

    dpath = build_dpath(params)

    vl, vu, dv = params['vmin'], params['vmax'], params['vinc']
    n_points = int(np.ceil((vu - vl + 0.5 * dv) / dv))
    if n_points <= 0:
        logging.error(f"Error: Parameter range is empty or invalid (min={vl:.3e}, max={vu:.3e}, inc={dv:.3e}).")
        sys.exit(1)
    vs = np.linspace(vl, vu, n_points)
    if params['logarithmic']:
        vs = 10**vs

    ntasks = vs.size
    nprocs = int(os.getenv('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    nprocs = min(ntasks, nprocs)

    params['data_path'] = dpath
    params['ntasks']    = ntasks

    if not os.path.exists(dpath):
        os.makedirs(dpath)

    logging.info("\nCalculation of the maximum growth rate dependence on several parameters for the selected eigenmode.")
    logging.info("Use option '-h' to show all possible arguments.\n")

    print_info(params)

    if ntasks > 1 and os.getenv("OMP_NUM_THREADS") != "1":
        logging.warning("\n\033[1mPlease set OMP_NUM_THREADS=1 to ensure optimal performance!\033[0m")
        nprocs = 1

    k = refine_wavenumber_bracket(vs, params)
    d = refine_thickness(vs, params)
    g = refine_growth_rate(vs, params)

    plural = 'es' if nprocs > 1 else ''
    logging.info(f'\nCalculation initiated with {nprocs} process{plural}'
                 f' for {ntasks} values.\n')

    delta_time = -time.time()

    shared_counter = mp.Value('i', 0)

    try:
        if params.get('step'):
            extrap_deg   = params.get('extrap_deg',   2)
            extrap_guard = params.get('extrap_guard', 0.01)

            k_extrap = Extrapolator(maxdeg=extrap_deg, ymin=1e-6)
            d_extrap = Extrapolator(maxdeg=extrap_deg, ymin=1e-10)
            g_extrap = Extrapolator(maxdeg=extrap_deg, ymin=1e-10)

            global counter
            counter = shared_counter

            # Initial values from refinement if any
            gm, dm = params.get('sigma'), params.get('delta')

            for n, v in enumerate(vs):
                # ── extrapolate wavenumber bracket ────────────────────────────────
                k_pred = k_extrap.predict(v)
                if k_pred is not None:
                    guard = extrap_guard * k_pred
                    kl    = max(k_pred - guard, 1e-6)
                    ku    = k_pred + guard
                    kn    = [kl, ku]
                    if params.get('verbose'):
                        logging.info(f"  k extrapolated: {k_pred:.4e}  →  bracket [{kl:.4e}, {ku:.4e}]")
                else:
                    kn = k[n]

                # ── use previous step results as guesses if refinement is default ─
                gn = gm if g[n] == params.get('sigma') else g[n]
                dn = dm if d[n] == params.get('delta') else d[n]

                km, gm, dm, N, status = task(v, kn, gn, dn, params)

                if not status:
                    break
                if gm.real < 1e-6:
                    logging.info(f"Growth rate dropped below 1e-6 ({gm.real:.3e}). Stopping sweep.")
                    break

                # ── record for next extrapolation ─────────────────────────────────
                k_extrap.add(v, km)
                if dm is not None:
                    d_extrap.add(v, dm)
                if gm is not None:
                    g_extrap.add(v, gm)
        else:
            with mp.Pool(
                processes=nprocs,
                initializer=init_worker,
                initargs=(shared_counter,)
            ) as pool:
                pool.starmap(task, [(v, k[n], g[n], d[n], params) for n, v in enumerate(vs)])
    except KeyboardInterrupt:
        logging.info("\n\nCalculation interrupted by user. Exiting cleanly...")
        sys.exit(1)

    delta_time += time.time()

    write_results(params, delta_time)

    if not params['verbose']:
        sys.stdout.write("\n\n\n\n")
    logging.info(f"\nCalculation done in {delta_time:.2f} seconds.\n")


if __name__ == "__main__":
    main()
