#!/usr/bin/env python3
#
import os, sys, time, logging
import multiprocessing as mp
import numpy as np

from tearing_eigenmodes import build_params, build_dpath, print_info, \
                             refine_growth_rate, refine_thickness, \
                             eigenmodes, write_results, save_eigenmode

counter = None

class SmartStreamHandler(logging.StreamHandler):
    """A logging handler that doesn't add a newline if the message starts with \r."""
    def emit(self, record):
        try:
            msg = self.format(record)
            if msg.startswith('\r'):
                self.terminator = ''
            else:
                self.terminator = '\n'
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

def init_worker(shared_counter):
    """Assign the shared object to the global variable in this worker."""
    global counter
    counter = shared_counter

def task(k, sigma, δinner, params):
    global counter

    params_base = dict(params)

    ntasks   = params_base.get('ntasks'  , 1)
    verbose  = params_base.get('verbose' , False)
    force    = params_base.get('force'   , False)
    Nmax     = params_base.get('Nmax'    , 2048)
    w        = params_base.get('w'       , 0.0)
    a        = params_base.get('a'       , 1.0)

    status   = False

    α = k * a

    sname  = os.path.join(params_base.get('data_path', './'), f'state_α{α:.6e}.npz')

    if os.path.exists(sname):
        with np.load(sname) as state:
            α   = state['wavenumber']
            σ   = state['growth_rate']
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

        status = not force and not (e > 1.0 and N < Nmax)

    if not status:
        try:
            params_base['sigma']   = sigma
            params_base['alpha']   = α
            params_base['delta']   = δinner

            σ, s, e, δin, nin, nwa, C, N, z, status = eigenmodes(params_base)

            if status:
                save_eigenmode(sname, value=α, wavenumber=α, growth_rate=σ, tolerance=e, \
                                    inner_scale=δin, scaling_factor=C, n_inner=nin, n_wa=nwa, \
                                    resolution=N, grid=z, **s)

        except Exception as ex:
            status = False
            logging.warning(f"Could not find eigenmode for α = {α:.3e}: {ex}")

    with counter.get_lock():
        counter.value += 1
        n = counter.value

    fmt    = r'[{:' + str(len(str(ntasks))) + 'd}/' + str(ntasks) + ']'
    output = f"{fmt.format(n)}  α = {α:.3e}: "

    if status:
        msg = (output \
             + f"{σ.size:3d} eigenmode{'s' if σ.size > 1 else ' '}," \
             + f" σ₀ = {σ.real:.3e}{σ.imag:+.4e}j" \
             + f" (δin = {δin:.3e}, nin = {nin}, nwa = {nwa}, tol = {e:.3e}, C = {C:.3e}, N = {N})" \
            + f" {'Did not converge!' if e > 1 else ''}")
        if verbose:
            logging.info(msg)
        else:
            logging.info('\r{:<150s}'.format(msg))
    else:
        msg = output + '   NO eigenmodes!'
        if verbose:
            logging.info(msg)
        else:
            logging.info('\r{:<150s}'.format(msg))

def main():
    '''
        Given options, calculates the dispersion relation under the equilibrium field with magnetic and velocity shear.
    '''
    # Parse command‑line arguments
    params = build_params(parser_type='dispersion')

    # Configure logging with custom SmartStreamHandler
    log_level = logging.DEBUG if params.get('verbose') else logging.INFO
    handler = SmartStreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logging.basicConfig(level=log_level, handlers=[handler])

    # Build data path
    dpath = build_dpath(params)

    # Wavenumber array
    kl, ku, dk = params['kmin'], params['kmax'], params['kinc']
    ks = np.linspace(kl, ku, int(np.ceil((ku - kl + 0.5 * dk) / dk)))
    if params['logarithmic']:
        ks = 10**ks
    ks /= params['a']

    # Determine the number of independent tasks and CPU cores to use.
    ntasks = ks.size
    nprocs = int(os.getenv('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    nprocs = min(ntasks, nprocs)

    params['data_path'] = dpath
    params['ntasks']    = ntasks

    if not os.path.exists(dpath):
        os.makedirs(dpath)

    logging.info('\nCalculation of the \033[1meigenmodes\033[0m for an equlibrium with the velocity and magnetic field shear.')
    logging.info("Use option '-h' to show all possible arguments.\n")

    print_info(params)

    if ntasks > 1 and os.getenv("OMP_NUM_THREADS") != "1":
        logging.warning("\n\033[1mPlease set OMP_NUM_THREADS=1 to ensure optimal performance!\033[0m")
        nprocs = 1

    plural = 'es' if nprocs > 1 else ''
    logging.info(f'\nCalculation initiated with {nprocs} process{plural}'
                 f' for {ntasks} values.\n')

    # Do calculations in parallel
    delta_time = -time.time()

    shared_counter = mp.Value('i', 0)

    s = refine_growth_rate(ks, params)
    d = refine_thickness(ks, params)

    with mp.Pool(
        processes=nprocs,
        initializer=init_worker,
        initargs=(shared_counter,)
    ) as pool:
        pool.starmap(task, [(k, s[n], d[n], params) for n, k in enumerate(ks)])

    delta_time += time.time()

    write_results(params, delta_time)

    if not params['verbose']:
        sys.stdout.write("\n")
    logging.info(f"\nCalculation done in {delta_time:.2f} seconds.\n")


if __name__ == "__main__":
    main()
