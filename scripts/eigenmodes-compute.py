#!/usr/bin/env python3
#
import os, sys, time, logging, signal
from typing import Any, Dict, Optional
import multiprocessing as mp
import numpy as np

from tearing_eigenmodes import build_params, build_dpath, print_info, \
                             refine_eigenvalues, refine_inner_scale, \
                             eigenmodes, write_results, save_eigenmode, setup_logging, \
                             check_state, compile_metadata, SimulationParams
from tearing_eigenmodes.parser import resolve_gevp_method

counter: Any = None

def init_worker(shared_counter: Any) -> None:
    """Assign the shared object to the global variable in this worker."""
    # Worker processes should ignore SIGINT; only the main process will handle it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global counter
    counter = shared_counter

def task(k: float, sigma: Any, delta: Optional[float], params: SimulationParams) -> None:
    global counter

    import copy
    params_base = copy.copy(params)

    # Initialize logging in worker processes (start methods like forkserver/spawn do not inherit log handlers)
    setup_logging(verbose=params_base.verbose, log_file=params_base.log_file)

    ntasks   = params_base.ntasks
    verbose  = params_base.verbose
    force    = params_base.force
    Nmax     = params_base.Nmax
    w        = params_base.w if params_base.w is not None else 0.0
    a        = params_base.a if params_base.a is not None else 1.0

    status   = False

    α = k * a

    # Initialize variables to satisfy static analysis
    σ: Optional[np.ndarray] = None
    e: Optional[float] = None
    δin: Optional[float] = None
    nin: Optional[int] = None
    nwa: Optional[int] = None
    N: Optional[int] = None
    C: Optional[float] = None
    z: Optional[np.ndarray] = None
    s: Optional[Dict[str, np.ndarray]] = None
    sname = os.path.join(params_base.data_path if params_base.data_path is not None else './', f'state_α{α:.6e}.npz')

    # Set the per-task computation parameters before the reuse check so the
    # compared run-config metadata matches what a recomputation would save
    # (in particular the per-wavenumber inner-scale guess).
    params_base.sigma       = sigma
    params_base.inner_scale = delta
    params_base.delta       = delta
    params_base.alpha       = α

    status, state_data = check_state(sname, force=force, Nmax=Nmax, params=params_base)
    if status:
        assert state_data is not None
        α   = float(state_data['wavenumber'])
        σ   = state_data['eigenvalue']
        e   = float(state_data['tolerance'])
        if 'minimum_physical_scale' in state_data:
            δin = float(state_data['minimum_physical_scale'])
        elif 'resistive_layer_thickness' in state_data:
            δin = float(state_data['resistive_layer_thickness'])
        else:
            δin = float(state_data.get('inner_scale', np.nan))

        if 'minimum_scale_nodes' in state_data:
            nin = int(state_data['minimum_scale_nodes'])
        elif 'resistive_layer_nodes' in state_data:
            nin = int(state_data['resistive_layer_nodes'])
        else:
            nin = int(state_data.get('n_inner', 0))

        nwa = int(state_data['current_sheet_nodes'])
        N   = int(state_data['resolution'])
        C   = float(state_data['grid_scaling_factor'])

    if not status:
        try:
            σ, s, e, δin, nin, nwa, C, N, z, status = eigenmodes(params_base)

            if status:
                assert s is not None
                # Include physical and numerical parameters for reproducibility
                metadata = compile_metadata(params_base)

                save_eigenmode(sname, wavenumber=α, eigenvalue=σ, tolerance=e, \
                                    grid_scaling_factor=C, current_sheet_nodes=nwa, \
                                    resolution=N, grid=z, **metadata, **s)

        except Exception as ex:
            status = False
            logging.warning(f"Could not find eigenmode for α = {α:.3e}: {ex}")

    n = 0
    if counter is not None:
        with counter.get_lock():
            counter.value += 1
            n = counter.value

    fmt    = r'[{:' + str(len(str(ntasks))) + 'd}/' + str(ntasks) + ']'
    output = f"{fmt.format(n)}  α = {α:.3e}: "

    if status:
        assert σ is not None
        assert δin is not None
        assert nin is not None
        assert nwa is not None
        assert e is not None
        assert C is not None
        assert N is not None
        if verbose and state_data and 'mode_scales' in state_data:
            from tearing_eigenmodes.printing import format_mode_scale_summary
            summary = format_mode_scale_summary(state_data.get('mode_scales'))
            if summary:
                logging.info(summary)
        σ_arr = np.atleast_1d(σ)
        σ_val = σ_arr[0]
        e_arr = np.atleast_1d(e)
        e_val = e_arr[0]
        msg = (output \
             + f"{σ_arr.size:3d} eigenmode{'s' if σ_arr.size > 1 else ' '}," \
             + f" σ₀ = {σ_val.real:.3e}{σ_val.imag:+.4e}j" \
             + f" (δin = {δin:.3e}, nin = {nin}, nwa = {nwa}, tol = {e_val:.3e}, C = {C:.3e}, N = {N})" \
             + f" {'Did not converge!' if e_val > 1 else ''}")
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

def main() -> None:
    '''
        Given options, calculates the dispersion relation under the equilibrium field with magnetic and velocity shear.
    '''
    # Parse command‑line arguments
    params = build_params(parser_type='dispersion')

    # Configure logging
    setup_logging(verbose=params.verbose, log_file=params.log_file)

    # Build data path
    dpath = build_dpath(params)

    # Wavenumber array
    kl, ku, dk = params.kmin, params.kmax, params.kinc
    assert kl is not None and ku is not None and dk is not None, "Wavenumber range parameters (kmin, kmax, kinc) must be defined."
    n_points = int(np.ceil((ku - kl + 0.5 * dk) / dk))
    if n_points <= 0:
        logging.error(f"Error: Wavenumber range is empty or invalid (min={kl:.3e}, max={ku:.3e}, inc={dk:.3e}).")
        sys.exit(1)
    ks = np.linspace(kl, ku, n_points)
    if params.logarithmic:
        ks = 10**ks
    assert params.a is not None, "Scaling parameter a must be defined."
    ks /= params.a

    # Determine the number of independent tasks and CPU cores to use.
    ntasks = ks.size
    nprocs = int(os.getenv('SLURM_CPUS_PER_TASK', mp.cpu_count()))
    nprocs = min(ntasks, nprocs)

    params.data_path = dpath
    params.ntasks    = ntasks

    if not os.path.exists(dpath):
        os.makedirs(dpath)

    logging.info('\nCalculation of the \033[1meigenmodes\033[0m for an equlibrium with the velocity and magnetic field shear.')
    logging.info("Use option '-h' to show all possible arguments.\n")

    print_info(params)

    # A single wavenumber has the machine to itself, so it can use the
    # threaded shift-invert solver; several run in worker processes.
    params.gevp_method = resolve_gevp_method(params.gevp_method, ntasks == 1)
    if ntasks == 1 and params.gevp_method == 'shift-invert' and os.getenv("OMP_NUM_THREADS") == "1":
        logging.warning("\n\033[1mOMP_NUM_THREADS=1 limits the threaded shift-invert solver used for a single wavenumber to one core; unset it to use them all.\033[0m")

    if ntasks > 1 and os.getenv("OMP_NUM_THREADS") != "1":
        logging.warning("\n\033[1mPlease set OMP_NUM_THREADS=1 to ensure optimal performance!\033[0m")
        nprocs = 1

    plural = 'es' if nprocs > 1 else ''
    logging.info(f'\nCalculation initiated with {nprocs} process{plural}'
                 f' for {ntasks} values.\n')

    # Do calculations in parallel
    delta_time = -time.time()

    shared_counter = mp.Value('i', 0)

    sigmas = refine_eigenvalues(ks, params)
    deltas = refine_inner_scale(ks, params)

    try:
        with mp.Pool(
            processes=nprocs,
            initializer=init_worker,
            initargs=(shared_counter,)
        ) as pool:
            pool.starmap(task, [(k, sigmas[n], deltas[n], params) for n, k in enumerate(ks)])
    except KeyboardInterrupt:
        logging.info("\n\nCalculation interrupted by user. Exiting cleanly...")
        sys.exit(1)

    delta_time += time.time()

    write_results(params, delta_time)

    if not params.verbose:
        sys.stdout.write("\n")
    logging.info(f"\nCalculation done in {delta_time:.2f} seconds.\n")


if __name__ == "__main__":
    main()
