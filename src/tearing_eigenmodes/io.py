import os
import glob
import numpy as np
from scipy.interpolate import make_interp_spline
from typing import Dict, Any
from psecas import ChebyshevRationalGrid
from .grid import find_peak_location

def build_dpath(params: dict) -> str:
    """
    Construct the directory name that will hold the results for a given run.

    Parameters
    ----------
    params : dict
        Dictionary containing all relevant simulation parameters.  It must
        provide at least the following keys:

            * ``CGL``         - bool, whether CGL physics is used
            * ``S``, ``Pr``   - floats
            * ``β``, ``Δβ``   - floats (only needed if ``CGL``)
            * ``ξ``, ``ϵ``    - floats
            * ``a`` , ``w``   - floats
            * ``suffix``      - string to append at the end

    Returns
    -------
    str
        The full path that will be used to store (or read) the results.
    """
    # Pull the required values out of the dictionary
    a, w = params['a'], params['w']
    S, Pr = params['S'], params['Pr']
    ξ, ϵ = params['xi'], params['Hall']
    β, Δβ = params['plasma_beta'], params['plasma_beta_difference']

    dpath  = './RESULTS/'
    dpath += '' if S  is None else f'S{S:.3e}'
    dpath += '' if Pr is None else f'Pr{Pr:.3e}'
    if params['CGL']:
        dpath += '' if β  is None else f'β{β:.3e}'
        dpath += '' if Δβ is None else f'Δβ{Δβ:+.2f}'
    dpath += '' if ξ is None else f'ξ{ξ:.3e}'
    dpath += '' if ϵ is None else f'ϵ{ϵ:.3e}'
    dpath += '' if a is None else f'a{a:.3e}'
    dpath += '' if w is None else f'w{w:.3e}'

    # Build the path and append any user‑supplied suffix
    return dpath + params['suffix']


def load_config(file_path: str) -> dict:
    """
    Read a simple key-value configuration file.
    Supports 'key = value' or 'key : value' formats.
    Lines starting with '#' are ignored.

    Parameters
    ----------
    file_path : str
        Path to the configuration file.

    Returns
    -------
    dict
        Dictionary of key-value pairs as strings.
    """
    config = {}
    if not os.path.exists(file_path):
        return config

    with open(file_path, 'r') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            if '=' in line:
                key, val = line.split('=', 1)
            elif ':' in line:
                key, val = line.split(':', 1)
            else:
                continue
            config[key.strip()] = val.strip()
    return config


def load_eigenmodes(path: str, pattern: str = "*.npz", recalculate_thickness=False):
    """
    Load eigenmode data from .npz files using glob and os instead of pathlib.
    """
    # Join the directory path and the search pattern
    search_path = os.path.join(path, pattern)

    # glob.glob returns a list of path strings matching the pattern
    files = sorted(glob.glob(search_path))

    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found in {path}")

    rows = []
    for f in files:
        with np.load(f) as state:
            if 'growth_rate' in state:
                growth = state['growth_rate']
            else:
                growth = state['eigenvalues'][0]
            if not 'value' in state:
                state['value'] = state['wavenumber']
            if 'l_inner' in state:
                lin = state['l_inner']
                nin = state['n_inner']
            else:
                N = state['resolution']
                C = state['scaling_factor']
                grid = ChebyshevRationalGrid(N, C=C)
                u = state['duz']
                b = state['dbz']
                lin, nin = find_peak_location(u, b, grid)
            if recalculate_thickness:
                dlt = state['inner_scale']
            else:
                dlt = state['inner_scale']

            rows.append([
                state['value'],
                state['wavenumber'],
                growth,
                state['tolerance'],
                dlt,
                lin,
                state['scaling_factor'],
                nin,
                state['resolution']
            ])

    # Sort the list of rows (default sort is lexicographical)
    rows.sort()

    # Unpack into individual numpy arrays
    v = np.array([x[0] for x in rows])
    α = np.array([x[1] for x in rows])
    σ = np.array([x[2] for x in rows])
    e = np.array([x[3] for x in rows])
    δ = np.array([x[4] for x in rows])
    l = np.array([x[5] for x in rows])
    c = np.array([x[6] for x in rows])
    n = np.array([x[7] for x in rows])
    N = np.array([x[8] for x in rows])

    return v, α, σ, e, δ, l, c, n, N


def load_eig_scales(path: str, pattern: str = "*.npz"):
    """
    Load eigenmode data from .npz files using glob and os instead of pathlib.
    """
    # Join the directory path and the search pattern
    search_path = os.path.join(path, pattern)

    # glob.glob returns a list of path strings matching the pattern
    files = sorted(glob.glob(search_path))

    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found in {path}")

    rows = []
    for f in files:
        with np.load(f) as state:
            if 'growth_rate' in state:
                growth = state['growth_rate']
            else:
                growth = state['eigenvalues'][0]
            if not 'value' in state:
                state['value'] = state['wavenumber']

            if 'l_inner' in state:
                lin = state['l_inner']
                nin = state['n_inner']
            else:
                N = state['resolution']
                C = state['scaling_factor']
                u = state['duz']
                b = state['dbz']
                grid = ChebyshevRationalGrid(N, C=C)
                lin, _ = find_peak_location(u, b, grid)

            rows.append([
                state['value'],
                lin,
            ])

    # Sort the list of rows (default sort is lexicographical)
    rows.sort()

    # Unpack into individual numpy arrays
    v = np.array([x[0] for x in rows])
    l = np.array([x[1] for x in rows])

    return v, l


def refine_inner_scale(vs, params):

    linner = [ params['l_inner'] ]*vs.size

    """Update thickness using cached eigenmodes if available."""
    try:
        v, l = load_eig_scales(params['data_path'])
    except FileNotFoundError:
        return linner

    if v.size < 2:
        return linner

    degree = min(3, v.size - 1)
    # if params['logarithmic']:
    #     spline = make_interp_spline(np.log10(v), l, k=degree)
    #     linner = spline(np.log10(vs))
    #     if linner.min() <= 0.0:
    #         spline = make_interp_spline(np.log(v), l, k=0)
    #         linner = spline(np.log10(vs))
    # else:
    #     spline = make_interp_spline(v, l, k=degree)
    #     linner = spline(vs)
    #     if linner.min() <= 0.0:
    #         spline = make_interp_spline(v, l, k=0)
    #         linner = spline(vs)

    spline = make_interp_spline(v, l, k=degree)
    linner = spline(vs)
    if linner.min() <= 0.0:
        spline = make_interp_spline(v, l, k=0)
        linner = spline(vs)

    vmn, vmx = v.min(), v.max()

    linner = linner.tolist()
    for i, x in enumerate(vs):
        if x < vmn or x > vmx:
            linner[i] = params['l_inner']

    return linner


def refine_growth_rate(vs, params):

    sigma = [ params['sigma'] ]*vs.size

    """Update thickness using cached eigenmodes if available."""
    try:
        v, _, σ, _, _, _, _, _, _ = load_eigenmodes(params['data_path'])
    except FileNotFoundError:
        return sigma

    if v.size < 2:
        return sigma

    degree = min(3, v.size - 1)
    # if params['logarithmic']:
    #     spline = make_interp_spline(np.log10(v), σ.real, k=degree)
    #     sigma = spline(np.log10(vs))
    #     if sigma.min() <= 0.0:
    #         spline = make_interp_spline(np.log(v), σ.real, k=0)
    #         sigma = spline(np.log10(vs))
    # else:
    #     spline = make_interp_spline(v, σ.real, k=degree)
    #     sigma = spline(vs)
    #     if sigma.min() <= 0.0:
    #         spline = make_interp_spline(v, σ.real, k=0)
    #         sigma = spline(vs)

    spline = make_interp_spline(v, σ, k=degree)
    sigma = spline(vs)
    if sigma.min() <= 0.0:
        spline = make_interp_spline(v, σ, k=0)
        sigma = spline(vs)

    vmn, vmx = v.min(), v.max()

    sigma = sigma.tolist()
    for i, x in enumerate(vs):
        if x < vmn or x > vmx:
            sigma[i] = params['sigma']

    return sigma


def refine_thickness(vs, params):

    δinner = [ params['delta'] ]*vs.size

    """Update thickness using cached eigenmodes if available."""
    try:
        v, _, _, _, δ, _, _, _, _ = load_eigenmodes(params['data_path'])
    except FileNotFoundError:
        return δinner

    if v.size < 2:
        return δinner

    degree = min(3, v.size - 1)
    # if params['logarithmic']:
    #     spline = make_interp_spline(np.log10(v), δ, k=degree)
    #     δinner = spline(np.log10(vs))
    #     if δinner.min() <= 0.0:
    #         spline = make_interp_spline(np.log(v), δ, k=0)
    #         δinner = spline(np.log10(vs))
    # else:
    #     spline = make_interp_spline(v, δ, k=degree)
    #     δinner = spline(vs)
    #     if δinner.min() <= 0.0:
    #         spline = make_interp_spline(v, δ, k=0)
    #         δinner = spline(vs)
    spline = make_interp_spline(v, δ, k=degree)
    δinner = spline(vs)
    if δinner.min() <= 0.0:
        spline = make_interp_spline(v, δ, k=0)
        δinner = spline(vs)

    vmn, vmx = v.min(), v.max()

    δinner = δinner.tolist()
    for i, x in enumerate(vs):
        if x < vmn or x > vmx:
            δinner[i] = params['delta']

    return δinner


def refine_wavenumber_bracket(vs, params):
    """Update wavenumber brackets using cached eigenmodes if available."""
    import numpy as np

    # Initialize with default bracket from params
    kbracket = [params.get('kbracket')] * vs.size

    try:
        # Assuming load_eigenmodes returns arrays
        v, α, *_ = load_eigenmodes(params['data_path'])
    except (FileNotFoundError, KeyError, TypeError):
        return kbracket

    if v.size < 2:
        return kbracket

    print("\nImproved wavenumber bounds:")

    vmn, vmx = v.min(), v.max()
    # Define a small tolerance based on the data scale
    # 1e-12 is generally safe for double precision
    atol = 1e-12 * max(abs(vmn), abs(vmx))

    def _alpha_at_last_leq(x):
        # Using searchsorted with a small epsilon to handle precision
        idx = np.searchsorted(v, x + atol, side="right") - 1
        return α[idx] if idx >= 0 else None

    def _alpha_at_first_geq(x):
        idx = np.searchsorted(v, x - atol, side="left")
        return α[idx] if idx < v.size else None

    for n, x in enumerate(vs):
        # Default brackets from params
        kl, ku = params.get('kbracket') or (None, None)

        # Robust boundary check: is x within range OR very close to boundaries
        within_bounds = (vmn <= x <= vmx) or np.isclose(x, vmn, atol=atol) or np.isclose(x, vmx, atol=atol)

        if within_bounds:
            αl = _alpha_at_last_leq(x)
            αu = _alpha_at_first_geq(x)

            if αl is not None and αu is not None:
                # Tighten the user-provided bracket with known cached values
                inner_l = min(αl, αu)
                inner_u = max(αl, αu)

                kl = inner_l if kl is None else max(kl, inner_l)
                ku = inner_u if ku is None else min(ku, inner_u)

        if kl is None or ku is None:
            kbracket[n] = None
            continue

        # If bracket collapsed to a single point, expand it slightly based on tolerance
        # to give the root-finder/optimizer room to breathe.
        todo = True
        if np.isclose(kl, ku, atol=atol):
            tol_factor = 5.0 * params.get('ktol', 1e-3)
            kl *= (1.0 - tol_factor)
            ku *= (1.0 + tol_factor)
            todo = False

        kbracket[n] = [float(kl), float(ku)]

        # Verbose output for updated brackets
        if todo or params.get('verbose') or params.get('force'):
             print(
                f"\t{params.get('dependence', 'v')} = {x:+.3e}: "
                f"α-bracket = [{kbracket[n][0]:.4e}, {kbracket[n][1]:.4e}]"
            )

    return kbracket


def refine_wavenumber_brackets_thickness(dpath, Rs, αbracket, δinner, args, log=False):
    """Update wavenumber brackets using cached eigenmodes if available."""
    import numpy as np
    from scipy.interpolate import interp1d

    try:
        v, α, _, _, δ, _, *_ = load_eigenmodes(dpath)
    except FileNotFoundError:
        return αbracket, δinner

    if v.size < 2:
        return αbracket, δinner

    print("\nImproved wavenumber bounds:")

    def _alpha_at_last_leq(x):
        idx = np.searchsorted(v, x, side="right") - 1
        return α[idx] if idx >= 0 else None

    def _alpha_at_first_geq(x):
        idx = np.searchsorted(v, x, side="left")
        return α[idx] if idx < v.size else None

    # Determine interpolation kind based on points
    if v.size >= 4:
        kind = 'cubic'
    elif v.size == 3:
        kind = 'quadratic'
    else:
        kind = 'linear'

    # Configure x-axis based on log setting
    I = np.where(v > 0.0)
    x_input = np.log10(v[I])  if log else v
    x_eval  = np.log10(Rs) if log else Rs

    # Initialize interpolator
    # fill_value=(δ[0], δ[-1]) ensures the last available values are used outside the range
    f = interp1d(x_input, δ, kind=kind, bounds_error=False, fill_value=(δ[0], δ[-1]))
    δinner = f(x_eval)

    # Fallback to nearest-neighbor (k=0) if negative values are encountered
    if δinner.min() <= 1.0e-4:
        f_fallback = interp1d(x_input, δ, kind='nearest', bounds_error=False, fill_value=(δ[0], δ[-1]))
        δinner = f_fallback(x_eval)

    vmn, vmx = v.min(), v.max()

    for n, x in enumerate(Rs):
        kl, ku = args.wavenumber_bracket or (None, None)

        if vmn <= x <= vmx: # we can interpolate or determine bracket between already known points
            αl = _alpha_at_last_leq(x)
            αu = _alpha_at_first_geq(x)

            if αl is not None and αu is not None:
                if kl is None:
                   kl = min(αl, αu)
                else:
                   kl = max(kl, min(αl, αu))
                if ku is None:
                   ku = max(αl, αu)
                else:
                   ku = min(ku, max(αl, αu))

        if kl is None or ku is None:
            αbracket[n] = None
            continue

        todo = not np.isclose(kl, ku)
        if not todo:
            kl *= max(0.5, 1.0 - 5.0 * args.wavenumber_tolerance)
            ku *= min(1.5, 1.0 + 5.0 * args.wavenumber_tolerance)

        αbracket[n] = [float(kl), float(ku)]
        if todo or args.force:
            print(
                f"\t{args.dependence} = {x:+.3e}: "
                f"α = {αbracket[n][0]:.4e} ... {αbracket[n][1]:.4e}"
            )

    return αbracket, δinner


def write_results(params: Dict[str, Any], delta_time: float) -> None:
    """
    Unified result writer for both dispersion and maxima/dependence runs.
    """
    dpath = params['data_path']
    if not dpath or not os.path.exists(dpath):
        raise FileNotFoundError(f"Data path {dpath!r} does not exist")

    fname = f"{dpath}.dat"
    # Load the eigenmodes (assuming this returns v for dependence or just arrays)
    # v is typically the independent variable (k for dispersion, or the 'dependence' var)
    v, α, σ, e, δ, l, c, n, N = load_eigenmodes(dpath)

    # Check if we are in dependence mode
    dep_key = params.get('dependence')

    with open(fname, 'w') as io:
        # --- Header Section ---
        io.write(f"#\n# Tearing Instability - mode {params['mode']}\n#\n")

        io.write("# Plasma parameters:\n")
        eqns = "Gyrotropic" if params.get('CGL') else "Classical"
        io.write(f"#   Equations                            =   {eqns} MHD\n")
        if params.get('CGL'):
            io.write(f"#   Equation of State                    =   {params.get('eos')}\n")

        # Helper to write header lines only if the parameter isn't the one being varied
        def write_head(label, key, fmt="10.3e"):
            val = params.get(key)
            if val is not None:
                io.write(f"#   {label:<36} =  {val:{fmt}}\n")

        write_head("Lundquist number (S)", 'S')
        write_head("Prandtl number (Pr)", 'Pr')

        if params.get('CGL'):
            write_head("Plasma-β (β)", 'plasma_beta')
            write_head("Plasma-β difference (Δβ)", 'plasma_beta_difference')

        write_head("Magnetic transverse field (ξ)", 'xi')
        write_head("Hall current term strength (ϵ)", 'Hall')

        if params.get('CGL'):
            write_head("  Parallel adiabatic index (γpar)", 'parallel_index')
            write_head("  Perpendicular adiabatic index (γper)", 'perpendicular_index')

        io.write("# Equilibrium parameters:\n")
        write_head("Current sheet thickness (a)", 'a')
        write_head("Current sheet half-width (w)", 'w')

        io.write("# Geometry/convergence parameters:\n")
        io.write(f"#   {'Resolution range (N)':<36} =  [{params['Nmin']}, {params['Nmax']}] increment {params['Ninc']}\n")
        io.write(f"#   {'Growth rate range':<36} =  [{params['sigma_lower']}, {params['sigma_upper']}]\n")

        if params.get('sigma_imag') is not None:
            io.write(f"#   {'Imaginary amplitude limit':<36} =  {params['sigma_imag']:10.3e}\n")

        write_head("Growth rate absolute tolerance", 'atol')
        write_head("Growth rate relative tolerance", 'rtol')
        write_head("Growth rate guess tolerance", 'gtol')

        if 'ktol' in params:
            write_head("Wavenumber relative tolerance", 'ktol')

        io.write(f"#   {'Selection order':<36} =   {params['orderby']}\n")
        io.write(f"#   {'Converge the mode':<36} =   {params['mode']}\n")
        write_head("Inner-layer thickness tolerance", 'dtol')
        io.write(f"#   {'Number of inner collocation points':<36} =   {params['n_inner']}\n")
        io.write(f"#   {'Amplitude fraction at zmax':<36} =   {params['f_outer']}\n")
        io.write(f"#   {'Decay e-folds at zmax':<36} =  {params['decay_efolds']:10.3e}\n")

        # --- Wall-clock time ---
        io.write(f"#\n# Calculation done in {delta_time:.2f} seconds.\n#\n")

        # --- Table Header ---
        if dep_key:
            # Table for Maxima/Dependence Study
            io.write(f"#    {dep_key:<2s}               α_max            Re(σ_max)        Im(σ_max)        δ_in             λ_eig            tolerance        C              n_in    N\n")
            io.write("#" + " --------------- "*8 + " ------  ------" + "\n")
        else:
            # Table for standard Dispersion Run
            io.write("#    α                Re(σ)            Im(σ)            δ_in             λ_eig            tolerance        C              n_in     N\n")
            io.write("#" + " --------------- "*7 + " ------  ------" + "\n")

        # --- Data Rows ---
        for i in range(len(α)):
            if dep_key:
                # Dependence format: v is the varied parameter, alpha is alpha_max
                io.write(f"  {v[i]:15.8e}  {α[i]:15.8e}  {σ[i].real:15.8e}  {σ[i].imag:15.8e}  {δ[i]:15.8e}  {l[i]:15.8e}  {e[i]:15.8e}  {c[i]:15.8e}  {n[i]:>6d}  {N[i]:>6d}\n")
            else:
                # Dispersion format: alpha is the independent variable
                io.write(f"  {α[i]:15.8e}  {σ[i].real:15.8e}  {σ[i].imag:15.8e}  {δ[i]:15.8e}  {l[i]:15.8e}  {e[i]:15.8e}  {c[i]:15.8e}  {n[i]:>6d}  {N[i]:>6d}\n")

        io.flush()
