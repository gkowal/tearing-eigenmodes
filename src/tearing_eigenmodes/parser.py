from .exceptions import ParameterError
from .physics import eos_indices
from .io import load_config
from .params import SimulationParams
from typing import Dict, Any, Tuple, Optional, List

import argparse
import os
import numpy as np


def parser_setup(description: str = "Computes the tearing-instability growth rates for a given set of parameters.") -> argparse.ArgumentParser:
    """
    Build an ArgumentParser that contains every option used in both
    eigenmodes-compute.py and eigenmodes-maximum.py.

    The parser can be reused; after parsing you will get attributes for
    all arguments, even those that are irrelevant to the particular
    script.  The caller is responsible for ignoring the unused ones.
    """
    parser = argparse.ArgumentParser(description=description)

    # ------------------------------------------------------------------
    #   COMMON OPTIONS (used by *both* scripts)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--CGL", "-CGL",
        action='store_true',
        default=False,
        help="Gyrotropic MHD"
    )

    parser.add_argument(
        "--Lundquist-number", "-S",
        type=float,
        default=1e4,
        help="the Lundquist number"
    )

    parser.add_argument(
        "--Prandtl-number", "-Pr",
        type=float,
        default=0,
        help="the Prandtl number"
    )

    parser.add_argument(
        "--plasma-beta", "-β",
        type=float,
        default=0,
        help="the perpendicular plasma-β"
    )

    parser.add_argument(
        '--eos',
        choices=['adiabatic', 'polytropic', 'isothermal', 'custom'],
        default='adiabatic',
        help=("Equation of state type. If 'custom' is selected, "
              "'--gamma-parallel' and '--gamma-perpendicular' are required.")
    )

    parser.add_argument(
        "--gamma-parallel", "-ɣpar",
        type=float,
        default=3,
        help="the parallel adiabatic index"
    )

    parser.add_argument(
        "--gamma-perpendicular", "-ɣper",
        type=float,
        default=2,
        help="the perpendicular adiabatic index"
    )

    parser.add_argument(
        "--plasma-beta-difference", "-Δβ",
        type=float,
        default=0,
        help="the parallel to perpendicular plasma-β difference"
    )

    parser.add_argument(
        "--magnetic-transverse-field", "-ξ",
        type=float,
        default=0,
        help="the transverse magnetic strength"
    )

    # Hall option – different flag names in the two scripts
    parser.add_argument(
        "--Hall-parameter", "-ϵ",
        type=float,
        default=0,
        help="the Hall current term strength"
    )

    parser.add_argument(
        "--thickness", "-a",
        type=float,
        default=1,
        help="the thickness of the current sheet"
    )

    parser.add_argument(
        "--width", "-w",
        type=float,
        default=0,
        help="the width of the current sheet"
    )

    parser.add_argument(
        "--resolution-range", "-N",
        type=int, nargs=3,
        default=[64, 2048, 32],
        help="range of grid resolutions followed by increment"
    )

    parser.add_argument(
        "--inner-scale", "-δin", "--resistive-scale", "-δres",
        dest="inner_scale",
        type=float,
        default=None,
        help="the initial inner grid resolution scale"
    )

    parser.add_argument(
        "--inner-resolution-safety",
        type=float,
        default=1.0,
        help="safety factor applied to estimated inner scale (>= 1.0)"
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help=("The initial guess for growth rate.")
    )

    parser.add_argument(
        "--n-equilibrium", "-neq", "--n-inner", "-nin",
        dest="n_equilibrium",
        type=int,
        default=5,
        help=("Minimum number of collocation points required to resolve "
              "the equilibrium current sheet scale (a + w).")
    )

    parser.add_argument(
        "--n-inner-scale", "-nscale", "--n-resistivity", "-nres",
        dest="n_inner_scale",
        type=int,
        default=5,
        help=("Minimum number of collocation points required to resolve "
              "the inner layer resolution scale.")
    )

    parser.add_argument(
        "--n-anisotropy", "-naniso",
        type=int,
        default=5,
        help=("Minimum number of collocation points required to resolve "
              "the anisotropy pressure scale.")
    )



    parser.add_argument(
        "--amp-fraction-outer", "--f-outer",
        type=float,
        default=0.01,
        help=("Fraction of the eigenmode amplitude that must be resolved "
              "at the outermost collocation point z_max.  For example, "
              "0.01 means the eigenmode amplitude should drop to 1%% of its "
              "maximum at z_max (equivalent to q = -ln(0.01) ≈ 4.605).")
    )

    parser.add_argument(
        "--scaling-factor", "-C",
        type=float,
        default=None,
        help="the scaling factor for the grid"
    )

    parser.add_argument(
        "--scaling-mean", "--cmean",
        choices=['geometric', 'harmonic', 'average', 'lower', 'outer', 'upper', 'inner'],
        default='geometric',
        help="the method to calculate the grid scaling factor C from inner and outer limits"
    )

    parser.add_argument(
        "--dynamic-C", "-dynamic-c",
        action='store_true',
        default=False,
        help="dynamically calculate the scaling factor C at each resolution step"
    )

    parser.add_argument(
        "--absolute-tolerance", "-atol",
        type=float,
        default=1e-10,
        help="the absolute tolerance for the growth rate"
    )

    parser.add_argument(
        "--relative-tolerance", "-rtol",
        type=float,
        default=1e-5,
        help="the relative tolerance for the growth rate"
    )

    parser.add_argument(
        "--guess-tolerance", "-gtol",
        type=float,
        default=1e-2,
        help="the guess tolerance for switching to an iterative solver"
    )

    parser.add_argument(
        "--thickness-tolerance", "-δtol",
        type=float,
        default=1e-3,
        help="(deprecated) Legacy inner‑layer thickness tolerance; local bracket interpolation on the Chebyshev grid is directly grid-resolved."
    )

    parser.add_argument(
        "--real-part-range", "-E",
        type=float, nargs=2,
        default=[1e-6, 1],
        help="range of growth rates to consider"
    )

    parser.add_argument(
        "--imag-part-range", "-I",
        type=float, nargs=2,
        default=[-10.0, 10.0],
        help="maximum magnitude of the eigenvalue's imaginary part"
    )

    parser.add_argument(
        "--orderby", "-O",
        choices=['amplitude', 'real', 'imaginary', 'tolerance', 'errors'],
        default="real",
        help=("ordering of wavenumber: magnitude/amplitude, real and "
              "imaginary parts, or tolerance/errors")
    )

    parser.add_argument(
        "--suffix", "-s",
        default="",
        help="the suffix added to the output file"
    )

    parser.add_argument(
        "--force", "-f",
        action='store_true',
        default=False,
        help="force recalculations even though eigenmodes are already stored in cache directory"
    )

    parser.add_argument(
        "--allmodes", "-all",
        action='store_true',
        default=False,
        help="return all possible modes up to maxmodes"
    )

    parser.add_argument(
        "--no-shear",
        action='store_true',
        default=False,
        help="no U shear"
    )

    parser.add_argument(
        "--verbose", "-v",
        action='store_true',
        default=False,
        help="be verbose"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="path to a file where detailed logs will be saved"
    )

    parser.add_argument(
        "--logarithmic", "-log",
        action='store_true',
        default=False,
        help="scale is logarithmic"
    )

    # ------------------------------------------------------------------
    return parser


def build_parser(parser_type: str = 'dispersion') -> argparse.Namespace:
    """
    Construct and parse the command‑line interface for the tearing‑instability solver.

    Returns
    -------
    argparse.Namespace
        The validated arguments object.
    """
    description: str = ""
    if parser_type == 'dispersion':
        description = "Calculates the dispersion relation for tearing instability."
    elif parser_type == 'maxima':
        description = "Calculates the maximum growth rate for tearing instability."
    elif parser_type == 'profiles':
        description = "Plots eigenfunctions for tearing instability."
    elif parser_type == 'maps':
        description = "Plots 2D maps of eigenfunctions for tearing instability."
    elif parser_type == 'plot':
        description = "Plots eigenmode solutions from a .npz file."

    # ------------------------------------------------------------------
    # 1️⃣  Create a basic ArgumentParser instance (your helper handles
    #     formatting, defaults, etc.).
    parser = parser_setup(description=description)

    # ------------------------------------------------------------------
    # 2️⃣  Add all command‑line options that are relevant to this script.

    if parser_type == 'dispersion':
        parser.add_argument(
            "--wavenumber-range", "-K",
            type=float, nargs=3,
            default=[0.0, 1.0, 0.01],
            help="bounds and step size for the wavenumber"
        )
        parser.add_argument(
            "--mode", "-m",
            type=int,
            default=0,
            help="the eigenmode number"
        )
    elif parser_type == 'maximum':
        parser.add_argument(
            "--dependence", "-d",
            choices=['S', 'Pr', 'β', 'Δβ', 'ξ', 'ϵ', 'w', 'a'],
            default='S',
            help=("the dependence of the quantity to calculate: S, Pr, β, ξ, "
                "ϵ, w, or a")
        )
        parser.add_argument(
            "--range", "-R",
            type=float, nargs=3,
            default=[0, 1, 0.1],
            help="the range of the dependent parameter to evaluate followed by the increment"
        )
        parser.add_argument(
            "--wavenumber-bracket", "-K",
            type=float, nargs=2,
            default=None,
            help="the bracket for the wavenumber"
        )
        parser.add_argument(
            "--wavenumber-tolerance", "-ktol",
            type=float,
            default=1e-3,
            help="relative tolerance for the maximum wavenumber"
        )
        parser.add_argument(
            "--modes", "-m",
            type=int,
            default=0,
            help="the eigenmode number"
        )
        parser.add_argument(
            "--step",
            action='store_true',
            default=False,
            help="calculate values sequentially using extrapolation for initial guesses"
        )
        parser.add_argument(
            "--extrap-deg",
            type=int,
            default=2,
            help="degree of polynomial extrapolation (1=linear, 2=quadratic, 3=cubic)"
        )
        parser.add_argument(
            "--extrap-guard",
            type=float,
            default=0.01,
            help="maximum fractional deviation allowed for extrapolated bracket"
        )
        parser.add_argument(
            "--step-lower-factor",
            type=float,
            default=None,
            help="Limit the minimum search range for the eigenvalue's real part relative to the previous step (e.g. 0.5 for 0.5 * σ_prev)."
        )
        parser.add_argument(
            "--step-upper-factor",
            type=float,
            default=None,
            help="Limit the maximum search range for the eigenvalue's real part relative to the previous step (e.g. 1.5 for 1.5 * σ_prev)."
        )
    elif parser_type == 'plot':
        parser.add_argument(
            "--zmin",
            type=float,
            default=None,
            help="Minimum z for plotting"
        )
        parser.add_argument(
            "--zmax",
            type=float,
            default=None,
            help="Maximum z for plotting"
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=None,
            help="Wavenumber alpha (for dispersion runs)"
        )
        parser.add_argument(
            "--value",
            type=float,
            default=None,
            help="Value of the dependent parameter (for maxima runs)"
        )
        parser.add_argument(
            "--dependence", "-d",
            choices=['S', 'Pr', 'β', 'Δβ', 'ξ', 'ϵ', 'w', 'a'],
            default=None,
            help=("the dependence of the quantity to calculate: S, Pr, β, ξ, "
                "ϵ, w, or a")
        )
        parser.add_argument(
            "--file", "-F",
            type=str,
            default=None,
            help="Direct path to .npz file"
        )
        parser.add_argument(
            "--dir", "-D",
            type=str,
            default=None,
            help="Directory containing .npz files to plot"
        )
        parser.add_argument(
            "--output", "-o",
            type=str,
            default=None,
            help="Output plot filename"
        )
        parser.add_argument(
            "--nx",
            type=int,
            default=100,
            help="Number of points in the x direction for 2D mapping"
        )
        parser.add_argument(
            "--nperiods",
            type=float,
            default=1.0,
            help="Number of periods (wavelengths) to plot in x for 2D mapping"
        )

    # ------------------------------------------------------------------
    # 3️⃣  Check for local configuration file (params.cfg)
    config_file = "params.cfg"
    if os.path.exists(config_file):
        config_data = load_config(config_file)

        # Build mapping from option names to action destinations
        opt_to_dest = {action.dest: action.dest for action in parser._actions}
        for action in parser._actions:
            for opt in action.option_strings:
                opt_to_dest[opt.lstrip('-')] = action.dest

        typed_config: Dict[str, Any] = {}
        for key, val in config_data.items():
            if key in opt_to_dest:
                dest = opt_to_dest[key]
                # Find the action associated with this destination
                action = next(a for a in parser._actions if a.dest == dest)

                try:
                    if action.nargs and action.nargs != 1:
                        # Handle list-like arguments (e.g., 64 2048 32)
                        parts = val.strip('[]()').replace(',', ' ').split()
                        if action.type and callable(action.type):
                            typed_config[dest] = [action.type(p) for p in parts]
                        else:
                            typed_config[dest] = parts
                    elif isinstance(action, argparse._StoreTrueAction):
                        typed_config[dest] = val.lower() in ('true', 'yes', '1', 'on')
                    elif isinstance(action, argparse._StoreFalseAction):
                        typed_config[dest] = val.lower() in ('false', 'no', '0', 'off')
                    elif action.type and callable(action.type):
                        typed_config[dest] = action.type(val)
                    else:
                        typed_config[dest] = val
                except (ValueError, TypeError):
                    continue

        parser.set_defaults(**typed_config)

    # ------------------------------------------------------------------
    # 4️⃣  Parse the command line.
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 4️⃣  Validate the resulting namespace.  Any problem will cause
    #     ``parser.error`` to be invoked, which prints a message and exits.
    try:
        validate_parameters(args)
    except ParameterError as exc:   # defined elsewhere in your codebase
        parser.error(str(exc))

    return args


def build_params(parser_type: str = 'dispersion') -> SimulationParams:
    """
    Unified parameter builder for simulation arguments.
    """
    args = build_parser(parser_type=parser_type)

    if args.eos in [ 'isothermal', 'adiabatic', 'polytropic' ]:
        parallel_index, perpendicular_index = eos_indices(args.eos)
    else:
        parallel_index = args.gamma_parallel
        perpendicular_index = args.gamma_perpendicular

    # 2. Map internal keys to args attributes
    n_equilibrium = getattr(args, 'n_equilibrium', getattr(args, 'n_inner', 5))
    n_inner_scale = getattr(args, 'n_inner_scale', getattr(args, 'n_resistivity', 5))
    inner_scale = getattr(args, 'inner_scale', getattr(args, 'resistive_scale', None))
    safety = getattr(args, 'inner_resolution_safety', 1.0)

    params: Dict[str, Any] = {
        'data_path'             : None,
        'Nmin'                  : args.resolution_range[0],
        'Nmax'                  : args.resolution_range[1],
        'Ninc'                  : args.resolution_range[2],
        'n_inner'               : n_equilibrium,
        'n_equilibrium'         : n_equilibrium,
        'n_resistivity'         : n_inner_scale,
        'n_inner_scale'         : n_inner_scale,
        'n_anisotropy'          : args.n_anisotropy,
        'f_outer'               : args.amp_fraction_outer,
        'decay_efolds'          : -np.log(args.amp_fraction_outer),
        'CGL'                   : args.CGL,
        'C'                     : args.scaling_factor,
        'Cmean'                 : args.scaling_mean,
        'delta'                 : inner_scale,
        'inner_scale'           : inner_scale,
        'inner_resolution_safety': safety,
        'dynamic_C'             : args.dynamic_C,
        'alpha'                 : None,
        'sigma'                 : args.sigma,
        'sigma_real_lower'      : args.real_part_range[0],
        'sigma_real_upper'      : args.real_part_range[1],
        'sigma_imag_lower'      : args.imag_part_range[0],
        'sigma_imag_upper'      : args.imag_part_range[1],
        'orderby'               : args.orderby,
        'atol'                  : args.absolute_tolerance,
        'rtol'                  : args.relative_tolerance,
        'gtol'                  : args.guess_tolerance,
        'dtol'                  : args.thickness_tolerance,
        'ntasks'                : 1,
        'allmodes'              : args.allmodes,
        'suffix'                : args.suffix,
        'parallel_index'        : parallel_index,
        'perpendicular_index'   : perpendicular_index,
        'eos'                   : args.eos,
        'logarithmic'           : args.logarithmic,
        'noshear'               : args.no_shear,
        'force'                 : args.force,
        'verbose'               : args.verbose,
        'log_file'              : args.log_file,
    }

    # 3. Handle 'mode' vs 'modes' naming discrepancy
    params['mode'] = getattr(args, 'mode', getattr(args, 'modes', None))

    # 4. Handle Conditional Dependence Logic
    dep = getattr(args, 'dependence', None)
    params['dependence'] = dep

    # Mapping for the "None if dependence == X" logic
    # Added 'Δβ' and ensured internal key naming consistency
    dep_map = {
        'a':  ('thickness', 'a'),
        'w':  ('width', 'w'),
        'S':  ('Lundquist_number', 'S'),
        'Pr': ('Prandtl_number', 'Pr'),
        'ξ':  ('magnetic_transverse_field', 'xi'),
        'ϵ':  ('Hall_parameter', 'Hall'),
        'β':  ('plasma_beta', 'plasma_beta'),
        'Δβ': ('plasma_beta_difference', 'plasma_beta_difference')
    }

    for symbol, (arg_attr, param_key) in dep_map.items():
        val = getattr(args, arg_attr, None)
        # If the current parameter is the independent variable (dependence),
        # we set its value to None in the params dict as it is being swept.
        params[param_key] = None if dep == symbol else val

    # 5. Handle Divergent Range Logic (Wavenumber vs Generic Sweep Range)
    if hasattr(args, 'wavenumber_range'):
        # Ensure kmin is at least one step size if not logarithmic to avoid k=0 issues
        params['kmin'] = args.wavenumber_range[0] if args.logarithmic else max(args.wavenumber_range[0], args.wavenumber_range[2])
        params['kmax'] = args.wavenumber_range[1]
        params['kinc'] = args.wavenumber_range[2]

    if hasattr(args, 'range'):
        params['vmin'] = args.range[0]
        params['vmax'] = args.range[1]
        params['vinc'] = args.range[2]

    # 6. Add specific optional solver tolerances
    if hasattr(args, 'wavenumber_tolerance'):
        params['ktol'] = args.wavenumber_tolerance

    if hasattr(args, 'wavenumber_bracket'):
        params['kbracket'] = args.wavenumber_bracket

    if hasattr(args, 'step'):
        params['step']         = args.step
        params['extrap_deg']   = args.extrap_deg
        params['extrap_guard'] = args.extrap_guard
        params['step_lower_factor'] = args.step_lower_factor
        params['step_upper_factor'] = args.step_upper_factor

    if hasattr(args, 'zmin'):
        params['zmin']        = args.zmin
        params['zmax']        = args.zmax
        params['alpha_plot']  = args.alpha
        params['value_plot']  = args.value
        params['file_plot']   = args.file
        params['dir_plot']    = args.dir
        params['output_plot'] = args.output
        params['nx']          = getattr(args, 'nx', 100)
        params['nperiods']    = getattr(args, 'nperiods', 1.0)

    return SimulationParams(**params)


def _check_positive_tol(name: str, value: float) -> None:
    """
    Helper that raises :class:`ParameterError` if *value* is not strictly positive.

    Parameters
    ----------
    name :
        Human‑readable name of the tolerance (e.g. ``"absolute tolerance"``
        or ``"relative tolerance"``).
    value :
        The numeric value supplied by the user.
    """
    if value <= 0.0:
        raise ParameterError(
            f"{name} ({value:.3g}) must be > 0"
        )


def validate_parameters(args: argparse.Namespace) -> None:
    """
    Validate all command‑line arguments produced by :func:`parser_setup`.

    The function mutates *args* only if it finds a problem – otherwise
    it simply returns.  Any violation raises :class:`ParameterError`
    with an explanatory message that is suitable for printing to stderr.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace returned by ``parser.parse_args()``.
    """

    def get_range(attr: str, symbol: str) -> Tuple[Any, Any]:
        val = getattr(args, attr)
        if hasattr(args, 'dependence') and args.dependence == symbol:
            v1, v2, _ = args.range
            if args.logarithmic:
                return 10**v1, 10**v2
            else:
                return v1, v2
        return val, val

    # ------------------------------------------------------------------
    # 1) Physical bounds (fixed and ranges)
    # ------------------------------------------------------------------
    S_min, S_max = get_range('Lundquist_number', 'S')
    if min(S_min, S_max) <= 0:
        raise ParameterError("Lundquist number must be > 0")

    Pr_min, Pr_max = get_range('Prandtl_number', 'Pr')
    if min(Pr_min, Pr_max) < 0:
        raise ParameterError("Prandtl number cannot be negative")

    beta_min, beta_max = get_range('plasma_beta', 'β')
    if min(beta_min, beta_max) < 0:
        raise ParameterError("Plasma‑β must be ≥ 0")

    dbeta_min, dbeta_max = get_range('plasma_beta_difference', 'Δβ')
    if min(beta_min, beta_max) + min(dbeta_min, dbeta_max) < 0:
        raise ParameterError(
            f"Parallel plasma‑β (β∥ = β⊥ + Δβ) must be ≥ 0 "
            f"(min β⊥ = {min(beta_min, beta_max):.3g}, min Δβ = {min(dbeta_min, dbeta_max):.3g})"
        )

    xi_min, xi_max = get_range('magnetic_transverse_field', 'ξ')
    if min(xi_min, xi_max) < 0:
        raise ParameterError("Magnetic transverse field strength (ξ) must be ≥ 0")

    hall_min, hall_max = get_range('Hall_parameter', 'ϵ')
    if min(hall_min, hall_max) < 0:
        raise ParameterError("Hall parameter (ϵ) must be ≥ 0")

    a_min, a_max = get_range('thickness', 'a')
    if min(a_min, a_max) <= 0:
        raise ParameterError("Current sheet thickness (a) must be > 0")

    w_min, w_max = get_range('width', 'w')
    if min(w_min, w_max) < 0:
        raise ParameterError("Current sheet half‑width (w) must be ≥ 0")

    # ------------------------------------------------------------------
    # 2) Equation‑of‑state consistency
    # ------------------------------------------------------------------
    if args.eos == "custom":
        # When custom EOS is requested, both gamma values must be supplied.
        if not hasattr(args, "gamma_parallel") or not hasattr(args, "gamma_perpendicular"):
            raise ParameterError(
                "Both --gamma-parallel (-ɣpar) and "
                "--gamma-perpendicular (-ɣper) must be provided when using the 'custom' EOS"
            )
    # For built‑in EOS choices we ignore any user‑supplied gamma.

    # ------------------------------------------------------------------
    # 3) Resolution range sanity
    # ------------------------------------------------------------------
    Nmin, Nmax, Ninc = args.resolution_range

    if not (Nmin <= Nmax):
        raise ParameterError(
            f"Resolution range start ({Nmin}) must be ≤ end ({Nmax})."
        )

    if Ninc <= 0:
        raise ParameterError("Resolution increment (--resolution-range / -N) must be positive")

    if Ninc > Nmin:
        raise ParameterError(
            f"Resolution increment ({Ninc}) cannot exceed the minimum resolution "
            f"({Nmin}). Use a smaller step or increase Nmin."
        )

    # ------------------------------------------------------------------
    # 4) Tolerance sanity
    # ------------------------------------------------------------------
    _check_positive_tol("absolute tolerance", args.absolute_tolerance)
    _check_positive_tol("relative tolerance", args.relative_tolerance)
    _check_positive_tol("guess tolerance",     args.guess_tolerance)
    _check_positive_tol("thickness tolerance", args.thickness_tolerance)

    # ------------------------------------------------------------------
    # 5) Growth‑rate bounds
    # ------------------------------------------------------------------
    gmin, gmax = args.real_part_range
    if not (gmin <= gmax):
        raise ParameterError(
            f"Growth‑rate lower bound ({gmin}) must be ≤ upper bound ({gmax})."
        )
    if gmin < 0:
        raise ParameterError("Growth‑rate lower bound cannot be negative")

    # ------------------------------------------------------------------
    # 6) Inner collocation points with corresponding width
    # ------------------------------------------------------------------
    n_eq = getattr(args, 'n_equilibrium', 5)
    if hasattr(args, 'n_inner') and args.n_inner != 5:
        n_eq = args.n_inner
    elif hasattr(args, 'n_equilibrium') and args.n_equilibrium != 5:
        n_eq = args.n_equilibrium
    args.n_equilibrium = n_eq
    args.n_inner = n_eq

    n_in_scale = getattr(args, 'n_inner_scale', 5)
    if hasattr(args, 'n_resistivity') and args.n_resistivity != 5:
        n_in_scale = args.n_resistivity
    elif hasattr(args, 'n_inner_scale') and args.n_inner_scale != 5:
        n_in_scale = args.n_inner_scale
    args.n_inner_scale = n_in_scale
    args.n_resistivity = n_in_scale

    inner_scale = getattr(args, 'inner_scale', None)
    if inner_scale is None and hasattr(args, 'resistive_scale'):
        inner_scale = args.resistive_scale
    args.inner_scale = inner_scale
    args.resistive_scale = inner_scale

    safety = getattr(args, 'inner_resolution_safety', 1.0)
    if safety < 1.0:
        raise ParameterError("Inner resolution safety factor (--inner-resolution-safety) must be >= 1.0")

    if args.n_equilibrium < 3:
        raise ParameterError(
            "Minimum number of inner collocation points (--n-equilibrium / --n-inner / -nin) must be >= 3"
        )
    if args.n_inner_scale < 3:
        raise ParameterError(
            "Minimum number of resistivity layer collocation points (--n-inner-scale / --n-resistivity / -nres) must be >= 3"
        )
    if args.n_anisotropy < 3:
        raise ParameterError(
            "Minimum number of anisotropy pressure scale collocation points (--n-anisotropy / -naniso) must be >= 3"
        )

    # ------------------------------------------------------------------
    # 7) Optional limits
    # ------------------------------------------------------------------
    if args.scaling_factor is not None and args.inner_scale is not None:
        raise ParameterError(
            "Both '--scaling_factor / -C' and '--inner-scale / --resistive-scale / -δres' were provided. "
            "Only one of these options may be set at a time."
        )
    if args.inner_scale is not None and args.inner_scale <= 0:
        raise ParameterError("Resistive scale (--inner-scale / --resistive-scale / -δres) must be > 0")

    if hasattr(args, 'nx') and args.nx is not None:
        if args.nx <= 0:
            raise ParameterError("Number of x points (--nx) must be > 0")
    if hasattr(args, 'nperiods') and args.nperiods is not None:
        if args.nperiods <= 0:
            raise ParameterError("Number of periods (--nperiods) must be > 0")

# If we reach this point everything passed.
