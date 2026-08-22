from .params import SimulationParams
import logging

logger = logging.getLogger(__name__)


def print_info(params: SimulationParams) -> None:
    """
    Print the physical and numerical parameters used for the calculation.
    """

    def log_head(label, key, fmt="10.3e"):
        val = getattr(params, key)
        if val is not None:
            logger.info(f"  {label:<34} =  {val:{fmt}}")

    logger.info("Plasma parameters:")
    eq_type = "Gyrotropic" if params.CGL else "Classical"
    logger.info(f"  {'Equations':<34} =  {eq_type} MHD")
    logger.info(f"  {'Equation of State':<34} =  {params.eos}")

    log_head("Lundquist number (S)", 'S')
    log_head("Prandtl number (Pr)", 'Pr')
    log_head("Plasma-β (β)", 'plasma_beta')
    log_head("Plasma-β difference (Δβ)", 'plasma_beta_difference')
    log_head("Magnetic transverse field (ξ)", 'xi')
    log_head("Hall current term strength (ϵ)", 'Hall')

    if params.CGL:
        logger.info(f"  {'Parallel adiabatic index':<34} =  {params.parallel_index}")
        logger.info(f"  {'Perpendicular adiabatic index':<34} =  {params.perpendicular_index}")

    logger.info("Equilibrium parameters:")
    log_head("Current sheet thickness (a)", 'a')
    log_head("Current sheet half-width (w)", 'w')

    logger.info(f"  {'Velocity shear     ':<34} =  {not params.noshear}")

    vmin, vmax, vinc = params.vmin, params.vmax, params.vinc
    is_log = "log10" if params.logarithmic else ""

    if params.dependence:
        logger.info("Dependence parameters:")
        logger.info(f"  {'Calculating dependence on':<34} =  {params.dependence}")
        logger.info(f"  {'Parameter value range (R)':<34} =  {is_log}[{vmin}, {vmax}] with increment {vinc}")

        if params.kbracket:
            logger.info(f"  {'Wavenumber bracket':<34} =  {params.kbracket}")

    else:
        logger.info("Wavenumbers:")
        kmin, kmax, kinc = params.kmin, params.kmax, params.kinc
        logger.info(f"  {'Wavenumber range (k)':<34} =  {is_log}[{kmin}, {kmax}] with increment {kinc}")

    logger.info("Geometry/convergence parameters:")
    Nmin, Nmax, Ninc = params.Nmin, params.Nmax, params.Ninc
    logger.info(f"  {'Resolution range (N)':<34} =  [{Nmin}, {Nmax}] with increment {Ninc}")

    log_head("Growth rate absolute tolerance", 'atol')
    log_head("Growth rate relative tolerance", 'rtol')
    log_head("Guess tolerance", 'gtol')
    logger.info(f"  {'Real part range':<34} =  [{params.sigma_real_lower}, {params.sigma_real_upper}]")
    logger.info(f"  {'Imaginary part range':<34} =  [{params.sigma_imag_lower}, {params.sigma_imag_upper}]")

    logger.info(f"  {'Equilibrium collocation points':<34} =  {params.n_equilibrium}")
    logger.info(f"  {'Inner scale collocation points':<34} =  {params.n_inner_scale}")
    logger.info(f"  {'Anisotropy scale points':<34} =  {params.n_anisotropy}")
    logger.info(f"  {'Inner resolution safety factor':<34} =  {params.inner_resolution_safety}")
    logger.info(f"  {'Amplitude fraction at zmax':<34} =  {params.f_outer}")
    logger.info(f"  {'Dynamic C grid':<34} =  {'on' if params.dynamic_C else 'off'}")

    log_head("Inner scale (δin)", 'inner_scale')
    log_head("Inner-layer thickness tolerance", 'dtol')

    logger.info("Miscellaneous/Runtime:")

    logger.info(f"  {'Eigenmode order':<34} =  {params.orderby}")

    mode_val = params.mode
    logger.info(f"  {'Converge the mode':<34} =  {mode_val if mode_val is not None else 'N/A'}")

    logger.info(f"  {'Return all modes':<34} =  {params.allmodes}")
    logger.info(f"  {'Force recalculation':<34} =  {params.force}")

    if params.suffix:
        logger.info(f"  {'Suffix':<34} =  {params.suffix}")
