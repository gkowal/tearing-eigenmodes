def print_info(params: dict) -> None:
    """
    Unified pretty-printer for simulation parameters.
    Handles both standard runs and parameter dependence studies.
    """
    # --- Helper: Print if key exists and is not None ---
    def print_val(label, key, fmt=".3e"):
        val = params.get(key)
        if val is not None:
            print(f"  {label:<34} =  {val:{fmt}}")

    # --- Plasma parameters ----------------------------------------------------
    print("Plasma parameters:")
    eq_type = 'Gyrotropic' if params.get('CGL') else 'Classical'
    print(f"  {'Equations':<34} =  {eq_type} MHD")
    print(f"  {'Equation of State':<34} =  {params.get('eos')}")

    print_val("Lundquist number (S)", 'S')
    print_val("Prandtl number (Pr)", 'Pr')

    if params.get('CGL'):
        print_val("Plasma-β (β)", 'plasma_beta')
        print_val("Plasma-β difference (Δβ)", 'plasma_beta_difference', "+.3f")
        print(f"  {'Parallel adiabatic index':<34} =  {params.get('parallel_index')}")
        print(f"  {'Perpendicular adiabatic index':<34} =  {params.get('perpendicular_index')}")

    print_val("Magnetic transverse field (ξ)", 'xi')
    print_val("Hall current term strength (ϵ)", 'Hall')

    # --- Equilibrium parameters -----------------------------------------------
    print("Equilibrium parameters:")
    print_val("Current sheet thickness (a)", 'a')
    print_val("Neutral layer width (w)", 'w')
    print_val("Initial inner-layer thickness", 'delta')
    print(f"  {'Velocity shear     ':<34} =  {not params.get('noshear')}")

    # --- Wavenumbers / Dependence ---------------------------------------------
    is_log = '10^' if params.get('logarithmic') else ''

    if params.get('dependence'):
        print("Dependence parameters:")
        print(f"  {'Calculating dependence on':<34} =  {params['dependence']}")
        vmin, vmax, vinc = params.get('vmin'), params.get('vmax'), params.get('vinc')
        print(f"  {'Parameter value range (R)':<34} =  {is_log}[{vmin}, {vmax}] with increment {vinc}")

        # --- Missing Wavenumber Search Params ---
        if 'kbracket' in params:
            print(f"  {'Wavenumber bracket':<34} =  {params['kbracket']}")
        print_val("Wavenumber tolerance (ktol)", 'ktol')
    else:
        print("Wavenumbers:")
        kmin, kmax, kinc = params.get('kmin'), params.get('kmax'), params.get('kinc')
        print(f"  {'Wavenumber range (k)':<34} =  {is_log}[{kmin}, {kmax}] with increment {kinc}")

    # --- Geometry / convergence parameters ------------------------------------
    print("Geometry/convergence parameters:")
    Nmin, Nmax, Ninc = params['Nmin'], params['Nmax'], params['Ninc']
    print(f"  {'Resolution range (N)':<34} =  [{Nmin}, {Nmax}] with increment {Ninc}")
    print_val("Scaling factor (C)", 'C')

    # --- Growth-rate / tolerance ---------------------------------------------
    print(f"  {'Real part range':<34} =  [{params['sigma_real_lower']}, {params['sigma_real_upper']}]")
    print(f"  {'Imaginary part range':<34} =  [{params['sigma_imag_lower']}, {params['sigma_imag_upper']}]")
    print_val("Imaginary amplitude limit", 'sigma_imag')
    print_val("Growth rate absolute tolerance", 'atol')
    print_val("Growth rate relative tolerance", 'rtol')
    print_val("Growth rate guess tolerance", 'gtol')
    print(f"  {'Number of inner collocation points':<34} =  {params['n_inner']}")
    print(f"  {'Width for the collocation points':<34} =  {params['l_inner']}")
    print(f"  {'Amplitude fraction at zmax':<34} =  {params['f_outer']}")
    print_val("Decay e-folds at zmax", 'decay_efolds')

    # --- Miscellaneous ---------------------------------------------------------
    print("Miscellaneous/Runtime:")
    print_val("Inner-layer thickness tolerance", 'dtol')
    print(f"  {'Eigenmode order':<34} =  {params['orderby']}")

    # Check 'is not None' so that mode 0 is printed
    mode_val = params.get('mode')
    print(f"  {'Converge the mode':<34} =  {mode_val if mode_val is not None else 'N/A'}")

    print(f"  {'Return all modes':<34} =  {params.get('allmodes')}")
    print(f"  {'Force recalculation':<34} =  {params.get('force')}")

    if params.get('suffix'):
        print(f"  {'Suffix':<34} =  {params['suffix']}")
