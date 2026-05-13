from .exceptions import DeltaError, ConvergenceError
import numpy as np
from scipy.optimize import minimize_scalar


def select_NC(params):
    """
    Determine the Chebyshev–TB grid resolution N and scaling factor C for the linear tearing
    instability eigenproblem under the mapping z = C tan(θ).

    This routine enforces BOTH of the following constraints for a given set of physical
    parameters (S, Pr, β, etc.) and numerical requirements:
      (1) INNER resolution: At least n_inner collocation points must lie within
          the inner-layer width δ_α = min(a, δ_Cop, δ_FKR). Using the m = (n_inner-1)/2 node near
          the origin on the TB grid, |z_m| = C * tan(mπ/(N+1)) ≤ δ_α gives an UPPER bound:
              C_in(N) = δ_α / tan(mπ / (N+1)).
      (2) OUTER decay extent: The outermost collocation z_max must reach the location where the
          eigenmode amplitude has decayed by 'decay_efolds' e-foldings:
              z_req = decay_efolds / λ,
          where λ = α / μ and μ is the gyrotropic anisotropy factor (1.0 for MHD).
          For TB grids, z_max(N, C) = C * cot(π / (2*(N+1))). This yields a LOWER bound on C:
              C_out(N) = z_req / cot(π / (2*(N+1))) = z_req * tan(π / (2*(N+1))).

    With our convention (smaller C packs more points near z=0), feasibility requires:
        C_out(N) ≤ C ≤ C_in(N).
    The algorithm searches N = Nmin, Nmin+Ninc, ... up to Nmax for the first N satisfying
    C_out(N) ≤ C_in(N). It returns:
        N    = smallest feasible resolution,
        Clo  = the lower bound from the outer decay extent at that N.
        Cup  = the upper bound from the inner-layer resolution at that N.
        δα   = the inner-layer thickness at the given α.

    Notes
    -----
    • Keeps your scaling for α_m (needed for c_FKR and c_Cop prefactors).
    • Handles CGL via μ = sqrt((1 + ((γ∥+γ⟂−2)/2)*β + Δβ/2) / (1 − Δβ/2)); set μ=1 if CGL=False.
    • Enforces odd n_inner (centered grid has a node at z=0).
    • Raises DeltaError if Δ' ≤ 0 (no tearing at that α).
    • Raises ConvergenceError if no feasible N is found up to Nmax.

    Parameters (params)
    -------------------
    Nmin         : int   , starting resolution (default 128)
    Nmax         : int   , maximum resolution to try (default 1024)
    Ninc         : int   , resolution increment per iteration (default 32)
    n_inner      : int   , minimum collocation points inside δ_min (must be odd; default 5)
    decay_efolds : float , number of e-foldings to resolve at |z|max (default 3.51 ~ 3% amplitude)
    CGL          : bool  , enable CGL anisotropy (default False)
    S            : float , Lundquist number (default 1e4)
    Pr           : float , magnetic Prandtl number (used softly here;) (default 0)
    β, Δβ        : floats, gyrotropic plasma-β and anisotropy (defaults β=0, Δβ=0)
    ɣpar, ɣper   : floats, aidabatic indices (γ∥=3, γ⟂=2)
    a            : float , current-sheet half-width (default 1)
    α            : float , dimensionless wavenumber (k * a) at which to set the grid (default 0.1)
    verbose      : bool  , print per-iteration diagnostics (default False)

    Returns
    -------
    N_opt : int
        Smallest resolution satisfying both constraints.
    C_opt : float
        Scaling factor chosen at the outer lower bound, C_out(N_opt).

    Raises
    ------
    DeltaError
    ConvergenceError
    """
    Nmin         = params.get('Nmin'                  ,  128       )
    Nmax         = params.get('Nmax'                  , 1024       )
    Ninc         = params.get('Ninc'                  ,   32       )
    n_inner      = params.get('n_inner'               ,    5       )
    m_osc        = params.get('m_osc'                 ,    9       )
    l_inner      = params.get('l_inner'               ,    0.1     )
    decay_efolds = params.get('decay_efolds'          ,    3.51    )
    CGL          = params.get('CGL'                   , False      )
    S            = params.get('S'                     ,    1.0e4   )
    Pr           = params.get('Pr'                    ,    0.0     )
    β            = params.get('plasma_beta'           ,    0.0     )
    Δβ           = params.get('plasma_beta_difference',    0.0     )
    ɣpar         = params.get('parallel_index'        ,    3       )
    ɣper         = params.get('perpendicular_index'   ,    2       )
    a            = params.get('a'                     ,    1       )
    w            = params.get('w'                     ,    0.0     )
    α            = params.get('alpha'                 ,    0.1     )
    θ            = params.get('theta'                 ,    8.0     )
    δ            = params.get('delta'                 , None       )
    σ            = params.get('sigma'                 , None       )
    ξ            = params.get('xi'                    ,    0.0     )
    verbose      = params.get('verbose'               , False      )
    Cmean        = params.get('Cmean'                 , 'geometric')

    # Enforce odd n_inner (so m is integer and z=0 is a collocation point)
    if n_inner % 2 == 0:
        n_inner += 1
    m = (n_inner - 1) / 2

    # --- Δ'(α); raise if stable at α
    if CGL and (Δβ <= - (2 + (ɣpar + ɣper - 2) * β) / ɣpar or Δβ >= 2):
        raise DeltaError(f"Δ' purely imaginary for Δβ = {Δβ:+.3e} => stable eigenmode for α = {α:.3e}.")

    # --- CGL anisotropy factor and scaled wavenumbers
    C = 1 - Δβ/2
    μ = np.sqrt((2 + (ɣpar + ɣper - 2) * β + ɣpar * Δβ) / (2 - Δβ)) if CGL else 1
    λ = α / μ

    if δ is None:
        # --- Δ'(α); raise if stable at α
        X = α * μ
        Δ = 2 * (1 / X - X)
        if Δ <= 0.0:
            raise DeltaError(f"Δ' <= 0 (Δ = {Δ:.3e}) ⇒ stable eigenmode for α = {α:.3e}.")

        # -- Prefactors from fits to numerical results
        fS   = (3.8288e-2 / (4.7443e-2 + S**-0.46355) + 7.9649e-2)
        gS   = 0.91451 - 2.0654 / (S**0.37651 + 0.88448)
        fPr  = (1.0777 + ((Pr * 0.71554) * (5.765 + Pr)))**0.079176
        gPr  = 2.4379 * ((Pr + 4.5991e-3)**0.16186)

        # --- Inner-layer widths from theoretical scalings modified by prefactors
        δCop = fS * fPr * a * (α * S)**(-1.0/3.0)
        δFKR = gS * gPr * a * ((S * α)**-2 * a * Δ)**(1.0/5.0)

        # --- Smooth the inner-layer thickness from both regimes
        δα = δCop * δFKR / (δCop**θ + δFKR**θ)**(1.0/θ)
        if verbose:
            print(f"[estimated inner scales for α={α:.4e}] a={a:.4e}, δCop={δCop:.4e}, δFKR={δFKR:.4e}, δα={δα:.4e}")
    else:
        δα = δ
        if verbose:
            print(f"[provided inner scales for α={α:.4e}] a={a:.4e}, δα={δα:.4e}")

    πh   = np.pi / 2
    πm   = np.pi * m
    Ntop = Nmax - 3 * Ninc

    # --- Outer requirement: amplitude reduced by e^{-decay_efolds} at |z|max
    #     z_req = decay_efolds / λ; TB outermost node: |z|max(N,C) = C * cot(π/(2(N+1)))
    #zmin = min(δα, l_inner, w + a) if δα > 1.0e-3 else min(l_inner, w + a)
    # zmin = min(δα, w + a) if δα > 1.0e-3 else w + a #np.sqrt(max(w, a) * (w + a))
    zmin = a + w
    zmax = decay_efolds / λ
    if verbose:
        print(f"[estimated for α={α:.4e}] zmin={zmin:.4e}, zmax={zmax:.4e}")
    lk = ξ / α
    if σ is not None and ξ > 0.0:
        lσ = ξ / σ.real
        lk = 2.0 * np.pi * ξ / np.abs(α + σ.imag)

        if lσ <= zmin:
            zmin = np.sqrt(zmin * lσ)
        else:
            zmax = max(zmax, decay_efolds * lσ)
        Copt = max(lσ, np.sqrt(max(w, a) * (w + a)))
    else:
        lσ   = 0.0
        Copt = np.sqrt(max(w, a) * (w + a))

    if verbose:
        print(f"[all scales for α={α:.4e}] (w+a) = {w+a:.4e}, lσ = {lσ:.4e}, lk = {lk:.4e}, zmin = {zmin:.4e}, zmax = {zmax:.4e},  Copt = {Copt:.4e},  Nmin = {Nmin:4d}")

    # Iterate N upward until C_out(N) <= C_in(N); then set C = C_out(N).
    N  = Nmin
    while True:
        if N > Ntop:
            raise ConvergenceError(
                f"Insufficient N up to Nmax={Nmax}: cannot satisfy C_out(N) <= C_in(N). "
                f"Try increasing Nmax or relaxing n_inner/decay_efolds."
            )
        Np   = N + 1
        Cinn  = zmin / np.tan(πm / Np)
        Cout  = zmax * np.tan(πh / Np)
        if verbose:
            print(f"[grid scale constrains for α={α:.4e}] Nmin = {N:4d}  C_inner={Cinn:.4e}  C_outer={Cout:.4e}")
        if Cout <= Cinn:
            break

        N += Ninc

    if Cmean == 'harmonic':
        C = 2.0 * Cinn * Cout / (Cinn + Cout)
    elif Cmean == 'geometric':
        C = np.sqrt(Cinn * Cout)
    elif Cmean == 'average':
        C = 0.5 * (Cinn + Cout)
    elif Cmean in ['lower', 'outer']:
        C = Cout
    elif Cmean in ['upper', 'inner']:
        C = Cinn
    else:
        C = max(Cinn, Cmax)

    if verbose:
        print(f"[grid scale constrains for α={α:.4e}] Nmin = {N:4d}  C_inner={Cinn:.4e}  C_outer={Cout:.4e} => C = {C:.6e} using {Cmean} mean")

    return N, C, δα


def make_fast_interpolator(grid):
    """
    Robust single-point interpolator for fields defined on grid.zg,
    using global barycentric interpolation on the mapped Chebyshev–Gauss nodes.

    This is intended to be constructed once per grid and reused.
    """
    # Cache mapped Chebyshev–Gauss nodes (ascending, matching grid.zg ordering)
    xj = np.clip(grid.zg / np.sqrt(grid.C**2 + grid.zg**2), -1.0, 1.0)
    N = xj.size

    # Barycentric weights for Chebyshev–Gauss nodes:
    # w_j ∝ (-1)^(j-1) * sin(theta_j), theta_j=(2j-1)π/(2N)
    j = np.arange(1, N + 1)
    theta = (2*j - 1) * np.pi / (2*N)
    wj = ((-1.0)**(j - 1)) * np.sin(theta)

    # theta construction corresponds to x=cos(theta) in descending order; reverse to match xj ascending
    wj = wj[::-1]

    eps = 50 * np.finfo(float).eps

    def fast_interpolate(z_target, f_values):
        f_values = np.asarray(f_values)

        x = z_target / np.sqrt(grid.C**2 + z_target**2)

        # Tolerant "hit a node" shortcut
        idx = int(np.argmin(np.abs(x - xj)))
        if np.abs(x - xj[idx]) <= eps:
            return float(f_values[idx])

        tmp = wj / (x - xj)
        return float(np.dot(tmp, f_values) / np.sum(tmp))

    return fast_interpolate


def inner_layer_thickness(system, δtol=1e-3, maxiter=20):
    """
    Calculate the inner layer thickness δ for the tearing instability eigenmode.
    Parameters
    ----------
    system : psecas.systems.tearing.Tearing
        The tearing instability system object.
    α : float
        The dimensionless wavenumber (k * a).
    σ : complex
        The eigenvalue (growth rate) of the mode.
    Returns
    -------
    δ : float
        The inner layer thickness.
    """
    grid = system.grid
    sol  = system.result

    C = grid.C
    s = system.shear
    a = system.a
    w = system.w
    S = system.S
    ξ = system.ξ
    α = system.kx * a
    u = sol['duz']
    b = sol['dbz']
    F = system.Bx
    G = system.Ux

    if s:
        Ti = np.abs(1j * α * (u * F - G * b) + ξ * grid.derivative(u, 1))
    else:
        Ti = np.abs(1j * α *  u * F          + ξ * grid.derivative(u, 1))
    Tn = np.abs(grid.derivative(b, 2) - α**2 * b) / S
    Td = Tn - Ti

    I   = np.where((0.0 <= grid.zg) & (grid.zg <= (w + 2 * a)))
    zin = grid.zg[I]
    Tin = Tn[I] - Ti[I]

    # Build interpolator once
    fast_interp = make_fast_interpolator(grid)

    # check if Td changes sign in the interval
    if Tin.min() <= 0.0 and Tin.max() >= 0.0:
        # start from z at which Tin is maximum and search at which distance Td becomes negative
        idx = Tin.argmax()
        while Tin[idx] > 0.0 and idx < (Tin.size - 1):
            idx += 1
        idx -= 1
        if idx < (Tin.size - 1):
            # endpoints of the bracket on the sliced arrays
            zl, zh = float(zin[idx]), float(zin[idx + 1])
            tl, th = float(Tin[idx]), float(Tin[idx + 1])

            # If bracket does not change sign, fall back to choosing the best of endpoints/midpoint
            if tl * th > 0.0:
                zm = 0.5 * (zl + zh)
                tm = fast_interp(zm, Td)
                # pick the point with smallest absolute value
                best = min(((abs(tl), zl), (abs(tm), zm), (abs(th), zh)), key=lambda x: x[0])[1]
                δ = float(best)
            else:
                it = 0
                while 2.0 * (zh - zl) > δtol * (zh + zl) and it < maxiter:
                    zm = 0.5 * (zl + zh)
                    tm = fast_interp(zm, Td)
                    if tm == 0.0:
                        zl = zh = zm
                        break
                    if tl * tm > 0:
                        zl, tl = zm, tm
                    else:
                        zh, th = zm, tm
                    it += 1
                δ = float(0.5 * (zl + zh))
    else:
        # no region where Td >= 0: set δ to zero (no inner layer detected)
        δ = 0.0

    I = np.where(np.abs(grid.zg) <= δ)
    n = I[0].size

    return δ, n


def find_peak_location(u0, b0, grid, a=1.0, w=0.0, ztol=1.0e-3, maxiter=50):
    """
    Find the closest positive peak location among u, b, and their first two derivatives.

    The initial scan is done on the grid points in 0 <= z <= a.  Once the field with
    the closest positive peak is selected, its peak location is refined by maximizing
    the interpolated absolute value over the neighboring grid interval.
    """
    u1 = grid.derivative(u0, 1)
    u2 = grid.derivative(u0, 2)
    b1 = grid.derivative(b0, 1)
    b2 = grid.derivative(b0, 2)

    fields = {'u0': u0, 'u1': u1, 'u2': u2, 'b0': b0, 'b1': b1, 'b2': b2}

    z_peak = w + 2 * a
    field_idx = None
    peak_idx = None

    # Quickly find the peak on the base grid before refining the winning field.
    I = np.where((0.0 <= grid.zg) & (grid.zg <= (w + 2 * a)))[0]
    z = grid.zg[I]
    for f, values in fields.items():
        i = np.abs(values[I]).argmax()
        idx = I[i]

        if 0.0 < z[i] < z_peak:
            z_peak = z[i]
            field_idx = f
            peak_idx = idx

    # Refine using one neighboring grid point on each side of the coarse peak.
    if field_idx is not None and 0 < peak_idx < (grid.zg.size - 1):
        zl = max(0.0, float(grid.zg[peak_idx - 1]))
        zh = min(float(a), float(grid.zg[peak_idx + 1]))

        if zl < z_peak < zh:
            def objective(zz):
                return -float(np.abs(grid.interpolate(zz, fields[field_idx])))

            result = minimize_scalar(
                objective,
                bounds=(zl, zh),
                method='bounded',
                options={'xatol': ztol * z_peak, 'maxiter': maxiter},
            )

            candidates = [float(z_peak), zl, zh]
            if result.success and zl <= result.x <= zh:
                candidates.append(float(result.x))

            z_peak = max(candidates, key=lambda zz: -objective(zz))

    I = np.where(np.abs(grid.zg) <= (w + a))
    n = I[0].size
    if w > 0:
        I = np.where(np.abs(grid.zg) <= w)
        n -= I[0].size

    return z_peak, n
