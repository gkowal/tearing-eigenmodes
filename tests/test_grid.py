import pytest
import numpy as np
from tearing_eigenmodes import select_NC, SimulationParams
from tearing_eigenmodes.grid import select_C_for_N
from tearing_eigenmodes.exceptions import DeltaError, ConvergenceError

def test_select_nc_default():
    # Simple default run
    params = SimulationParams(
        alpha=0.1,
        a=1.0,
        w=0.0,
        CGL=False,
    )
    N, C = select_NC(params)
    assert isinstance(N, int)
    assert isinstance(C, float)
    assert N >= 64
    assert C > 0.0

def test_select_nc_cmean_strategies():
    means = ['geometric', 'harmonic', 'average', 'lower', 'outer', 'upper', 'inner', 'unknown_mean']
    results = {}
    for m in means:
        params = SimulationParams(
            alpha=0.1,
            a=1.0,
            w=0.0,
            CGL=False,
            Cmean=m,
        )
        N, C = select_NC(params)
        results[m] = (N, C)

    # Check that all scaling strategies return float scaling factors and int resolutions
    for m in means:
        assert isinstance(results[m][0], int)
        assert isinstance(results[m][1], float)

def test_select_nc_cgl_success():
    params = SimulationParams(
        alpha=0.1,
        a=1.0,
        w=0.0,
        CGL=True,
        plasma_beta=1.0,
        plasma_beta_difference=0.1,
        parallel_index=3.0,
        perpendicular_index=2.0,
    )
    N, C = select_NC(params)
    assert isinstance(N, int)
    assert isinstance(C, float)

def test_select_nc_cgl_infinite_decay():
    # Trigger D = 0
    # D = 1.0 + R + 0.5 * ((ɣpar + ɣper - 2) * β + ɣpar * Δβ)
    # Let R = 0.0 (by setting σ to 0.0j)
    # We want (ɣpar + ɣper - 2) * β + ɣpar * Δβ = -2
    # Let ɣpar = 3, ɣper = 2. Then (ɣpar + ɣper - 2) = 3.
    # If β = 0, then 3 * Δβ = -2 => Δβ = -2/3.
    params = SimulationParams(
        alpha=0.1,
        a=1.0,
        w=0.0,
        CGL=True,
        plasma_beta=0.0,
        plasma_beta_difference=-2.0 / 3.0,
        parallel_index=3.0,
        perpendicular_index=2.0,
        sigma=complex(0.0, 0.0),
    )
    with pytest.raises(DeltaError, match="Infinite decaying factor"):
        select_NC(params)

def test_select_nc_cgl_imaginary_decay():
    # Trigger lambda_sq_ratio < 0 => A / R0 < 0
    # A = 1.0 + chi - Δβ/2
    # Let σ = 0.0j => chi = 0 => A = 1.0 - Δβ/2
    # Let Δβ = 3.0 => A = 1.0 - 1.5 = -0.5
    # R0 = 1.0 + 0.5 * (3 * β + 3 * 3) = 1.0 + 1.5 * β + 4.5
    # If β = 0, R0 = 5.5 > 0.
    # Since A = -0.5 < 0 and R0 = 5.5 > 0, lambda_sq_ratio = A / R0 = -0.5 / 5.5 < 0.
    params = SimulationParams(
        alpha=0.1,
        a=1.0,
        w=0.0,
        CGL=True,
        plasma_beta=0.0,
        plasma_beta_difference=3.0,
        parallel_index=3.0,
        perpendicular_index=2.0,
        sigma=complex(0.0, 0.0),
    )
    with pytest.raises(DeltaError, match="Stable or unphysical regime"):
        select_NC(params)

def test_select_nc_convergence_error():
    # Trigger ConvergenceError by forcing N > Ntop immediately
    params = SimulationParams(
        alpha=0.1,
        a=1.0,
        w=0.0,
        Nmin=64,
        Nmax=64,
        Ninc=1,
    )
    with pytest.raises(ConvergenceError, match="Insufficient N up to Nmax"):
        select_NC(params)

def test_select_nc_invalid_inputs():
    # alpha <= 0
    with pytest.raises(ValueError, match="Wavenumber alpha must be positive"):
        select_NC(SimulationParams(alpha=0.0))
    
    # decay_efolds <= 0
    with pytest.raises(ValueError, match="decay_efolds must be positive"):
        select_NC(SimulationParams(alpha=0.1, decay_efolds=0.0))

    # a <= 0
    with pytest.raises(ValueError, match="Current sheet thickness a must be positive"):
        select_NC(SimulationParams(alpha=0.1, a=0.0))

    # w < 0
    with pytest.raises(ValueError, match="Current sheet half-width w must be non-negative"):
        select_NC(SimulationParams(alpha=0.1, w=-0.1))

    # Nmin <= 0
    with pytest.raises(ValueError, match="Resolution limits Nmin and Nmax must be positive"):
        select_NC(SimulationParams(alpha=0.1, Nmin=0))

    # Ninc <= 0
    with pytest.raises(ValueError, match="Resolution increment Ninc must be positive"):
        select_NC(SimulationParams(alpha=0.1, Ninc=0))

def test_select_nc_cgl_coefficient_guards():
    # A <= 0
    # A = 1.0 + chi - Δβ/2. Set Δβ = 3.0 => A = -0.5
    with pytest.raises(DeltaError, match="Stable or unphysical regime"):
        select_NC(SimulationParams(
            alpha=0.1,
            a=1.0,
            w=0.0,
            CGL=True,
            plasma_beta_difference=3.0,
        ))

    # D <= 0
    # D = 1.0 + R + 0.5 * ((ɣpar + ɣper - 2) * β + ɣpar * Δβ)
    # Let β = 1.0, Δβ = -2.0, ɣpar = 3.0, ɣper = 2.0.
    # D = 1.0 + 0.5 * (3 * 1.0 + 3 * (-2.0)) = 1.0 + 0.5 * (-3.0) = -0.5
    with pytest.raises(DeltaError, match="purely imaginary"):
        select_NC(SimulationParams(
            alpha=0.1,
            a=1.0,
            w=0.0,
            CGL=True,
            plasma_beta=1.0,
            plasma_beta_difference=-2.0,
            parallel_index=3.0,
            perpendicular_index=2.0,
        ))

def test_select_nc_stable_sigma_guarded():
    # If σ.real <= 0, the lσ calculation should be bypassed and not crash
    params = SimulationParams(
        alpha=0.1,
        a=1.0,
        w=0.0,
        xi=1.0,
        sigma=complex(-0.05, 0.0), # σ.real <= 0
    )
    # This should complete successfully because the stable σ.real is bypassed
    N, C = select_NC(params)
    assert N >= 64
    assert C > 0.0

def test_select_nc_resistive_scale():
    # Verify that specifying a small resistive scale delta and n_resistivity forces a larger resolution N (or smaller C)
    params_no_res = SimulationParams(
        alpha=0.1,
        a=1.0,
        w=0.0,
        delta=None,
    )
    N_no_res, C_no_res = select_NC(params_no_res)

    params_with_res = SimulationParams(
        alpha=0.1,
        a=1.0,
        w=0.0,
        delta=0.01,
        n_resistivity=5,
    )
    N_with_res, C_with_res = select_NC(params_with_res)

    # With a small delta, C_inner is much smaller, so it should take a higher N (or smaller C) to satisfy Cout <= Cinn
    assert (N_with_res > N_no_res) or (C_with_res < C_no_res)


def test_select_nc_anisotropy_scale():
    from tearing_eigenmodes.grid import calculate_anisotropy_scale, select_C_for_N
    
    # 1. verify calculate_anisotropy_scale handles non-CGL or zero sigma
    params_no_cgl = SimulationParams(CGL=False, alpha=0.1, a=1.0, sigma=0.05)
    assert calculate_anisotropy_scale(params_no_cgl) == 0.0

    # 2. verify with CGL and growth rate > 0
    # beta_bar = 0.5 * ((3 + 2 - 2) * 1.0 + (3 - 1) * 0.1) = 0.5 * (3 * 1.0 + 2 * 0.1) = 0.5 * 3.2 = 1.6
    # delta_aniso = gamma / (alpha * sqrt(beta_bar)) = 0.05 / (0.1 * sqrt(1.6))
    params_cgl = SimulationParams(
        CGL=True,
        alpha=0.1,
        a=1.0,
        plasma_beta=1.0,
        plasma_beta_difference=0.1,
        parallel_index=3.0,
        perpendicular_index=2.0,
        sigma=0.05,
    )
    scale = calculate_anisotropy_scale(params_cgl)
    expected = 0.05 / (0.1 * np.sqrt(1.6))
    assert np.isclose(scale, expected)

    # 3. verify that dynamic C selection shrinks when anisotropy scale is small
    # Large scale -> large C
    params_large = SimulationParams(
        CGL=True,
        alpha=0.1,
        a=1.0,
        plasma_beta=1.0,
        plasma_beta_difference=0.1,
        parallel_index=3.0,
        perpendicular_index=2.0,
        sigma=1.0,  # larger scale
    )
    _, _, C_large = select_C_for_N(64, params_large)

    # Small scale -> smaller C to resolve the narrow layer
    params_small = SimulationParams(
        CGL=True,
        alpha=0.1,
        a=1.0,
        plasma_beta=1.0,
        plasma_beta_difference=0.1,
        parallel_index=3.0,
        perpendicular_index=2.0,
        sigma=0.01,  # smaller scale
    )
    _, _, C_small = select_C_for_N(64, params_small)

    assert C_small < C_large


def test_select_c_for_n_exact_formulas():
    """Document existing exact formula evaluation in select_C_for_N."""
    params = SimulationParams(
        alpha=0.2,
        a=1.5,
        w=0.5,
        n_inner=5,
        decay_efolds=4.0,
        CGL=False,
        delta=None,
    )
    N = 128
    Cinn, Cout, C = select_C_for_N(N, params)

    zmin = 1.5 + 0.5  # a + w = 2.0
    m = (5 - 1) / 2   # 2
    Np = N + 1        # 129
    expected_Cinn = zmin / np.tan(np.pi * m / Np)

    zmax = 4.0 / 0.2  # decay_efolds / alpha = 20.0
    expected_Cout = zmax * np.tan((np.pi / 2) / Np)

    assert np.isclose(Cinn, expected_Cinn)
    assert np.isclose(Cout, expected_Cout)
    assert np.isclose(C, Cinn)


def test_select_c_for_n_explicit_resistive_override():
    """Document existing explicit delta override in select_C_for_N."""
    params = SimulationParams(
        alpha=0.2,
        a=1.0,
        w=0.0,
        n_inner=5,
        n_resistivity=7,
        delta=0.05,
        decay_efolds=4.0,
        CGL=False,
    )
    N = 128
    Cinn, Cout, C = select_C_for_N(N, params)

    Np = N + 1
    Cinn_eq = 1.0 / np.tan(np.pi * 2 / Np)
    Cinn_res = 0.05 / np.tan(np.pi * 3 / Np)

    assert Cinn_res < Cinn_eq
    assert np.isclose(Cinn, Cinn_res)
    assert np.isclose(C, Cinn_res)


def test_select_nc_resolves_equilibrium_and_resistive_constraints():
    """Document select_NC resolution selection with equilibrium width vs explicit delta."""
    params_base = SimulationParams(
        alpha=0.2,
        a=1.0,
        w=0.0,
        delta=None,
    )
    N_base, C_base = select_NC(params_base)

    # Wider current sheet increases zmin, loosening inner limit Cinn
    params_w = SimulationParams(
        alpha=0.2,
        a=1.0,
        w=2.0,
        delta=None,
    )
    N_w, C_w = select_NC(params_w)
    Cinn_base, _, _ = select_C_for_N(N_base, params_base)
    Cinn_w, _, _ = select_C_for_N(N_base, params_w)
    assert Cinn_w > Cinn_base

    # Explicit small delta tightens inner limit Cinn, forcing higher N or lower C
    params_res = SimulationParams(
        alpha=0.2,
        a=1.0,
        w=0.0,
        delta=0.005,
        n_resistivity=5,
    )
    N_res, C_res = select_NC(params_res)
    assert N_res > N_base
    assert C_res < C_base

    # Verify C_out(N) <= C_in(N) at chosen N
    Cinn_chosen, Cout_chosen, _ = select_C_for_N(N_res, params_res)
    assert Cout_chosen <= Cinn_chosen

