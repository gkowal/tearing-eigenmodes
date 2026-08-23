import pytest
import numpy as np
from psecas import ChebyshevRationalGrid
from tearing_eigenmodes.analysis import (
    inner_layer_thickness,
    find_peak_location,
    extract_central_dominance_scale,
    minimum_eigenmode_scale,
    measure_eigenmode_scales,
    MODE_SCALE_SCHEMA_VERSION,
    CLASSICAL_GRID_SCALE_KEYS,
    CGL_GRID_SCALE_KEYS,
)

class MockSystem:
    def __init__(self, grid, a=1.0, w=0.0, S=1.0, kx=1.0, Bx=None, Ux=None, shear=False, xi=0.0, Pr=0.0):
        self.model = "classical"
        self.grid = grid
        self.a = a
        self.w = w
        self.S = S
        self.kx = kx
        self.Bx = Bx if Bx is not None else np.tanh(grid.zg / a)
        self.Ux = Ux if Ux is not None else np.zeros_like(grid.zg)
        self.shear = shear
        self.ξ = xi
        self.Pr = Pr
        self.η = 1.0 / S if S > 0 else 0.0
        self.ν = Pr / S if S > 0 else 0.0
        self.result = {}

def test_inner_layer_thickness_sign_change():
    grid = ChebyshevRationalGrid(N=64, C=1.0, max_derivative_order=4)
    system = MockSystem(grid, a=1.0, w=0.0, S=1.0, kx=1.0)
    system.result['sigma'] = 0.05 + 0.0j
    system.result['dbz'] = np.ones_like(grid.zg)
    system.result['duz'] = 2.0 * np.tanh(grid.zg)

    delta, nin, nwa = inner_layer_thickness(system, δtol=1e-5)

    assert delta > 0.0
    assert isinstance(nin, int)
    assert isinstance(nwa, int)
    assert nin > 0
    assert nwa > 0

def test_inner_layer_thickness_no_sign_change():
    grid = ChebyshevRationalGrid(N=64, C=1.0, max_derivative_order=4)
    system = MockSystem(grid, a=1.0, w=0.0, S=1e6, kx=1.0)
    system.result['sigma'] = 0.05 + 0.0j
    system.result['dbz'] = np.zeros_like(grid.zg)
    system.result['duz'] = 10.0 * np.ones_like(grid.zg)

    delta, nin, nwa = inner_layer_thickness(system)

    assert delta == 0.0
    assert nin == 1


def test_inner_layer_thickness_compatibility_wrapper_vs_minimum():
    # Construct a system where viscous scale is smaller than resistive scale
    grid = ChebyshevRationalGrid(N=64, C=1.0, max_derivative_order=4)
    system = MockSystem(grid, a=1.0, w=0.0, S=1e4, Pr=10.0, kx=0.5)
    system.result['sigma'] = 0.03 + 0.0j
    system.result['duz'] = np.exp(-(grid.zg / 0.1)**2)
    system.result['dbz'] = np.exp(-(grid.zg / 0.4)**2)

    scales = measure_eigenmode_scales(system)
    min_scale, min_key = minimum_eigenmode_scale(scales, model="classical")
    delta_res, nin_res, nwa_res = inner_layer_thickness(system)

    # Prove wrapper returns specifically the resistive scale, not the global minimum
    res_scale = scales["classical.bz_induction.eta_vs_ideal"]
    assert np.isclose(delta_res, res_scale, rtol=1e-6)
    if min_key != "classical.bz_induction.eta_vs_ideal" and min_scale is not None:
        assert delta_res != min_scale

def test_find_peak_location():
    grid = ChebyshevRationalGrid(N=64, C=1.0, max_derivative_order=2)

    # Create a clean Gaussian peak at z = 0.5 for u0
    # u0 = exp(-((z - 0.5) / 0.1)^2)
    u0 = np.exp(-((grid.zg - 0.5) / 0.1)**2)
    b0 = np.zeros_like(grid.zg)

    z_peak, n = find_peak_location(u0, b0, grid, a=1.0, w=0.0, ztol=1e-4)

    # The peak should be extremely close to 0.5
    assert np.isclose(z_peak, 0.5, atol=1e-3)
    assert isinstance(n, int)
    assert n > 0


def test_schema_constants():
    assert MODE_SCALE_SCHEMA_VERSION == 1
    assert isinstance(CLASSICAL_GRID_SCALE_KEYS, tuple)
    assert isinstance(CGL_GRID_SCALE_KEYS, tuple)
    assert len(CLASSICAL_GRID_SCALE_KEYS) == 8
    assert len(CGL_GRID_SCALE_KEYS) == 4
    for key in CLASSICAL_GRID_SCALE_KEYS:
        assert key.startswith("classical.")
    for key in CGL_GRID_SCALE_KEYS:
        assert key.startswith("cgl.")


def test_extract_central_dominance_clean_crossing():
    # Grid and known scale
    z = np.linspace(-2.0, 2.0, 501)
    delta_known = 0.35
    # T_num is 1 inside and decays; T_ref = |z| / delta_known
    T_num = np.exp(-(z / delta_known)**2)
    T_ref = np.abs(z) / delta_known

    scale = extract_central_dominance_scale(z, T_num, T_ref, z_max=2.0)
    # At z = 0.35, T_num = exp(-1) ≈ 0.3679, T_ref = 1.0; crossing occurs near z ~ 0.28
    # Exact crossing: exp(-(z/d)^2) = z/d => let x = z/d, exp(-x^2) = x => x ~ 0.65291864
    x_exact = 0.6529186404192047
    expected_scale = x_exact * delta_known
    assert np.isclose(scale, expected_scale, atol=1e-3)


def test_extract_central_dominance_chebyshev_grid():
    grid = ChebyshevRationalGrid(N=128, C=0.5, max_derivative_order=2)
    z = grid.zg
    delta_known = 0.2
    x_exact = 0.6529186404192047
    expected = x_exact * delta_known

    T_num = np.exp(-(z / delta_known)**2)
    T_ref = np.abs(z) / delta_known

    scale = extract_central_dominance_scale(z, T_num, T_ref, z_max=1.0)
    assert np.isclose(scale, expected, atol=5e-3)


def test_extract_central_dominance_no_dominance_at_center():
    # T_num < T_ref everywhere, including at center
    z = np.linspace(-1.0, 1.0, 201)
    T_num = 0.1 * np.abs(z)
    T_ref = 1.0 + np.abs(z)

    scale = extract_central_dominance_scale(z, T_num, T_ref, z_max=1.0)
    assert np.isinf(scale)


def test_extract_central_dominance_dominates_whole_interval():
    # T_num >= T_ref everywhere on [-1, 1]
    z = np.linspace(-1.0, 1.0, 201)
    T_num = 10.0 * np.ones_like(z)
    T_ref = 1.0 * np.ones_like(z)

    scale = extract_central_dominance_scale(z, T_num, T_ref, z_max=1.0)
    assert np.isinf(scale)


def test_extract_central_dominance_multiple_crossings():
    # Connected to center: dominates from 0 to 0.3, drops below, then rises again at 0.8
    z = np.linspace(-1.5, 1.5, 301)
    # Construct T_num - T_ref = (0.3 - |z|) * (|z| - 0.7)
    # Near 0: (0.3)*(-0.7) = -0.21 -> wait, we want T_num >= T_ref near 0!
    # So construct T_num and T_ref directly:
    # Dominance in [0, 0.3] and [0.8, 1.2]
    T_ref = np.ones_like(z)
    T_num = np.where(np.abs(z) <= 0.3, 2.0, 0.5)
    # Add an off-centre island of dominance at [0.8, 1.2]
    T_num = np.where((np.abs(z) >= 0.8) & (np.abs(z) <= 1.2), 3.0, T_num)

    scale = extract_central_dominance_scale(z, T_num, T_ref, z_max=1.5)
    # The scale should be 0.3 (the first crossing connected to center), not 1.2
    assert np.isclose(scale, 0.3, atol=0.02)


def test_extract_central_dominance_disconnected_offcentre_island():
    # Dominance only at [0.6, 0.9], but not at center
    z = np.linspace(-1.0, 1.0, 201)
    T_ref = np.ones_like(z)
    T_num = np.where((np.abs(z) >= 0.6) & (np.abs(z) <= 0.9), 2.0, 0.2)

    scale = extract_central_dominance_scale(z, T_num, T_ref, z_max=1.0)
    # Since center is not dominant, this is an off-center island -> returns inf
    assert np.isinf(scale)


def test_extract_central_dominance_exact_zero_at_resonance():
    z = np.linspace(-1.0, 1.0, 201)
    # T_ref has exact 0 at z=0 (e.g. F = tanh(z))
    T_ref = np.abs(np.tanh(z))
    T_num = 0.5 * np.ones_like(z)  # nonideal term is constant 0.5

    scale = extract_central_dominance_scale(z, T_num, T_ref, z_max=1.0)
    # Crossing when tanh(z) = 0.5 => z = atanh(0.5) ≈ 0.549306
    expected = np.arctanh(0.5)
    assert np.isclose(scale, expected, atol=1e-3)


def test_extract_central_dominance_symmetry_and_asymmetry():
    z = np.linspace(-2.0, 2.0, 401)
    # Symmetric crossing at 0.4
    T_ref = np.abs(z)
    T_num = 0.4 * np.ones_like(z)
    scale_sym = extract_central_dominance_scale(z, T_num, T_ref, z_max=2.0)
    assert np.isclose(scale_sym, 0.4, atol=1e-3)

    # Asymmetric: positive side crosses at 0.3, negative side crosses at 0.5
    T_num_asym = np.where(z >= 0, 0.3, 0.5)
    scale_asym = extract_central_dominance_scale(z, T_num_asym, T_ref, z_max=2.0)
    # Should take the conservative minimum: 0.3
    assert np.isclose(scale_asym, 0.3, atol=1e-3)


def test_extract_central_dominance_invalid_and_noisy():
    z = np.linspace(-1.0, 1.0, 101)
    # All zeros / below activity floor
    scale_zero = extract_central_dominance_scale(z, np.zeros_like(z), np.zeros_like(z))
    assert np.isnan(scale_zero)

    # NaN in inputs
    T_nan = np.ones_like(z)
    T_nan[10] = np.nan
    scale_nan = extract_central_dominance_scale(z, T_nan, np.ones_like(z))
    assert np.isnan(scale_nan)

    # Empty array
    scale_empty = extract_central_dominance_scale(np.array([]), np.array([]), np.array([]))
    assert np.isnan(scale_empty)


def test_minimum_eigenmode_scale_selection():
    # Classical test
    scales = {
        "classical.bz_induction.eta_vs_ideal": 0.05,
        "classical.bz_induction.eta_vs_f": 0.08,
        "classical.bz_induction.g_vs_f": np.nan,
        "classical.bz_induction.xi_vs_f": np.inf,
        "classical.uz_vorticity.nu_vs_ideal": 0.03,
        "classical.uz_vorticity.nu_vs_f": 0.04,
        "classical.uz_vorticity.g_vs_f": np.nan,
        "classical.uz_vorticity.xi_vs_f": np.nan,
    }

    min_scale, min_key = minimum_eigenmode_scale(scales, model="classical")
    assert min_scale == 0.03
    assert min_key == "classical.uz_vorticity.nu_vs_ideal"

    # All nan/inf test
    all_invalid = {k: np.nan for k in CLASSICAL_GRID_SCALE_KEYS}
    all_invalid["classical.bz_induction.xi_vs_f"] = np.inf
    s, k = minimum_eigenmode_scale(all_invalid, model="classical")
    assert s is None
    assert k is None

    # CGL model test
    cgl_scales = {
        "cgl.by_induction.eta_vs_ideal": 0.06,
        "cgl.by_induction.eta_vs_f": 0.07,
        "cgl.bz_induction.eta_vs_ideal": 0.02,
        "cgl.bz_induction.eta_vs_f": 0.025,
    }
    s_cgl, k_cgl = minimum_eigenmode_scale(cgl_scales, model="cgl")
    assert s_cgl == 0.02
    assert k_cgl == "cgl.bz_induction.eta_vs_ideal"

    # Tie breaking test (deterministic key order)
    tie_scales = {
        "classical.bz_induction.eta_vs_ideal": 0.05,
        "classical.bz_induction.eta_vs_f": 0.05,
    }
    s_tie, k_tie = minimum_eigenmode_scale(tie_scales, model="classical")
    assert s_tie == 0.05
    assert k_tie == "classical.bz_induction.eta_vs_ideal"


def test_extract_central_dominance_normalization_invariance():
    """Dominance crossing must remain invariant under very small and large complex scalings."""
    z = np.linspace(-2.0, 2.0, 401)
    # Known profile crossing reference at |z| = 0.4 where T_num(0.4) = 0.5 = T_ref
    T_num = 0.5**((z / 0.4)**2)
    T_ref = 0.5 * np.ones_like(z)
    L_lhs = np.ones_like(z)

    # 1. Base crossing
    scale_base = extract_central_dominance_scale(z, T_num, T_ref, z_max=2.0, L_lhs=L_lhs)
    assert np.isclose(scale_base, 0.4, atol=1e-3)

    # 2. Very small scaling (1e-20)
    scale_tiny = extract_central_dominance_scale(
        z, 1e-20 * T_num, 1e-20 * T_ref, z_max=2.0, L_lhs=1e-20 * L_lhs
    )
    assert np.isclose(scale_tiny, 0.4, atol=1e-3)

    # 3. Large complex scaling factor ((3-4j)*1e15)
    c_factor = (3.0 - 4.0j) * 1e15
    scale_large = extract_central_dominance_scale(
        z, c_factor * T_num, c_factor * T_ref, z_max=2.0, L_lhs=c_factor * L_lhs
    )
    assert np.isclose(scale_large, 0.4, atol=1e-3)


def test_extract_central_dominance_equation_local_floor():
    """Profiles entirely below equation-local activity floor must return nan."""
    z = np.linspace(-1.0, 1.0, 101)
    L_lhs = np.ones_like(z) * 1.0
    # num and ref are order 1e-18, while LHS is 1.0 -> eps_q ~ 100 * eps_mach * 1.0 ~ 2e-14
    T_num = np.ones_like(z) * 1e-18
    T_ref = np.ones_like(z) * 1e-18
    scale = extract_central_dominance_scale(z, T_num, T_ref, z_max=1.0, L_lhs=L_lhs)
    assert np.isnan(scale)


def test_extract_central_dominance_subfloor_numerator_noise_rejection():
    """Sub-floor numerator noise must return nan even if reference profile is active."""
    z = np.linspace(-1.0, 1.0, 2001)
    T_num = np.full_like(z, 1.0e-20)
    T_ref = np.abs(z)
    L_lhs = np.ones_like(z)

    scale = extract_central_dominance_scale(z, T_num, T_ref, z_max=1.0, L_lhs=L_lhs)
    assert np.isnan(scale)


def test_extract_central_dominance_out_of_window_activity_rejection():
    """Activity outside z_max must not validate sub-floor noise inside z_max."""
    z = np.linspace(-5.0, 5.0, 1001)
    # Inside [-1, 1], profiles are 1e-20 (sub-floor compared to L_lhs=1.0)
    # Outside [-1, 1], profiles are large (1.0)
    T_num = np.where(np.abs(z) <= 1.0, 1e-20, 1.0)
    T_ref = np.where(np.abs(z) <= 1.0, np.abs(z) * 1e-20, np.abs(z))
    L_lhs = np.ones_like(z)

    # z_max = 1.0: within window, numerator is sub-floor relative to LHS -> returns nan
    scale = extract_central_dominance_scale(z, T_num, T_ref, z_max=1.0, L_lhs=L_lhs)
    assert np.isnan(scale)


def test_extract_central_dominance_neutral_core_first_active_reference_dominant():
    """A neutral sub-floor core where first active sample is reference-dominant must return inf."""
    z = np.linspace(-2.0, 2.0, 401)
    # Central core |z| < 0.1 has sub-floor noise (1e-20)
    # For |z| >= 0.1, reference term is active (0.5) and dominates numerator (0.1)
    T_num = np.where(np.abs(z) < 0.1, 1e-20, 0.1)
    T_ref = np.where(np.abs(z) < 0.1, 1e-20, 0.5)
    L_lhs = np.ones_like(z)

    scale = extract_central_dominance_scale(z, T_num, T_ref, z_max=2.0, L_lhs=L_lhs)
    assert np.isinf(scale)


def test_measure_eigenmode_scales_unsupported_model_rejection():
    """measure_eigenmode_scales must raise NotImplementedError for unknown equation systems."""
    class UnknownSystem:
        def __init__(self):
            self.result = {"duz": np.zeros(10), "dbz": np.zeros(10), "sigma": 1.0}

    sys = UnknownSystem()
    with pytest.raises(NotImplementedError, match="Unsupported system"):
        measure_eigenmode_scales(sys)

    with pytest.raises(NotImplementedError, match="Unsupported model"):
        measure_eigenmode_scales(sys, model="relativistic_mhd")
