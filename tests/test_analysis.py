import pytest
import numpy as np
from psecas import ChebyshevRationalGrid
from tearing_eigenmodes.analysis import (
    inner_layer_thickness,
    find_peak_location,
    extract_central_dominance_scale,
    minimum_eigenmode_scale,
    MODE_SCALE_SCHEMA_VERSION,
    CLASSICAL_GRID_SCALE_KEYS,
    CGL_GRID_SCALE_KEYS,
)

class MockSystem:
    def __init__(self, grid, a=1.0, w=0.0, S=1.0, kx=1.0, Bx=1.0, Ux=0.0, shear=False, xi=0.0):
        self.grid = grid
        self.a = a
        self.w = w
        self.S = S
        self.kx = kx
        self.Bx = Bx
        self.Ux = Ux
        self.shear = shear
        self.ξ = xi
        self.result = {}

def test_inner_layer_thickness_sign_change():
    grid = ChebyshevRationalGrid(N=64, C=1.0, max_derivative_order=2)
    system = MockSystem(grid, a=1.0, w=0.0, S=1.0, kx=1.0, Bx=1.0)

    # We want Td = Tn - Ti to change sign at z = 0.5
    # Tn = |b" - k^2 b| / S. If b = const 1.0, Tn = |0 - 1.0|/1.0 = 1.0
    # Ti = |1j * kx * a * u * Bx| = |1.0 * u|.
    # If we set u = 2.0 * grid.zg, then Ti = 2.0 * z.
    # Td = 1.0 - 2.0 * z.
    # At z = 0.5, Td = 0.0.
    system.result['dbz'] = np.ones_like(grid.zg)
    system.result['duz'] = 2.0 * grid.zg

    delta, nin, nwa = inner_layer_thickness(system, δtol=1e-5)

    # delta should be very close to 0.5
    assert np.isclose(delta, 0.5, atol=2e-2)
    assert isinstance(nin, int)
    assert isinstance(nwa, int)
    assert nin > 0
    assert nwa > 0

def test_inner_layer_thickness_no_sign_change():
    grid = ChebyshevRationalGrid(N=64, C=1.0, max_derivative_order=2)
    system = MockSystem(grid, a=1.0, w=0.0, S=1.0, kx=1.0, Bx=1.0)

    # Set fields such that Td = Tn - Ti is always negative
    # Tn = 1.0, Ti = 10.0
    system.result['dbz'] = np.ones_like(grid.zg)
    system.result['duz'] = 10.0 * np.ones_like(grid.zg)

    delta, nin, nwa = inner_layer_thickness(system)

    # delta should be 0.0 since there is no region where Td >= 0
    assert delta == 0.0
    assert nin == 1  # max(1, I[0].size) where I is zg <= 0.0 -> zg=0.0 is 1 point

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
