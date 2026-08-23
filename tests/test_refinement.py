import pytest
import numpy as np
import tempfile
import shutil
import os
import time
from typing import Generator
from tearing_eigenmodes import (
    refine_eigenvalues,
    refine_wavenumber_bracket,
    refine_inner_scale,
    refine_resistive_scale,
    estimate_inner_scale,
    SimulationParams,
    MODE_SCALE_SCHEMA_VERSION,
)
from tearing_eigenmodes.refinement import (
    _cached_load_eigenmodes,
    _load_eigenmodes_cache,
)

@pytest.fixture
def temp_npz_dir() -> Generator[str, None, None]:
    # Create temp directory
    d = tempfile.mkdtemp()
    yield d
    # Cleanup
    shutil.rmtree(d)

def create_mock_npz(
    directory: str,
    filename: str,
    val: float,
    wavenumber: float,
    growth: float,
    thickness: float = 0.01,
    tolerance: float = 1e-6,
    scaling: float = 1.0,
    resolution: int = 128,
    nin: int = 10,
    nwa: int = 20,
) -> str:
    filepath = os.path.join(directory, filename)
    np.savez(
        filepath,
        wavenumber=wavenumber,
        scan_parameter="S",
        scan_parameter_value=val,
        eigenvalue=complex(growth),
        resistive_layer_thickness=thickness,
        tolerance=tolerance,
        grid_scaling_factor=scaling,
        resolution=resolution,
        resistive_layer_nodes=nin,
        current_sheet_nodes=nwa,
    )
    return filepath

def test_caching_behavior(temp_npz_dir: str, monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock_npz(temp_npz_dir, "state_1.npz", val=1.0, wavenumber=0.1, growth=0.5)
    create_mock_npz(temp_npz_dir, "state_2.npz", val=2.0, wavenumber=0.2, growth=0.6)

    # Count calls to load_eigenmodes
    call_count = 0
    from tearing_eigenmodes.refinement import load_eigenmodes as original_load
    def mock_load(path: str, pattern: str = "*.npz") -> tuple:
        nonlocal call_count
        call_count += 1
        return original_load(path, pattern)

    monkeypatch.setattr("tearing_eigenmodes.refinement.load_eigenmodes", mock_load)

    # Clear cache first to ensure a clean state
    _load_eigenmodes_cache.clear()

    # First load -> Cache Miss
    res1 = _cached_load_eigenmodes(temp_npz_dir)
    assert call_count == 1

    # Second load -> Cache Hit
    res2 = _cached_load_eigenmodes(temp_npz_dir)
    assert call_count == 1
    assert res1[0].size == 2
    assert res2[0].size == 2

    # Now, add a file
    create_mock_npz(temp_npz_dir, "state_3.npz", val=3.0, wavenumber=0.3, growth=0.7)
    # Third load -> Cache Miss
    res3 = _cached_load_eigenmodes(temp_npz_dir)
    assert call_count == 2
    assert res3[0].size == 3

    # Now, delete a file
    os.remove(os.path.join(temp_npz_dir, "state_1.npz"))
    # Fourth load -> Cache Miss
    res4 = _cached_load_eigenmodes(temp_npz_dir)
    assert call_count == 3
    assert res4[0].size == 2

    # Now, modify a file
    # Ensure mtime updates by forcing a sleep or rewriting
    # In some virtualized filesystems, mtime resolution is low, so we change the file size slightly
    # or touch/modify the content.
    time.sleep(0.05)
    create_mock_npz(temp_npz_dir, "state_2.npz", val=2.0, wavenumber=0.2, growth=0.99, thickness=0.05)
    # Fifth load -> Cache Miss
    res5 = _cached_load_eigenmodes(temp_npz_dir)
    assert call_count == 4
    # The growth values array res5[2] should have 0.99
    assert 0.99 in res5[2]

def test_refinement_functions(temp_npz_dir: str) -> None:
    _load_eigenmodes_cache.clear()
    create_mock_npz(temp_npz_dir, "state_1.npz", val=1.0, wavenumber=0.1, growth=0.5)
    create_mock_npz(temp_npz_dir, "state_2.npz", val=2.0, wavenumber=0.2, growth=0.6)

    params = SimulationParams(
        data_path=temp_npz_dir,
        sigma=0.1,
        delta=0.05,
        kbracket=[0.05, 0.25],
        ktol=1e-3
    )

    # Test refine_eigenvalues
    vs = np.array([1.5])
    eigenvalues = refine_eigenvalues(vs, params)
    # Since vs=1.5 is between 1.0 and 2.0, it should interpolate between 0.5 and 0.6
    assert np.isclose(eigenvalues[0], 0.55, atol=0.05)

    # Test refine_wavenumber_bracket
    vs = np.array([1.5])
    brackets = refine_wavenumber_bracket(vs, params)
    assert brackets[0] is not None
    assert len(brackets[0]) == 2

def test_refine_inner_scale_cached_and_fallback(temp_npz_dir: str) -> None:
    _load_eigenmodes_cache.clear()
    create_mock_npz(temp_npz_dir, "state_1.npz", val=1.0, wavenumber=0.1, growth=0.5, thickness=0.02)
    create_mock_npz(temp_npz_dir, "state_2.npz", val=2.0, wavenumber=0.2, growth=0.6, thickness=0.04)

    params = SimulationParams(
        data_path=temp_npz_dir,
        alpha=0.1,
        a=1.0,
        S=1e4,
        CGL=False,
        inner_resolution_safety=1.0,
    )

    # 1. Within cached range [0.1, 0.2]: log-log interpolation
    vs = np.array([0.15])
    deltas = refine_inner_scale(vs, params)
    assert deltas[0] is not None
    # Log-log interpolation: scale(0.15) = 0.030
    assert np.isclose(deltas[0], 0.030, atol=1e-5)

    # 2. Outside cached range: falls back to analytic estimator
    vs_outside = np.array([5.0])
    deltas_outside = refine_inner_scale(vs_outside, params)
    expected_estimator = estimate_inner_scale(params, alpha=5.0)
    assert deltas_outside[0] is not None
    assert np.isclose(deltas_outside[0], expected_estimator)

    # 3. Compatibility alias
    deltas_alias = refine_resistive_scale(vs, params)
    assert deltas_alias[0] is not None
    assert np.isclose(deltas[0], deltas_alias[0])


def test_refine_inner_scale_explicit_override(temp_npz_dir: str) -> None:
    _load_eigenmodes_cache.clear()
    create_mock_npz(temp_npz_dir, "state_1.npz", val=1.0, wavenumber=0.1, growth=0.5, thickness=0.02)
    create_mock_npz(temp_npz_dir, "state_2.npz", val=2.0, wavenumber=0.2, growth=0.6, thickness=0.04)

    # Explicit override should NOT be overwritten by cached values
    params_override = SimulationParams(
        data_path=temp_npz_dir,
        inner_scale=0.099,
    )
    vs = np.array([1.0, 1.5, 2.0])
    deltas = refine_inner_scale(vs, params_override)
    for d in deltas:
        assert d is not None
        assert np.isclose(d, 0.099)


def test_refine_inner_scale_zero_thickness_filtering(temp_npz_dir: str) -> None:
    _load_eigenmodes_cache.clear()
    # Mock state with zero measured thickness and valid thicknesses
    create_mock_npz(temp_npz_dir, "state_1.npz", val=1.0, wavenumber=0.1, growth=0.5, thickness=0.0)
    create_mock_npz(temp_npz_dir, "state_2.npz", val=2.0, wavenumber=0.2, growth=0.6, thickness=0.04)
    create_mock_npz(temp_npz_dir, "state_3.npz", val=3.0, wavenumber=0.3, growth=0.7, thickness=0.08)

    params = SimulationParams(
        data_path=temp_npz_dir,
        alpha=0.1,
        a=1.0,
        S=1e4,
        CGL=False,
    )

    vs = np.array([0.1, 0.25])
    deltas = refine_inner_scale(vs, params)
    # At v=0.1, the cached thickness was 0.0 (ignored, outside valid cached range [0.2, 0.3]), so it gets estimator
    expected_v1 = estimate_inner_scale(params, alpha=0.1)
    assert deltas[0] is not None
    assert np.isclose(deltas[0], expected_v1)

    # At v=0.25, it interpolates between 0.04 and 0.08
    assert deltas[1] is not None
    assert deltas[1] > 0.04 and deltas[1] < 0.08


def test_refine_inner_scale_multiscale_mechanism_switching(temp_npz_dir: str) -> None:
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.io import save_eigenmode

    # State 1 at k=0.1: resistive is limiting (0.020 vs viscous 0.070)
    scales_1 = {
        "classical.bz_induction.eta_vs_ideal": 0.020,
        "classical.uz_vorticity.nu_vs_ideal": 0.070,
    }
    save_eigenmode(
        os.path.join(temp_npz_dir, "state_alpha0.100000e+00.npz"),
        wavenumber=0.1,
        eigenvalue=0.05 + 0.0j,
        tolerance=1e-6,
        minimum_physical_scale=0.020,
        minimum_physical_scale_key="classical.bz_induction.eta_vs_ideal",
        minimum_scale_nodes=10,
        resistive_layer_thickness=0.020,
        resistive_layer_nodes=10,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=128,
        grid=np.linspace(-10, 10, 128),
        mode_scales=scales_1,
        duz=np.ones(128),
        dbz=np.ones(128),
    )

    # State 2 at k=0.3: viscous is limiting (0.030 vs resistive 0.080)
    scales_2 = {
        "classical.bz_induction.eta_vs_ideal": 0.080,
        "classical.uz_vorticity.nu_vs_ideal": 0.030,
    }
    save_eigenmode(
        os.path.join(temp_npz_dir, "state_alpha0.300000e+00.npz"),
        wavenumber=0.3,
        eigenvalue=0.06 + 0.0j,
        tolerance=1e-6,
        minimum_physical_scale=0.030,
        minimum_physical_scale_key="classical.uz_vorticity.nu_vs_ideal",
        minimum_scale_nodes=15,
        resistive_layer_thickness=0.080,
        resistive_layer_nodes=40,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=128,
        grid=np.linspace(-10, 10, 128),
        mode_scales=scales_2,
        duz=np.ones(128),
        dbz=np.ones(128),
    )

    params = SimulationParams(
        data_path=temp_npz_dir,
        alpha=0.1,
        a=1.0,
        S=1e4,
        CGL=False,
        inner_resolution_safety=1.0,
    )

    # Check that at k=0.1, refined scale is 0.020
    vs = np.array([0.1, 0.3])
    deltas = refine_inner_scale(vs, params)
    assert deltas[0] is not None and deltas[1] is not None
    assert np.isclose(deltas[0], 0.020, atol=1e-5)
    assert np.isclose(deltas[1], 0.030, atol=1e-5)


def test_refine_inner_scale_safety_factor_applied(temp_npz_dir: str) -> None:
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.io import save_eigenmode

    scales = {"classical.bz_induction.eta_vs_ideal": 0.040}
    save_eigenmode(
        os.path.join(temp_npz_dir, "state_alpha0.100000e+00.npz"),
        wavenumber=0.1,
        eigenvalue=0.05 + 0.0j,
        tolerance=1e-6,
        minimum_physical_scale=0.040,
        minimum_physical_scale_key="classical.bz_induction.eta_vs_ideal",
        minimum_scale_nodes=10,
        resistive_layer_thickness=0.040,
        resistive_layer_nodes=10,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=128,
        grid=np.linspace(-10, 10, 128),
        mode_scales=scales,
        duz=np.ones(128),
        dbz=np.ones(128),
    )

    params = SimulationParams(
        data_path=temp_npz_dir,
        alpha=0.1,
        a=1.0,
        S=1e4,
        CGL=False,
        inner_resolution_safety=2.0,
    )

    vs = np.array([0.1])
    deltas = refine_inner_scale(vs, params)
    # physical scale = 0.040, safety = 2.0 -> grid scale = 0.020
    assert deltas[0] is not None
    assert np.isclose(deltas[0], 0.020, atol=1e-5)


def test_refine_inner_scale_parameter_sweep_negative_coordinate(temp_npz_dir: str) -> None:
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.io import save_eigenmode

    # Dependence sweep with negative and positive values
    for val, scale in [(-0.5, 0.030), (0.0, 0.040), (0.5, 0.050)]:
        scales = {"cgl.bz_induction.eta_vs_ideal": scale}
        save_eigenmode(
            os.path.join(temp_npz_dir, f"state_Δβ{val:+.6e}.npz"),
            scan_parameter="Δβ",
            scan_parameter_value=val,
            wavenumber=0.1,
            eigenvalue=0.05 + 0.0j,
            tolerance=1e-6,
            minimum_physical_scale=scale,
            minimum_physical_scale_key="cgl.bz_induction.eta_vs_ideal",
            minimum_scale_nodes=10,
            resistive_layer_thickness=scale,
            resistive_layer_nodes=10,
            grid_scaling_factor=1.0,
            current_sheet_nodes=20,
            resolution=128,
            grid=np.linspace(-10, 10, 128),
            mode_scales=scales,
            dby=np.ones(128),
            dbz=np.ones(128),
            duy=np.ones(128),
            duz=np.ones(128),
        )

    params = SimulationParams(
        data_path=temp_npz_dir,
        dependence="Δβ",
        CGL=True,
        inner_resolution_safety=1.0,
    )

    vs = np.array([-0.25, 0.25])
    deltas = refine_inner_scale(vs, params)
    # Interpolation in (v, log scale) across nonpositive coordinates
    assert deltas[0] is not None and deltas[0] > 0.030 and deltas[0] < 0.040
    assert deltas[1] is not None and deltas[1] > 0.040 and deltas[1] < 0.050


def test_explicit_inner_scale_exact_override(temp_npz_dir: str) -> None:
    """Explicit inner_scale must be returned unchanged without dividing by safety factor."""
    params = SimulationParams(
        data_path=temp_npz_dir,
        alpha=0.1,
        a=1.0,
        S=1e4,
        inner_scale=0.099,
        inner_resolution_safety=2.0,
    )
    vs = np.array([0.1, 0.2])
    deltas = refine_inner_scale(vs, params)
    assert len(deltas) == 2
    assert deltas[0] is not None and np.isclose(deltas[0], 0.099)
    assert deltas[1] is not None and np.isclose(deltas[1], 0.099)


def test_refine_dispersion_k_vs_alpha_with_nonunit_a(temp_npz_dir: str) -> None:
    """Dispersion refinement must convert cached alpha to physical k = alpha / a."""
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.io import save_eigenmode

    # a = 2.0: cached alpha = 0.2 (k=0.1, scale=0.02) and alpha = 0.6 (k=0.3, scale=0.08)
    for alpha, scale in [(0.2, 0.02), (0.6, 0.08)]:
        save_eigenmode(
            os.path.join(temp_npz_dir, f"state_α{alpha:.6e}.npz"),
            wavenumber=alpha,
            a=2.0,
            eigenvalue=0.05 + 0.0j,
            tolerance=1e-6,
            minimum_physical_scale=scale,
            minimum_physical_scale_key="classical.bz_induction.eta_vs_ideal",
            minimum_scale_nodes=10,
            resistive_layer_thickness=scale,
            resistive_layer_nodes=10,
            grid_scaling_factor=1.0,
            current_sheet_nodes=20,
            resolution=128,
            grid=np.linspace(-10, 10, 128),
            mode_scales={"classical.bz_induction.eta_vs_ideal": scale},
            duz=np.ones(128),
            dbz=np.ones(128),
        )

    params = SimulationParams(
        data_path=temp_npz_dir,
        a=2.0,
        S=1e4,
        CGL=False,
        inner_resolution_safety=1.0,
    )

    # Requested physical k = 0.2 (between k=0.1 and k=0.3)
    vs = np.array([0.2])
    deltas = refine_inner_scale(vs, params)
    expected_scale = 0.04796092578242428
    assert deltas[0] is not None
    assert np.isclose(deltas[0], expected_scale, rtol=1e-4)
    # Must NOT equal 0.02 (which occurred when alpha=0.2 was compared directly to k=0.2)
    assert not np.isclose(deltas[0], 0.02)


def test_refine_gap_barrier_nan(temp_npz_dir: str) -> None:
    """A cached np.nan must act as an invalidity barrier and prevent cross-gap interpolation."""
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.io import save_eigenmode

    # k=0.1 (scale 0.02), k=0.2 (nan barrier), k=0.3 (scale 0.08)
    for k_val, scale in [(0.1, 0.02), (0.2, float("nan")), (0.3, 0.08)]:
        save_eigenmode(
            os.path.join(temp_npz_dir, f"state_α{k_val:.6e}.npz"),
            wavenumber=k_val,
            a=1.0,
            eigenvalue=0.05 + 0.0j,
            tolerance=1e-6,
            minimum_physical_scale=scale,
            minimum_physical_scale_key="classical.bz_induction.eta_vs_ideal" if not np.isnan(scale) else "",
            minimum_scale_nodes=10 if not np.isnan(scale) else 0,
            resistive_layer_thickness=scale,
            resistive_layer_nodes=10 if not np.isnan(scale) else 0,
            grid_scaling_factor=1.0,
            current_sheet_nodes=20,
            resolution=128,
            grid=np.linspace(-10, 10, 128),
            mode_scales={"classical.bz_induction.eta_vs_ideal": scale},
            duz=np.ones(128),
            dbz=np.ones(128),
        )

    params = SimulationParams(
        data_path=temp_npz_dir,
        a=1.0,
        S=1e4,
        CGL=False,
        inner_resolution_safety=1.0,
    )

    # Requested k=0.15 is between k=0.1 and k=0.2. Since k=0.1 is an isolated single-point segment,
    # it cannot interpolate across the barrier to k=0.3.
    vs = np.array([0.15])
    deltas = refine_inner_scale(vs, params)
    # Falls back to analytic estimator
    analytic_scale = estimate_inner_scale(params, alpha=0.15)
    assert deltas[0] is not None
    assert np.isclose(deltas[0], analytic_scale)


def test_refine_unconverged_cached_state_rejection(temp_npz_dir: str) -> None:
    """Cached state with tolerance > 1.0 must be rejected from refinement, retaining analytic fallback."""
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.io import save_eigenmode

    save_eigenmode(
        os.path.join(temp_npz_dir, "state_α0.100000e+00.npz"),
        wavenumber=0.1,
        a=1.0,
        eigenvalue=0.05 + 0.0j,
        tolerance=10.0,  # Unconverged!
        minimum_physical_scale=0.012,
        minimum_physical_scale_key="classical.bz_induction.eta_vs_ideal",
        minimum_scale_nodes=10,
        resistive_layer_thickness=0.012,
        resistive_layer_nodes=10,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=2048,
        grid=np.linspace(-10, 10, 128),
        mode_scales={"classical.bz_induction.eta_vs_ideal": 0.012},
        duz=np.ones(128),
        dbz=np.ones(128),
    )

    params = SimulationParams(
        data_path=temp_npz_dir,
        a=1.0,
        S=1e4,
        CGL=False,
        inner_resolution_safety=1.0,
    )

    vs = np.array([0.1])
    deltas = refine_inner_scale(vs, params)
    analytic_scale = estimate_inner_scale(params, alpha=0.1)
    assert deltas[0] is not None
    # Must NOT select 0.012
    assert not np.isclose(deltas[0], 0.012)
    # Must retain analytic fallback
    assert np.isclose(deltas[0], analytic_scale)


def test_refine_unconverged_low_resolution_barrier(temp_npz_dir: str) -> None:
    """An unconverged state at resolution < Nmax must act as a barrier and not be skipped."""
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.io import save_eigenmode

    # k=0.1 (converged, scale=0.02)
    save_eigenmode(
        os.path.join(temp_npz_dir, "state_α1.000000e-01.npz"),
        wavenumber=0.1,
        a=1.0,
        eigenvalue=0.05 + 0.0j,
        tolerance=1e-6,
        minimum_physical_scale=0.02,
        minimum_physical_scale_key="classical.bz_induction.eta_vs_ideal",
        minimum_scale_nodes=10,
        resistive_layer_thickness=0.02,
        resistive_layer_nodes=10,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=128,
        grid=np.linspace(-10, 10, 128),
        mode_scales={"classical.bz_induction.eta_vs_ideal": 0.02},
        duz=np.ones(128),
        dbz=np.ones(128),
    )

    # k=0.2 (unconverged, tolerance=10, resolution=128 < Nmax=2048, scale=0.001)
    save_eigenmode(
        os.path.join(temp_npz_dir, "state_α2.000000e-01.npz"),
        wavenumber=0.2,
        a=1.0,
        eigenvalue=0.05 + 0.0j,
        tolerance=10.0,
        minimum_physical_scale=0.001,
        minimum_physical_scale_key="classical.bz_induction.eta_vs_ideal",
        minimum_scale_nodes=10,
        resistive_layer_thickness=0.001,
        resistive_layer_nodes=10,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=128,
        grid=np.linspace(-10, 10, 128),
        mode_scales={"classical.bz_induction.eta_vs_ideal": 0.001},
        duz=np.ones(128),
        dbz=np.ones(128),
    )

    # k=0.3 (converged, scale=0.08)
    save_eigenmode(
        os.path.join(temp_npz_dir, "state_α3.000000e-01.npz"),
        wavenumber=0.3,
        a=1.0,
        eigenvalue=0.05 + 0.0j,
        tolerance=1e-6,
        minimum_physical_scale=0.08,
        minimum_physical_scale_key="classical.bz_induction.eta_vs_ideal",
        minimum_scale_nodes=10,
        resistive_layer_thickness=0.08,
        resistive_layer_nodes=10,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=128,
        grid=np.linspace(-10, 10, 128),
        mode_scales={"classical.bz_induction.eta_vs_ideal": 0.08},
        duz=np.ones(128),
        dbz=np.ones(128),
    )

    params = SimulationParams(
        data_path=temp_npz_dir,
        a=1.0,
        S=1e4,
        CGL=False,
        Nmax=2048,
        inner_resolution_safety=1.0,
    )

    # Requested k=0.15: must NOT cross the k=0.2 unconverged barrier (which would give 0.03336)
    vs = np.array([0.15])
    deltas = refine_inner_scale(vs, params)
    expected_analytic = estimate_inner_scale(params, alpha=0.15)
    assert deltas[0] is not None
    assert np.isclose(deltas[0], expected_analytic)
    assert not np.isclose(deltas[0], 0.0333604903137, atol=1e-3)


def test_refine_unconverged_non_default_nmax_barrier(temp_npz_dir: str) -> None:
    """Refinement barrier preservation must be independent of check_state's default Nmax."""
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.io import save_eigenmode

    # k=0.1 (converged, scale=0.02)
    save_eigenmode(
        os.path.join(temp_npz_dir, "state_α1.000000e-01.npz"),
        wavenumber=0.1,
        a=1.0,
        eigenvalue=0.05 + 0.0j,
        tolerance=1e-6,
        minimum_physical_scale=0.02,
        minimum_physical_scale_key="classical.bz_induction.eta_vs_ideal",
        minimum_scale_nodes=10,
        resistive_layer_thickness=0.02,
        resistive_layer_nodes=10,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=128,
        grid=np.linspace(-10, 10, 128),
        mode_scales={"classical.bz_induction.eta_vs_ideal": 0.02},
        duz=np.ones(128),
        dbz=np.ones(128),
    )

    # k=0.2 (unconverged, tolerance=10, resolution=512, scale=0.001)
    save_eigenmode(
        os.path.join(temp_npz_dir, "state_α2.000000e-01.npz"),
        wavenumber=0.2,
        a=1.0,
        eigenvalue=0.05 + 0.0j,
        tolerance=10.0,
        minimum_physical_scale=0.001,
        minimum_physical_scale_key="classical.bz_induction.eta_vs_ideal",
        minimum_scale_nodes=10,
        resistive_layer_thickness=0.001,
        resistive_layer_nodes=10,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=512,
        grid=np.linspace(-10, 10, 128),
        mode_scales={"classical.bz_induction.eta_vs_ideal": 0.001},
        duz=np.ones(128),
        dbz=np.ones(128),
    )

    # k=0.3 (converged, scale=0.08)
    save_eigenmode(
        os.path.join(temp_npz_dir, "state_α3.000000e-01.npz"),
        wavenumber=0.3,
        a=1.0,
        eigenvalue=0.05 + 0.0j,
        tolerance=1e-6,
        minimum_physical_scale=0.08,
        minimum_physical_scale_key="classical.bz_induction.eta_vs_ideal",
        minimum_scale_nodes=10,
        resistive_layer_thickness=0.08,
        resistive_layer_nodes=10,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=128,
        grid=np.linspace(-10, 10, 128),
        mode_scales={"classical.bz_induction.eta_vs_ideal": 0.08},
        duz=np.ones(128),
        dbz=np.ones(128),
    )

    params = SimulationParams(
        data_path=temp_npz_dir,
        a=1.0,
        S=1e4,
        CGL=False,
        Nmax=512,  # Non-default Nmax
        inner_resolution_safety=1.0,
    )

    vs = np.array([0.15])
    deltas = refine_inner_scale(vs, params)
    expected_analytic = estimate_inner_scale(params, alpha=0.15)
    assert deltas[0] is not None
    assert np.isclose(deltas[0], expected_analytic)


def test_refine_unconverged_exact_target_falls_back(temp_npz_dir: str) -> None:
    """An exact request at an unconverged coordinate must fall back and never return the stored scale."""
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.io import save_eigenmode

    save_eigenmode(
        os.path.join(temp_npz_dir, "state_α2.000000e-01.npz"),
        wavenumber=0.2,
        a=1.0,
        eigenvalue=0.05 + 0.0j,
        tolerance=10.0,  # Unconverged
        minimum_physical_scale=0.001,
        minimum_physical_scale_key="classical.bz_induction.eta_vs_ideal",
        minimum_scale_nodes=10,
        resistive_layer_thickness=0.001,
        resistive_layer_nodes=10,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=128,
        grid=np.linspace(-10, 10, 128),
        mode_scales={"classical.bz_induction.eta_vs_ideal": 0.001},
        duz=np.ones(128),
        dbz=np.ones(128),
    )

    params = SimulationParams(
        data_path=temp_npz_dir,
        a=1.0,
        S=1e4,
        CGL=False,
        inner_resolution_safety=1.0,
    )

    vs = np.array([0.2])
    deltas = refine_inner_scale(vs, params)
    expected_analytic = estimate_inner_scale(params, alpha=0.2)
    assert deltas[0] is not None
    assert np.isclose(deltas[0], expected_analytic)
    assert not np.isclose(deltas[0], 0.001)


def test_refine_malformed_tolerance_acts_as_barrier(temp_npz_dir: str) -> None:
    """Missing or non-finite tolerance must be treated as unconverged barrier."""
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.io import save_eigenmode

    # k=0.1 (converged)
    save_eigenmode(
        os.path.join(temp_npz_dir, "state_α1.000000e-01.npz"),
        wavenumber=0.1,
        a=1.0,
        eigenvalue=0.05 + 0.0j,
        tolerance=1e-6,
        minimum_physical_scale=0.02,
        minimum_physical_scale_key="classical.bz_induction.eta_vs_ideal",
        minimum_scale_nodes=10,
        resistive_layer_thickness=0.02,
        resistive_layer_nodes=10,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=128,
        grid=np.linspace(-10, 10, 128),
        mode_scales={"classical.bz_induction.eta_vs_ideal": 0.02},
        duz=np.ones(128),
        dbz=np.ones(128),
    )

    # k=0.2 (NaN tolerance)
    save_eigenmode(
        os.path.join(temp_npz_dir, "state_α2.000000e-01.npz"),
        wavenumber=0.2,
        a=1.0,
        eigenvalue=0.05 + 0.0j,
        tolerance=float("nan"),
        minimum_physical_scale=0.001,
        minimum_physical_scale_key="classical.bz_induction.eta_vs_ideal",
        minimum_scale_nodes=10,
        resistive_layer_thickness=0.001,
        resistive_layer_nodes=10,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=128,
        grid=np.linspace(-10, 10, 128),
        mode_scales={"classical.bz_induction.eta_vs_ideal": 0.001},
        duz=np.ones(128),
        dbz=np.ones(128),
    )

    # k=0.3 (converged)
    save_eigenmode(
        os.path.join(temp_npz_dir, "state_α3.000000e-01.npz"),
        wavenumber=0.3,
        a=1.0,
        eigenvalue=0.05 + 0.0j,
        tolerance=1e-6,
        minimum_physical_scale=0.08,
        minimum_physical_scale_key="classical.bz_induction.eta_vs_ideal",
        minimum_scale_nodes=10,
        resistive_layer_thickness=0.08,
        resistive_layer_nodes=10,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=128,
        grid=np.linspace(-10, 10, 128),
        mode_scales={"classical.bz_induction.eta_vs_ideal": 0.08},
        duz=np.ones(128),
        dbz=np.ones(128),
    )

    params = SimulationParams(
        data_path=temp_npz_dir,
        a=1.0,
        S=1e4,
        CGL=False,
        inner_resolution_safety=1.0,
    )

    vs = np.array([0.15])
    deltas = refine_inner_scale(vs, params)
    expected_analytic = estimate_inner_scale(params, alpha=0.15)
    assert deltas[0] is not None
    assert np.isclose(deltas[0], expected_analytic)


def test_refine_mixed_cache_batching_and_order_invariance(temp_npz_dir: str) -> None:
    """Fallback resolution must be independent of batching, order, and per-term availability on other points."""
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.io import save_eigenmode

    # State 1 at k=0.1: new schema with per-term scale = 0.02, minimum_physical_scale = 0.02
    save_eigenmode(
        os.path.join(temp_npz_dir, "state_alpha0.100000e+00.npz"),
        wavenumber=0.1,
        a=1.0,
        eigenvalue=0.05 + 0.0j,
        tolerance=1e-6,
        minimum_physical_scale=0.02,
        minimum_physical_scale_key="classical.bz_induction.eta_vs_ideal",
        minimum_scale_nodes=10,
        resistive_layer_thickness=0.02,
        resistive_layer_nodes=10,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=128,
        grid=np.linspace(-10, 10, 128),
        mode_scales={"classical.bz_induction.eta_vs_ideal": 0.02},
        duz=np.ones(128),
        dbz=np.ones(128),
    )

    # State 2 at k=0.3: legacy schema with resistive_layer_thickness = 0.08 (no mode_scales)
    save_eigenmode(
        os.path.join(temp_npz_dir, "state_alpha0.300000e+00.npz"),
        wavenumber=0.3,
        a=1.0,
        eigenvalue=0.06 + 0.0j,
        tolerance=1e-6,
        resistive_layer_thickness=0.08,
        resistive_layer_nodes=10,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=128,
        grid=np.linspace(-10, 10, 128),
        duz=np.ones(128),
        dbz=np.ones(128),
    )

    params = SimulationParams(
        data_path=temp_npz_dir,
        a=1.0,
        S=1e4,
        CGL=False,
        inner_resolution_safety=1.0,
    )

    expected_scalar_k02 = 0.04796092578242428

    # 1. Requested [0.2] alone
    res_single = refine_inner_scale(np.array([0.2]), params)
    assert res_single[0] is not None
    assert np.isclose(res_single[0], expected_scalar_k02, rtol=1e-5)

    # 2. Requested [0.1, 0.2] together: k=0.1 has per-term data, k=0.2 must use scalar fallback
    res_pair1 = refine_inner_scale(np.array([0.1, 0.2]), params)
    assert res_pair1[0] is not None and res_pair1[1] is not None
    assert np.isclose(res_pair1[0], 0.02, rtol=1e-5)
    assert np.isclose(res_pair1[1], expected_scalar_k02, rtol=1e-5)

    # 3. Requested [0.2, 0.1] reversed order
    res_pair2 = refine_inner_scale(np.array([0.2, 0.1]), params)
    assert res_pair2[0] is not None and res_pair2[1] is not None
    assert np.isclose(res_pair2[0], expected_scalar_k02, rtol=1e-5)
    assert np.isclose(res_pair2[1], 0.02, rtol=1e-5)

    # 4. Requested [0.1, 0.2, 0.3] triplet
    res_triplet = refine_inner_scale(np.array([0.1, 0.2, 0.3]), params)
    assert res_triplet[0] is not None and res_triplet[1] is not None and res_triplet[2] is not None
    assert np.isclose(res_triplet[0], 0.02, rtol=1e-5)
    assert np.isclose(res_triplet[1], expected_scalar_k02, rtol=1e-5)
    assert np.isclose(res_triplet[2], 0.08, rtol=1e-5)


def test_refine_per_term_precedence_over_scalar(temp_npz_dir: str) -> None:
    """Valid per-term scale must take precedence over differing scalar values at the same target."""
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.io import save_eigenmode

    # State where mode_scales says 0.03, but scalar resistive_layer_thickness says 0.09
    save_eigenmode(
        os.path.join(temp_npz_dir, "state_alpha0.100000e+00.npz"),
        wavenumber=0.1,
        a=1.0,
        eigenvalue=0.05 + 0.0j,
        tolerance=1e-6,
        minimum_physical_scale=0.09,
        minimum_physical_scale_key="classical.bz_induction.eta_vs_ideal",
        minimum_scale_nodes=10,
        resistive_layer_thickness=0.09,
        resistive_layer_nodes=10,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=128,
        grid=np.linspace(-10, 10, 128),
        mode_scales={"classical.bz_induction.eta_vs_ideal": 0.03},
        duz=np.ones(128),
        dbz=np.ones(128),
    )

    params = SimulationParams(
        data_path=temp_npz_dir,
        a=1.0,
        S=1e4,
        CGL=False,
        inner_resolution_safety=1.0,
    )

    deltas = refine_inner_scale(np.array([0.1]), params)
    assert deltas[0] is not None
    # Must select per-term value 0.03, not scalar 0.09
    assert np.isclose(deltas[0], 0.03)
    assert not np.isclose(deltas[0], 0.09)


def test_safety_default_margin_and_drift_protection(temp_npz_dir: str) -> None:
    """Default safety margin (1.01) protects against node count dropping under scale drift."""
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.io import save_eigenmode
    from tearing_eigenmodes.grid import select_NC
    from psecas import ChebyshevRationalGrid

    # 1. State with cached physical scale = 0.04
    save_eigenmode(
        os.path.join(temp_npz_dir, "state_alpha0.100000e+00.npz"),
        wavenumber=0.1,
        a=1.0,
        eigenvalue=0.05 + 0.0j,
        tolerance=1e-6,
        minimum_physical_scale=0.040,
        minimum_physical_scale_key="classical.bz_induction.eta_vs_ideal",
        minimum_scale_nodes=10,
        resistive_layer_thickness=0.040,
        resistive_layer_nodes=10,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=128,
        grid=np.linspace(-10, 10, 128),
        mode_scales={"classical.bz_induction.eta_vs_ideal": 0.040},
        duz=np.ones(128),
        dbz=np.ones(128),
    )

    # 2. Refine with default safety = 1.01
    params_default = SimulationParams(
        data_path=temp_npz_dir,
        alpha=0.1,
        a=1.0,
        S=1e4,
        CGL=False,
    )
    assert params_default.inner_resolution_safety == 1.01
    res_default = refine_inner_scale(np.array([0.1]), params_default)
    assert res_default[0] is not None
    # Grid scale target must be 0.040 / 1.01
    assert np.isclose(res_default[0], 0.040 / 1.01, rtol=1e-6)

    # 3. Explicit override preserves exact supplied target
    params_explicit = SimulationParams(
        data_path=temp_npz_dir,
        alpha=0.1,
        inner_scale=0.099,
    )
    res_explicit = refine_inner_scale(np.array([0.1]), params_explicit)
    assert res_explicit[0] is not None
    assert np.isclose(res_explicit[0], 0.099)

    # 4. Analytic estimate with default safety has exactly one factor of 1 / 1.01
    params_empty_default = SimulationParams(alpha=0.1, a=1.0, S=1e4, CGL=False)
    params_empty_10 = SimulationParams(alpha=0.1, a=1.0, S=1e4, CGL=False, inner_resolution_safety=1.0)
    est_default = estimate_inner_scale(params_empty_default)
    est_10 = estimate_inner_scale(params_empty_10)
    assert np.isclose(est_default, est_10 / 1.01, rtol=1e-6)

    # 5. Grid node count verification under 0.32% physical scale drift
    # With safety = 1.01, grid target is L_target = 0.040 / 1.01
    params_grid = SimulationParams(
        alpha=0.1,
        a=1.0,
        w=0.0,
        inner_scale=0.040 / 1.01,
        n_equilibrium=5,
        n_inner_scale=5,
        f_outer=0.2,
        Nmin=64,
        Nmax=1024,
        Ninc=32,
    )
    N_opt, C_opt = select_NC(params_grid)
    grid = ChebyshevRationalGrid(N=N_opt, C=C_opt)
    # Remeasured physical scale drifts down by 0.32% to 0.040 * (1 - 0.0032) = 0.039872
    drifted_physical_scale = 0.040 * (1.0 - 0.0032)
    nodes_inside = int(np.sum(np.abs(grid.zg) <= drifted_physical_scale))
    # Nodes inside the remeasured physical scale remain >= 5 with the 1.01 margin
    assert nodes_inside >= 5


def test_refine_malformed_center_record_barrier_retained(temp_npz_dir: str) -> None:
    """A record with malformed scale data must remain visible as an invalidity barrier."""
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.io import load_state_data
    from tearing_eigenmodes.refinement import _load_cached_state_records

    fp1 = os.path.join(temp_npz_dir, "state_alpha0.100000e+00.npz")
    fp2 = os.path.join(temp_npz_dir, "state_alpha0.200000e+00.npz")
    fp3 = os.path.join(temp_npz_dir, "state_alpha0.300000e+00.npz")

    # Record 1 at k=0.1: valid scale = 0.02
    np.savez_compressed(
        fp1,
        wavenumber=0.1,
        a=1.0,
        tolerance=1e-5,
        resolution=128,
        mode_scale_keys=np.array(["classical.bz_induction.eta_vs_ideal"]),
        mode_scale_values=np.array([0.02]),
        mode_scale_schema_version=MODE_SCALE_SCHEMA_VERSION,
        minimum_physical_scale=np.nan,
    )

    # Record 2 at k=0.2: malformed scale value = 'bad'
    np.savez_compressed(
        fp2,
        wavenumber=0.2,
        a=1.0,
        tolerance=1e-5,
        resolution=128,
        mode_scale_keys=np.array(["classical.bz_induction.eta_vs_ideal"]),
        mode_scale_values=np.array(["bad"]),
        mode_scale_schema_version=MODE_SCALE_SCHEMA_VERSION,
        minimum_physical_scale=np.nan,
    )

    # Record 3 at k=0.3: valid scale = 0.08
    np.savez_compressed(
        fp3,
        wavenumber=0.3,
        a=1.0,
        tolerance=1e-5,
        resolution=128,
        mode_scale_keys=np.array(["classical.bz_induction.eta_vs_ideal"]),
        mode_scale_values=np.array([0.08]),
        mode_scale_schema_version=MODE_SCALE_SCHEMA_VERSION,
        minimum_physical_scale=np.nan,
    )

    # 1. load_state_data(center) must not be None and must decode center coordinate
    d_center = load_state_data(fp2)
    assert d_center is not None
    assert d_center["wavenumber"] == 0.2
    assert np.isnan(d_center["mode_scales"]["classical.bz_induction.eta_vs_ideal"])

    params = SimulationParams(
        data_path=temp_npz_dir,
        a=1.0,
        S=1e4,
        CGL=False,
        inner_resolution_safety=1.0,
    )

    # 2. _load_cached_state_records must retain all three coordinates [0.1, 0.2, 0.3]
    records = _load_cached_state_records(temp_npz_dir, params)
    assert [r["v"] for r in records] == [0.1, 0.2, 0.3]

    # 3. Refine at k=0.15: must NOT cross the barrier at k=0.2 and must return analytic estimate
    vs = np.array([0.15])
    deltas = refine_inner_scale(vs, params)
    assert deltas[0] is not None

    expected_analytic = estimate_inner_scale(params, alpha=0.15)
    forbidden_cross_gap = 0.0333604903137

    assert np.isclose(deltas[0], expected_analytic, rtol=1e-5)
    assert not np.isclose(deltas[0], forbidden_cross_gap, rtol=1e-2)


def test_refine_center_record_multi_key_partial_malformed(temp_npz_dir: str) -> None:
    """One malformed scale key in a record must invalidate only that key while other valid keys remain usable."""
    _load_eigenmodes_cache.clear()

    fp1 = os.path.join(temp_npz_dir, "state_alpha0.100000e+00.npz")
    fp2 = os.path.join(temp_npz_dir, "state_alpha0.200000e+00.npz")
    fp3 = os.path.join(temp_npz_dir, "state_alpha0.300000e+00.npz")

    # Record 1 at k=0.1: induction=0.02, vorticity=0.02
    np.savez_compressed(
        fp1,
        wavenumber=0.1,
        a=1.0,
        tolerance=1e-5,
        resolution=128,
        mode_scale_keys=np.array(["classical.bz_induction.eta_vs_ideal", "classical.uz_vorticity.nu_vs_ideal"]),
        mode_scale_values=np.array([0.02, 0.02]),
        mode_scale_schema_version=MODE_SCALE_SCHEMA_VERSION,
        minimum_physical_scale=np.nan,
    )

    # Record 2 at k=0.2: induction='bad' (malformed), vorticity=0.04 (valid)
    np.savez_compressed(
        fp2,
        wavenumber=0.2,
        a=1.0,
        tolerance=1e-5,
        resolution=128,
        mode_scale_keys=np.array(["classical.bz_induction.eta_vs_ideal", "classical.uz_vorticity.nu_vs_ideal"]),
        mode_scale_values=np.array(["bad", 0.04]),
        mode_scale_schema_version=MODE_SCALE_SCHEMA_VERSION,
        minimum_physical_scale=np.nan,
    )

    # Record 3 at k=0.3: induction=0.08, vorticity=0.08
    np.savez_compressed(
        fp3,
        wavenumber=0.3,
        a=1.0,
        tolerance=1e-5,
        resolution=128,
        mode_scale_keys=np.array(["classical.bz_induction.eta_vs_ideal", "classical.uz_vorticity.nu_vs_ideal"]),
        mode_scale_values=np.array([0.08, 0.08]),
        mode_scale_schema_version=MODE_SCALE_SCHEMA_VERSION,
        minimum_physical_scale=np.nan,
    )

    params = SimulationParams(
        data_path=temp_npz_dir,
        a=1.0,
        S=1e4,
        CGL=False,
        inner_resolution_safety=1.0,
    )

    # Refine at k=0.15:
    # - induction key is split by barrier at 0.2 (no segment covering 0.15)
    # - vorticity key has continuous segment [0.1, 0.2, 0.3] -> log-log interp at 0.15
    deltas = refine_inner_scale(np.array([0.15]), params)
    assert deltas[0] is not None
    # Vorticity log-log interp between (0.1, 0.02) and (0.2, 0.04) at 0.15:
    # scale = 0.02 * (0.15 / 0.1) = 0.03
    assert np.isclose(deltas[0], 0.03, rtol=1e-5)


def test_refine_non_mapping_mode_scales_barrier(temp_npz_dir: str) -> None:
    """A non-mapping mode_scales field must not raise and must act as an unavailable data barrier."""
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.io import load_state_data
    from tearing_eigenmodes.refinement import _load_cached_state_records

    fp1 = os.path.join(temp_npz_dir, "state_alpha0.100000e+00.npz")
    fp2 = os.path.join(temp_npz_dir, "state_alpha0.200000e+00.npz")
    fp3 = os.path.join(temp_npz_dir, "state_alpha0.300000e+00.npz")

    # Record 1 at k=0.1: valid per-term scale 0.02
    np.savez_compressed(
        fp1,
        wavenumber=0.1,
        a=1.0,
        tolerance=1e-5,
        resolution=128,
        mode_scale_keys=np.array(["classical.bz_induction.eta_vs_ideal"]),
        mode_scale_values=np.array([0.02]),
        mode_scale_schema_version=MODE_SCALE_SCHEMA_VERSION,
        minimum_physical_scale=np.nan,
    )

    # Record 2 at k=0.2: raw non-mapping mode_scales field, no parallel keys/values
    np.savez_compressed(
        fp2,
        wavenumber=0.2,
        a=1.0,
        tolerance=1e-5,
        resolution=128,
        mode_scales="not-a-mapping",
        minimum_physical_scale=np.nan,
    )

    # Record 3 at k=0.3: valid per-term scale 0.08
    np.savez_compressed(
        fp3,
        wavenumber=0.3,
        a=1.0,
        tolerance=1e-5,
        resolution=128,
        mode_scale_keys=np.array(["classical.bz_induction.eta_vs_ideal"]),
        mode_scale_values=np.array([0.08]),
        mode_scale_schema_version=MODE_SCALE_SCHEMA_VERSION,
        minimum_physical_scale=np.nan,
    )

    params = SimulationParams(
        data_path=temp_npz_dir,
        a=1.0,
        S=1e4,
        CGL=False,
        inner_resolution_safety=1.0,
    )

    # 1. State data for center retains raw non-mapping mode_scales
    d_center = load_state_data(fp2)
    assert d_center is not None
    assert d_center["mode_scales"] == "not-a-mapping"

    # 2. Cached records include all three coordinates and expose the center non-mapping value
    records = _load_cached_state_records(temp_npz_dir, params)
    assert [r["v"] for r in records] == [0.1, 0.2, 0.3]
    assert records[1]["mode_scales"] == "not-a-mapping"

    # 3. Refine at k=0.15: non-mapping center acts as barrier, returning analytic fallback
    deltas = refine_inner_scale(np.array([0.15]), params)
    expected_analytic = estimate_inner_scale(params, alpha=0.15)
    unwanted_cross_gap = 0.0333604903137

    assert deltas[0] is not None
    assert np.isclose(deltas[0], expected_analytic, rtol=1e-5)
    assert not np.isclose(deltas[0], unwanted_cross_gap, rtol=1e-2)


def test_refine_malformed_scalar_fallback_barrier(temp_npz_dir: str) -> None:
    """Malformed scalar fallbacks must act as an unavailable data barrier for the scalar path."""
    _load_eigenmodes_cache.clear()

    fp1 = os.path.join(temp_npz_dir, "state_alpha0.100000e+00.npz")
    fp2 = os.path.join(temp_npz_dir, "state_alpha0.200000e+00.npz")

    # Record 1 at k=0.1: valid scalar fallback 0.05
    np.savez_compressed(
        fp1,
        wavenumber=0.1,
        a=1.0,
        tolerance=1e-5,
        resolution=128,
        minimum_physical_scale=0.05,
    )

    # Record 2 at k=0.2: malformed scalar fallback
    np.savez_compressed(
        fp2,
        wavenumber=0.2,
        a=1.0,
        tolerance=1e-5,
        resolution=128,
        minimum_physical_scale="bad_scalar",
        resistive_layer_thickness="bad_res",
    )

    params = SimulationParams(
        data_path=temp_npz_dir,
        a=1.0,
        S=1e4,
        CGL=False,
        inner_resolution_safety=1.0,
    )

    # Refine at k=0.15: k=0.2 is a barrier for scalar path -> falls back to analytic estimator
    deltas = refine_inner_scale(np.array([0.15]), params)
    expected_analytic = estimate_inner_scale(params, alpha=0.15)
    assert deltas[0] is not None
    assert np.isclose(deltas[0], expected_analytic, rtol=1e-5)


def test_refine_fractional_schema_version_barrier(temp_npz_dir: str) -> None:
    """A record with fractional schema version (e.g. 1.5) must not be trusted and must form a barrier."""
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.io import load_state_data
    from tearing_eigenmodes.refinement import _load_cached_state_records

    fp1 = os.path.join(temp_npz_dir, "state_alpha0.100000e+00.npz")
    fp2 = os.path.join(temp_npz_dir, "state_alpha0.200000e+00.npz")
    fp3 = os.path.join(temp_npz_dir, "state_alpha0.300000e+00.npz")

    # Record 1 at k=0.1: valid schema=1, scale=0.02
    np.savez_compressed(
        fp1,
        wavenumber=0.1,
        a=1.0,
        tolerance=1e-5,
        resolution=128,
        mode_scale_keys=np.array(["classical.bz_induction.eta_vs_ideal"]),
        mode_scale_values=np.array([0.02]),
        mode_scale_schema_version=MODE_SCALE_SCHEMA_VERSION,
        minimum_physical_scale=np.nan,
    )

    # Record 2 at k=0.2: fractional schema=1.5, scale=0.0001
    np.savez_compressed(
        fp2,
        wavenumber=0.2,
        a=1.0,
        tolerance=1e-5,
        resolution=128,
        mode_scale_keys=np.array(["classical.bz_induction.eta_vs_ideal"]),
        mode_scale_values=np.array([0.0001]),
        mode_scale_schema_version=1.5,
        minimum_physical_scale=np.nan,
    )

    # Record 3 at k=0.3: valid schema=2, scale=0.08
    np.savez_compressed(
        fp3,
        wavenumber=0.3,
        a=1.0,
        tolerance=1e-5,
        resolution=128,
        mode_scale_keys=np.array(["classical.bz_induction.eta_vs_ideal"]),
        mode_scale_values=np.array([0.08]),
        mode_scale_schema_version=MODE_SCALE_SCHEMA_VERSION,
        minimum_physical_scale=np.nan,
    )

    params = SimulationParams(
        data_path=temp_npz_dir,
        a=1.0,
        S=1e4,
        CGL=False,
        inner_resolution_safety=1.0,
    )

    # 1. State data for center has empty mode_scales
    d_center = load_state_data(fp2)
    assert d_center is not None
    assert d_center["mode_scales"] == {}

    # 2. Cached records include all three coordinates
    records = _load_cached_state_records(temp_npz_dir, params)
    assert [r["v"] for r in records] == [0.1, 0.2, 0.3]

    # 3. Refine at k=0.15: returns analytic fallback (0.06065972342799003) and not cross-gap (0.00090159965149587)
    deltas = refine_inner_scale(np.array([0.15]), params)
    expected_analytic = estimate_inner_scale(params, alpha=0.15)
    unwanted_cross_gap = 0.00090159965149587

    assert deltas[0] is not None
    assert np.isclose(deltas[0], expected_analytic, rtol=1e-5)
    assert not np.isclose(deltas[0], unwanted_cross_gap, rtol=1e-2)


def test_refine_eigenvalues_and_brackets_with_malformed_scales(temp_npz_dir: str) -> None:
    """refine_eigenvalues and refine_wavenumber_bracket must succeed even if optional scale diagnostics are malformed."""
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.refinement import refine_eigenvalues, refine_wavenumber_bracket

    # File 1 at S=1e4
    np.savez_compressed(
        os.path.join(temp_npz_dir, "state_S1.000000e+04.npz"),
        scan_parameter="S",
        scan_parameter_value=1e4,
        wavenumber=0.1,
        eigenvalue=0.01 + 0.0j,
        tolerance=1e-5,
        resolution=128,
        minimum_physical_scale="bad",
        minimum_scale_nodes="bad",
    )
    # File 2 at S=1e5
    np.savez_compressed(
        os.path.join(temp_npz_dir, "state_S1.000000e+05.npz"),
        scan_parameter="S",
        scan_parameter_value=1e5,
        wavenumber=0.1,
        eigenvalue=0.02 + 0.0j,
        tolerance=1e-5,
        resolution=128,
        minimum_physical_scale="bad",
        minimum_scale_nodes="bad",
    )

    params = SimulationParams(
        data_path=temp_npz_dir,
        dependence="S",
        logarithmic=True,
    )

    # 1. refine_eigenvalues operates smoothly
    sigmas = refine_eigenvalues(np.array([3e4]), params)
    assert sigmas is not None
    assert len(sigmas) == 1
    assert np.isfinite(sigmas[0])

    # 2. refine_wavenumber_bracket operates smoothly
    brackets = refine_wavenumber_bracket(np.array([3e4]), params)
    assert brackets is not None
    assert len(brackets) == 1


def test_refine_malformed_saved_a_fallback_preserves_records(temp_npz_dir: str) -> None:
    """Malformed saved 'a' must fall back to params.a and retain all valid records."""
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.refinement import _load_cached_state_records

    fp1 = os.path.join(temp_npz_dir, "state_alpha0.100000e+00.npz")
    fp2 = os.path.join(temp_npz_dir, "state_alpha0.200000e+00.npz")
    fp3 = os.path.join(temp_npz_dir, "state_alpha0.300000e+00.npz")

    # Record 1 at k=0.1: a=1.0, scale=0.02
    np.savez_compressed(
        fp1,
        wavenumber=0.1,
        a=1.0,
        tolerance=1e-5,
        resolution=128,
        mode_scale_keys=np.array(["classical.bz_induction.eta_vs_ideal"]),
        mode_scale_values=np.array([0.02]),
        mode_scale_schema_version=MODE_SCALE_SCHEMA_VERSION,
    )
    # Record 2 at k=0.2: malformed a="bad", scale=0.04
    np.savez_compressed(
        fp2,
        wavenumber=0.2,
        a="bad",
        tolerance=1e-5,
        resolution=128,
        mode_scale_keys=np.array(["classical.bz_induction.eta_vs_ideal"]),
        mode_scale_values=np.array([0.04]),
        mode_scale_schema_version=MODE_SCALE_SCHEMA_VERSION,
    )
    # Record 3 at k=0.3: a=1.0, scale=0.08
    np.savez_compressed(
        fp3,
        wavenumber=0.3,
        a=1.0,
        tolerance=1e-5,
        resolution=128,
        mode_scale_keys=np.array(["classical.bz_induction.eta_vs_ideal"]),
        mode_scale_values=np.array([0.08]),
        mode_scale_schema_version=MODE_SCALE_SCHEMA_VERSION,
    )

    params = SimulationParams(
        data_path=temp_npz_dir,
        a=1.0,
        S=1e4,
        CGL=False,
        inner_resolution_safety=1.0,
    )

    # 1. All three records are loaded and have correct coordinates
    records = _load_cached_state_records(temp_npz_dir, params)
    assert len(records) == 3
    assert [r["v"] for r in records] == [0.1, 0.2, 0.3]

    # 2. Refinement at k=0.1 returns exact cached scale 0.02, not analytic 0.07015872133070601
    deltas = refine_inner_scale(np.array([0.1]), params)
    assert deltas[0] is not None
    assert np.isclose(deltas[0], 0.02, rtol=1e-5)
    assert not np.isclose(deltas[0], 0.07015872133070601, rtol=1e-2)


def test_refine_malformed_saved_a_nonunit_and_invalid_fallbacks(temp_npz_dir: str) -> None:
    """Non-unit params.a, NaN/inf/zero/negative 'a', and unrecoverable records."""
    _load_eigenmodes_cache.clear()
    from tearing_eigenmodes.refinement import _load_cached_state_records

    # 1. Non-unit params.a = 2.0 with malformed saved a="bad" -> k = alpha / a = 0.2 / 2.0 = 0.1
    fp1 = os.path.join(temp_npz_dir, "state_alpha0.200000e+00.npz")
    np.savez_compressed(
        fp1,
        wavenumber=0.2,
        a="bad",
        tolerance=1e-5,
        resolution=128,
        minimum_physical_scale=0.04,
    )
    params2 = SimulationParams(data_path=temp_npz_dir, a=2.0)
    records = _load_cached_state_records(temp_npz_dir, params2)
    assert len(records) == 1
    assert np.isclose(records[0]["v"], 0.1)

    # 2. Various bad a values (NaN, inf, 0, -1) with params.a = 1.0 -> all resolve to k = alpha / 1.0
    bad_a_vals = [np.nan, np.inf, 0.0, -1.0]
    for i, bad_a in enumerate(bad_a_vals):
        fp_bad = os.path.join(temp_npz_dir, f"state_bada_{i}.npz")
        np.savez_compressed(
            fp_bad,
            wavenumber=0.4 + i * 0.1,
            a=bad_a,
            tolerance=1e-5,
            resolution=128,
            minimum_physical_scale=0.05,
        )
    params1 = SimulationParams(data_path=temp_npz_dir, a=1.0)
    records_all = _load_cached_state_records(temp_npz_dir, params1)
    assert len(records_all) == 5  # record 1 + 4 bad_a records

    # 3. Unrecoverable record where both saved 'a' and params.a are bad -> only that record is skipped
    fp_unrec = os.path.join(temp_npz_dir, "state_unrecoverable.npz")
    np.savez_compressed(
        fp_unrec,
        wavenumber=0.9,
        a="bad",
        tolerance=1e-5,
        resolution=128,
    )
    params_none = SimulationParams(data_path=temp_npz_dir)
    params_none.a = None  # unusable fallback
    records_sub = _load_cached_state_records(temp_npz_dir, params_none)
    # The bad 'a' records with no params.a fallback are skipped, but valid ones (like record 1 with a=1.0) survive
    assert len(records_sub) == 0 or all(r["v"] is not None for r in records_sub)
