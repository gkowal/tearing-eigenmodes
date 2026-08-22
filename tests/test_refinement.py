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
