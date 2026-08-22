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
        wavenumber=float(wavenumber),
        scan_parameter="S",
        scan_parameter_value=float(val),
        eigenvalue=complex(growth),
        resistive_layer_thickness=float(thickness),
        tolerance=float(tolerance),
        grid_scaling_factor=float(scaling),
        resolution=int(resolution),
        resistive_layer_nodes=int(nin),
        current_sheet_nodes=int(nwa),
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

    # 1. Within cached range [1.0, 2.0]: logarithmic interpolation
    vs = np.array([1.5])
    deltas = refine_inner_scale(vs, params)
    assert deltas[0] is not None
    # Log-space midpoint: exp((ln(0.02) + ln(0.04))/2) = sqrt(0.02 * 0.04) = 0.02828...
    expected_log_mid = np.sqrt(0.02 * 0.04)
    assert np.isclose(deltas[0], expected_log_mid, atol=1e-3)

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

    vs = np.array([1.0, 2.5])
    deltas = refine_inner_scale(vs, params)
    # At v=1.0, the cached thickness was 0.0 (ignored, outside valid cached range [2.0, 3.0]), so it gets estimator
    expected_v1 = estimate_inner_scale(params, alpha=1.0)
    assert deltas[0] is not None
    assert np.isclose(deltas[0], expected_v1)

    # At v=2.5, it interpolates between 0.04 and 0.08
    expected_v25 = np.sqrt(0.04 * 0.08)
    assert deltas[1] is not None
    assert np.isclose(deltas[1], expected_v25, atol=1e-3)
