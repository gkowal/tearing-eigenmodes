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
