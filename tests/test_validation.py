import os
import tempfile
import numpy as np
import pytest

from tearing_eigenmodes import save_eigenmode
from tearing_eigenmodes.validation import validate_and_fix_file


def test_validate_dispersion_file():
    """Test validation and key mapping of a dispersion run .npz file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "state_a0.1.npz")
        
        # Save a mock old dispersion state file
        mock_data = {
            "value": np.array(0.1),
            "wavenumber": np.array(0.1),
            "growth_rate": np.array(0.05 + 0.0j),
            "tolerance": np.array(1e-6),
            "inner_scale": np.array(0.15),
            "scaling_factor": np.array(2.5),
            "resolution": np.array(128),
            "grid": np.linspace(-10, 10, 128),
            "a": np.array(1.0),
            "w": np.array(0.0),
            "CGL": np.array(False),
            # Missing: n_inner (resistive_layer_nodes), n_wa (current_sheet_nodes)
            "duz": np.sin(np.linspace(0, np.pi, 128)),
            "dbz": np.cos(np.linspace(0, np.pi, 128)),
        }
        
        np.savez(filepath, **mock_data)
        
        # Run validation
        success, modified = validate_and_fix_file(filepath, dry_run=False, verbose=True)
        assert success is True
        assert modified is True
        
        # Load and assert new names and values
        with np.load(filepath) as state:
            # Check renames
            assert "value" not in state
            assert "wavenumber" in state
            assert state["wavenumber"] == 0.1
            assert "eigenvalue" in state
            assert state["eigenvalue"] == 0.05
            assert "resistive_layer_thickness" in state
            assert state["resistive_layer_thickness"] == 0.15
            assert "grid_scaling_factor" in state
            assert state["grid_scaling_factor"] == 2.5
            
            # Check recalculations
            assert "resistive_layer_nodes" in state
            # grid goes from -10 to 10 with 128 points
            # points <= 0.15 in absolute value
            grid = state["grid"]
            expected_nodes = np.where(np.abs(grid) <= 0.15)[0].size
            assert state["resistive_layer_nodes"] == expected_nodes
            
            assert "current_sheet_nodes" in state
            expected_wa = np.where(np.abs(grid) <= 1.0)[0].size
            assert state["current_sheet_nodes"] == expected_wa


def test_validate_dependence_file():
    """Test validation and key mapping of a dependence sweep .npz file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "state_S1e4.npz")
        
        # Save a mock old dependence sweep file
        mock_data = {
            "dependence": np.array("S"),
            "value": np.array(1e4),
            "wavenumber": np.array(0.2),
            "growth_rate": np.array(0.12 + 0.01j),
            "tolerance": np.array(1e-5),
            "inner_scale": np.array(0.08),
            "scaling_factor": np.array(3.0),
            "resolution": np.array(256),
            "grid": np.linspace(-15, 15, 256),
            "a": np.array(1.0),
            "w": np.array(0.5),
            "CGL": np.array(False),
            # Missing: n_inner, n_wa, niter, wavenumber_error, eigenvalue_error
            "duz": np.sin(np.linspace(0, np.pi, 256)),
            "dbz": np.cos(np.linspace(0, np.pi, 256)),
            "rtol": np.array(1e-5),
        }
        
        np.savez(filepath, **mock_data)
        
        # Run validation
        success, modified = validate_and_fix_file(filepath, dry_run=False, verbose=True)
        assert success is True
        assert modified is True
        
        # Load and assert recalculated/renamed fields
        with np.load(filepath) as state:
            assert "dependence" not in state
            assert "value" not in state
            assert state["scan_parameter"] == "S"
            assert state["scan_parameter_value"] == 1e4
            assert state["eigenvalue"] == 0.12 + 0.01j
            
            # Recalculations
            assert "niter" in state
            assert state["niter"] == 1
            
            assert "wavenumber_error" in state
            assert state["wavenumber_error"] == 1e-3 * 0.2
            
            assert "eigenvalue_error" in state
            expected_err = 1e-5 * 0.12 * 1e-5
            assert np.isclose(state["eigenvalue_error"], expected_err)


def test_invalid_file_missing_cores():
    """Test that validation fails gracefully if core fields are completely missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "corrupt.npz")
        
        # Save a file missing the grid and eigenfunctions
        mock_data = {
            "wavenumber": np.array(0.1),
            "growth_rate": np.array(0.05),
        }
        
        np.savez(filepath, **mock_data)
        
        success, modified = validate_and_fix_file(filepath, dry_run=False, verbose=True)
        assert success is False
        assert modified is False


def test_validate_dependence_guess_from_filename():
    """Test guessing dependence parameter/value from filename when missing in data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "state_S+1.000000e+02.npz")
        
        # Save a mock old sweep file lacking scan_parameter/dependence fields
        mock_data = {
            "wavenumber": np.array(0.2),
            "growth_rate": np.array(0.12),
            "tolerance": np.array(1e-5),
            "inner_scale": np.array(0.08),
            "scaling_factor": np.array(3.0),
            "resolution": np.array(256),
            "grid": np.linspace(-15, 15, 256),
            "a": np.array(1.0),
            "w": np.array(0.5),
            "CGL": np.array(False),
            "duz": np.sin(np.linspace(0, np.pi, 256)),
            "dbz": np.cos(np.linspace(0, np.pi, 256)),
            "rtol": np.array(1e-5),
        }
        
        np.savez(filepath, **mock_data)
        
        success, modified = validate_and_fix_file(filepath, dry_run=False, verbose=True)
        assert success is True
        assert modified is True
        
        with np.load(filepath) as state:
            assert state["scan_parameter"] == "S"
            assert state["scan_parameter_value"] == 100.0


def test_validate_dispersion_guess_from_filename():
    """Test guessing wavenumber from filename for dispersion runs if missing in data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "state_α2.500000e-01.npz")
        
        # Save a mock dispersion file lacking wavenumber field
        mock_data = {
            "growth_rate": np.array(0.05),
            "tolerance": np.array(1e-6),
            "inner_scale": np.array(0.15),
            "scaling_factor": np.array(2.5),
            "resolution": np.array(128),
            "grid": np.linspace(-10, 10, 128),
            "a": np.array(1.0),
            "w": np.array(0.0),
            "CGL": np.array(False),
            "duz": np.sin(np.linspace(0, np.pi, 128)),
            "dbz": np.cos(np.linspace(0, np.pi, 128)),
        }
        
        np.savez(filepath, **mock_data)
        
        success, modified = validate_and_fix_file(filepath, dry_run=False, verbose=True)
        assert success is True
        assert modified is True
        
        with np.load(filepath) as state:
            assert "scan_parameter" not in state
            assert state["wavenumber"] == 0.25
