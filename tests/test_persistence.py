import os
import tempfile
import numpy as np
import pytest
from tearing_eigenmodes.io import (
    save_eigenmode,
    check_state,
    load_eigenmodes,
    write_results,
)
from tearing_eigenmodes.analysis import (
    MODE_SCALE_SCHEMA_VERSION,
    CLASSICAL_GRID_SCALE_KEYS,
    CGL_GRID_SCALE_KEYS,
)
from tearing_eigenmodes.params import SimulationParams


def test_serialization_round_trip_without_pickle():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "state_alpha0.1.npz")

        scales = {
            "classical.bz_induction.eta_vs_ideal": 0.045,
            "classical.bz_induction.eta_vs_f": 0.050,
            "classical.bz_induction.g_vs_f": float("nan"),
            "classical.bz_induction.xi_vs_f": float("inf"),
            "classical.uz_vorticity.nu_vs_ideal": 0.025,
            "classical.uz_vorticity.nu_vs_f": 0.030,
            "classical.uz_vorticity.g_vs_f": float("nan"),
            "classical.uz_vorticity.xi_vs_f": float("nan"),
        }

        save_eigenmode(
            filepath,
            wavenumber=0.1,
            eigenvalue=0.05 + 0.01j,
            tolerance=1e-5,
            minimum_physical_scale=0.025,
            minimum_physical_scale_key="classical.uz_vorticity.nu_vs_ideal",
            minimum_scale_nodes=12,
            resistive_layer_thickness=0.045,
            resistive_layer_nodes=20,
            grid_scaling_factor=1.5,
            current_sheet_nodes=64,
            resolution=128,
            grid=np.linspace(-10, 10, 128),
            mode_scales=scales,
            duz=np.ones(128),
            dbz=np.ones(128),
        )

        # 1. Verify np.load works with allow_pickle=False
        with np.load(filepath, allow_pickle=False) as npz:
            assert "mode_scale_schema_version" in npz
            assert npz["mode_scale_schema_version"] == MODE_SCALE_SCHEMA_VERSION
            assert "mode_scale_keys" in npz
            assert "mode_scale_values" in npz
            assert "minimum_physical_scale" in npz
            assert "minimum_physical_scale_key" in npz
            assert "minimum_scale_nodes" in npz
            assert "resistive_layer_thickness" in npz
            assert "resistive_layer_nodes" in npz

            # Verify deterministic sort order of keys
            loaded_keys = [str(k) for k in npz["mode_scale_keys"]]
            assert loaded_keys == sorted(scales.keys())

            # Verify values including nan and inf
            loaded_values = npz["mode_scale_values"]
            for k, val in zip(loaded_keys, loaded_values):
                expected = scales[k]
                if np.isnan(expected):
                    assert np.isnan(val)
                elif np.isinf(expected):
                    assert np.isinf(val)
                else:
                    assert np.isclose(val, expected)

        # 2. Verify check_state reconstructs the dictionary
        status, data = check_state(filepath)
        assert status is True
        assert data is not None
        assert "mode_scales" in data
        reconstructed = data["mode_scales"]
        for k in scales:
            assert k in reconstructed
            if np.isnan(scales[k]):
                assert np.isnan(reconstructed[k])
            elif np.isinf(scales[k]):
                assert np.isinf(reconstructed[k])
            else:
                assert np.isclose(reconstructed[k], scales[k])

        assert data["minimum_physical_scale"] == 0.025
        assert data["minimum_physical_scale_key"] == "classical.uz_vorticity.nu_vs_ideal"
        assert data["minimum_scale_nodes"] == 12
        assert data["resistive_layer_thickness"] == 0.045
        assert data["resistive_layer_nodes"] == 20


def test_old_state_backward_compatibility():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "state_legacy.npz")

        # Save old-style state with only resistive_layer_thickness
        np.savez(
            filepath,
            wavenumber=0.2,
            eigenvalue=0.04 + 0.0j,
            tolerance=1e-6,
            resistive_layer_thickness=0.08,
            resistive_layer_nodes=16,
            grid_scaling_factor=2.0,
            current_sheet_nodes=60,
            resolution=64,
            grid=np.linspace(-5, 5, 64),
            duz=np.ones(64),
            dbz=np.ones(64),
        )

        status, data = check_state(filepath)
        assert status is True
        assert data is not None
        # Fallback fields populated
        assert data["minimum_physical_scale"] == 0.08
        assert data["minimum_scale_nodes"] == 16
        assert data["resistive_layer_thickness"] == 0.08

        # Test load_eigenmodes
        v, alpha, sigma, e, delta, c, nin, nwa, N = load_eigenmodes(tmpdir, "state_legacy.npz")
        assert len(delta) == 1
        assert delta[0] == 0.08
        assert nin[0] == 16


def test_write_results_uses_minimum_physical_scale():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a new-style state file where viscous scale (0.03) is smaller than resistive scale (0.09)
        filepath = os.path.join(tmpdir, "state_alpha0.200000e+00.npz")
        scales = {k: float("nan") for k in CLASSICAL_GRID_SCALE_KEYS}
        scales["classical.bz_induction.eta_vs_ideal"] = 0.09
        scales["classical.uz_vorticity.nu_vs_ideal"] = 0.03

        save_eigenmode(
            filepath,
            wavenumber=0.2,
            eigenvalue=0.04 + 0.0j,
            tolerance=1e-5,
            minimum_physical_scale=0.03,
            minimum_physical_scale_key="classical.uz_vorticity.nu_vs_ideal",
            minimum_scale_nodes=8,
            resistive_layer_thickness=0.09,
            resistive_layer_nodes=22,
            grid_scaling_factor=1.2,
            current_sheet_nodes=50,
            resolution=128,
            grid=np.linspace(-8, 8, 128),
            mode_scales=scales,
            duz=np.ones(128),
            dbz=np.ones(128),
        )

        params = SimulationParams(
            alpha=0.2,
            data_path=tmpdir,
            mode=0,
            CGL=False,
            S=1e4,
            Pr=1.0,
            a=1.0,
            w=0.0,
            Nmin=64,
            Nmax=128,
            Ninc=32,
        )

        write_results(params, delta_time=1.23)

        dat_path = f"{tmpdir}.dat"
        assert os.path.exists(dat_path)
        with open(dat_path, "r") as f:
            content = f.read()

        # Check table contents
        # δ_in column should contain 3.00000000e-02 (the minimum physical scale)
        assert "3.00000000e-02" in content
        # n_in column should contain 8 (the minimum scale nodes)
        assert " 8 " in content
