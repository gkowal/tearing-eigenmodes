from tearing_eigenmodes.io import build_dpath, check_state, compile_metadata, save_eigenmode
from tearing_eigenmodes.params import SimulationParams
import numpy as np
import os


def test_model_tag_distinguishes_classical_and_cgl():
    classical = SimulationParams(
        CGL=False,
        plasma_beta=None,
        plasma_beta_difference=None,
        zeta=None,
    )
    cgl = SimulationParams(
        CGL=True,
        plasma_beta=None,
        plasma_beta_difference=None,
        eos=None,
        parallel_index=None,
        perpendicular_index=None,
    )
    classical_dpath = build_dpath(classical)
    cgl_dpath = build_dpath(cgl)
    assert classical_dpath != cgl_dpath
    assert "MHD" in classical_dpath
    assert "CGL" in cgl_dpath


def test_cgl_eos_distinguishes_dirs():
    adiabatic = SimulationParams(CGL=True, eos="adiabatic")
    isothermal = SimulationParams(CGL=True, eos="isothermal")
    assert build_dpath(adiabatic) != build_dpath(isothermal)
    assert "adiabatic" in build_dpath(adiabatic)
    assert "isothermal" in build_dpath(isothermal)


def test_cgl_gamma_indices_distinguish_dirs():
    default = SimulationParams(
        CGL=True, parallel_index=3.0, perpendicular_index=2.0
    )
    custom = SimulationParams(
        CGL=True, parallel_index=1.5, perpendicular_index=1.0
    )
    parallel_only = SimulationParams(
        CGL=True, parallel_index=1.5, perpendicular_index=2.0
    )
    assert build_dpath(default) != build_dpath(custom)
    assert build_dpath(default) != build_dpath(parallel_only)


def test_noshear_distinguishes_dirs():
    shear = SimulationParams(CGL=False, noshear=False)
    noshear = SimulationParams(CGL=False, noshear=True)
    shear_dpath = build_dpath(shear)
    noshear_dpath = build_dpath(noshear)
    assert shear_dpath != noshear_dpath
    assert "noshear" not in shear_dpath
    assert "noshear" in noshear_dpath


def test_zeta_distinguishes_dirs():
    low = SimulationParams(CGL=False, zeta=1.0)
    high = SimulationParams(CGL=False, zeta=2.0)
    assert build_dpath(low) != build_dpath(high)
    assert f"ζ{1.0:.3e}" in build_dpath(low)
    assert f"ζ{2.0:.3e}" in build_dpath(high)


def test_swept_none_params_omitted():
    swept = SimulationParams(CGL=False, S=None, zeta=None)
    swept_dpath = build_dpath(swept)
    assert f"S{1e4:.3e}" not in swept_dpath
    assert "ζ" not in swept_dpath

    swept_cgl = SimulationParams(
        CGL=True, eos=None, parallel_index=None, perpendicular_index=None
    )
    swept_cgl_dpath = build_dpath(swept_cgl)
    assert "ɣpar" not in swept_cgl_dpath
    assert "ɣper" not in swept_cgl_dpath
    assert "adiabatic" not in swept_cgl_dpath


def test_suffix_preserved():
    classical = SimulationParams(CGL=False, suffix="_test123")
    cgl = SimulationParams(CGL=True, suffix="_test123")
    assert build_dpath(classical).endswith("_test123")
    assert build_dpath(cgl).endswith("_test123")


def test_delta_beta_distinguishes_dirs_scientific_notation():
    low = SimulationParams(CGL=True, plasma_beta_difference=0.001)
    high = SimulationParams(CGL=True, plasma_beta_difference=0.004)
    low_dpath = build_dpath(low)
    high_dpath = build_dpath(high)
    assert low_dpath != high_dpath
    assert f"Δβ{0.001:+.3e}" in low_dpath
    assert f"Δβ{0.004:+.3e}" in high_dpath

    lower = SimulationParams(CGL=True, plasma_beta_difference=0.005)
    upper = SimulationParams(CGL=True, plasma_beta_difference=0.014)
    lower_dpath = build_dpath(lower)
    upper_dpath = build_dpath(upper)
    assert lower_dpath != upper_dpath
    assert f"Δβ{0.005:+.3e}" in lower_dpath
    assert f"Δβ{0.014:+.3e}" in upper_dpath


def test_default_classical_dir_contains_core_segments():
    dpath = build_dpath(SimulationParams())
    assert f"S{1e4:.3e}" in dpath
    assert f"Pr{0.0:.3e}" in dpath
    assert f"a{1.0:.3e}" in dpath
    assert f"w{0.0:.3e}" in dpath


def _write_config_state(file_path: str, params: SimulationParams) -> None:
    """Save a converged state carrying the run-config metadata of params."""
    save_eigenmode(
        file_path,
        wavenumber=0.1,
        eigenvalue=0.05 + 0.0j,
        tolerance=1e-5,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=128,
        grid=np.linspace(-5.0, 5.0, 8),
        **compile_metadata(params),
    )


def test_check_state_matching_metadata_reuses(tmp_path):
    params = SimulationParams()
    fp = os.path.join(str(tmp_path), "state_match.npz")
    _write_config_state(fp, params)
    status, data = check_state(fp, params=SimulationParams())
    assert status is True
    assert data is not None


def test_check_state_changed_lundquist_forces_recompute(tmp_path):
    fp = os.path.join(str(tmp_path), "state_S.npz")
    _write_config_state(fp, SimulationParams(S=1e4))
    status, data = check_state(fp, params=SimulationParams(S=1e5))
    assert status is False
    assert data is None


def test_check_state_changed_model_flag_forces_recompute(tmp_path):
    fp = os.path.join(str(tmp_path), "state_cgl.npz")
    _write_config_state(fp, SimulationParams(CGL=False))
    status, data = check_state(fp, params=SimulationParams(CGL=True))
    assert status is False
    assert data is None


def test_check_state_legacy_file_missing_metadata_forces_recompute(tmp_path):
    fp = os.path.join(str(tmp_path), "state_legacy.npz")
    np.savez_compressed(fp, wavenumber=0.1, tolerance=1e-5, resolution=128)
    status, data = check_state(fp, params=SimulationParams())
    assert status is False
    assert data is None


def test_check_state_params_none_preserves_tolerance_only_behavior(tmp_path):
    fp = os.path.join(str(tmp_path), "state_none.npz")
    np.savez_compressed(fp, wavenumber=0.1, tolerance=1e-5, resolution=128)
    status, data = check_state(fp, params=None)
    assert status is True
    assert data is not None


def test_save_eigenmode_temp_hidden_from_npz_glob(tmp_path, monkeypatch):
    """Mid-save temp file must not match the *.npz globs used by readers."""
    import fnmatch
    import glob

    import tearing_eigenmodes.io as io_module

    real_savez = io_module.np.savez_compressed
    target = os.path.join(str(tmp_path), "state.npz")
    seen = {}

    def recorder(file, *args, **kwargs):
        tmp_name = getattr(file, "name", file)
        seen["tmp"] = tmp_name
        base = os.path.basename(tmp_name)
        assert not fnmatch.fnmatch(base, "*.npz"), (
            f"atomic-write temp {base!r} matches *.npz glob"
        )
        assert os.path.dirname(os.path.abspath(tmp_name)) == os.path.dirname(
            os.path.abspath(target)
        )
        return real_savez(file, *args, **kwargs)

    monkeypatch.setattr(io_module.np, "savez_compressed", recorder)
    save_eigenmode(target, wavenumber=0.42, tolerance=1e-5, resolution=128)

    assert seen["tmp"] is not None
    assert os.path.exists(target)
    assert glob.glob(os.path.join(str(tmp_path), "*.npz.tmp")) == []
    with np.load(target) as loaded:
        assert float(loaded["wavenumber"]) == 0.42
