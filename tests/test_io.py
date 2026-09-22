from tearing_eigenmodes.io import build_dpath, check_state, compile_metadata, load_state_data, save_eigenmode
from tearing_eigenmodes.params import SimulationParams
import logging
import numpy as np
import os
import pytest


def test_no_model_tag_or_noshear_marker_in_dpath():
    classical = SimulationParams(
        CGL=False,
        plasma_beta=None,
        plasma_beta_difference=None,
        zeta=None,
        noshear=True,
    )
    cgl = SimulationParams(
        CGL=True,
        plasma_beta=None,
        plasma_beta_difference=None,
        eos=None,
        parallel_index=None,
        perpendicular_index=None,
    )
    for dpath in (build_dpath(classical), build_dpath(cgl)):
        assert "MHD" not in dpath
        assert "CGL" not in dpath
        assert "noshear" not in dpath


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


def test_noshear_flag_leaves_dpath_unchanged():
    shear = SimulationParams(CGL=False, noshear=False)
    noshear = SimulationParams(CGL=False, noshear=True)
    assert build_dpath(shear) == build_dpath(noshear)
    assert "noshear" not in build_dpath(noshear)


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


def test_compile_metadata_carries_zeta():
    assert compile_metadata(SimulationParams())["zeta"] == 1.0
    assert compile_metadata(SimulationParams(zeta=0.3))["zeta"] == 0.3


def test_zeta_round_trip_through_save_load(tmp_path):
    fp = os.path.join(str(tmp_path), "state_zeta.npz")
    _write_config_state(fp, SimulationParams(zeta=0.3))
    data = load_state_data(fp)
    assert data is not None
    assert data["zeta"] == 0.3


def test_check_state_changed_zeta_forces_recompute(tmp_path):
    fp = os.path.join(str(tmp_path), "state_zeta.npz")
    _write_config_state(fp, SimulationParams(zeta=1.0))
    status, data = check_state(fp, params=SimulationParams(zeta=1.0))
    assert status is True
    assert data is not None
    status, data = check_state(fp, params=SimulationParams(zeta=0.3))
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


def _write_config_state_at_resolution(
    file_path: str, params: SimulationParams, resolution: int
) -> None:
    """Save a converged state like _write_config_state at a given N."""
    save_eigenmode(
        file_path,
        wavenumber=0.1,
        eigenvalue=0.05 + 0.0j,
        tolerance=1e-5,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=resolution,
        grid=np.linspace(-5.0, 5.0, 8),
        **compile_metadata(params),
    )


def test_check_state_numerics_only_changes_reuse(tmp_path):
    base = SimulationParams()
    fp = os.path.join(str(tmp_path), "state_numerics.npz")
    _write_config_state(fp, base)
    variants = [
        SimulationParams(Nmin=128),
        SimulationParams(Nmax=4096),
        SimulationParams(rtol=1e-8),
        SimulationParams(n_inner_scale=8),
        SimulationParams(inner_scale=1e-3),
    ]
    for variant in variants:
        status, data = check_state(fp, params=variant)
        assert status is True
        assert data is not None


def test_check_state_nmin_floor_forces_recompute(tmp_path):
    fp = os.path.join(str(tmp_path), "state_floor.npz")
    _write_config_state_at_resolution(fp, SimulationParams(), 64)
    status, data = check_state(fp, params=SimulationParams(Nmin=128))
    assert status is False
    assert data is None


def test_check_state_nmin_floor_boundary_reuses(tmp_path):
    fp = os.path.join(str(tmp_path), "state_floor_ok.npz")
    _write_config_state_at_resolution(fp, SimulationParams(), 128)
    for nmin in (128, 64):
        status, data = check_state(fp, params=SimulationParams(Nmin=nmin))
        assert status is True
        assert data is not None


def test_check_state_physics_and_mode_changes_force_recompute(tmp_path):
    fp = os.path.join(str(tmp_path), "state_phys.npz")
    _write_config_state(fp, SimulationParams())
    for variant in (
        SimulationParams(a=2.0),
        SimulationParams(f_outer=0.05),
        SimulationParams(mode="tearing"),
    ):
        status, data = check_state(fp, params=variant)
        assert status is False
        assert data is None


def test_check_state_nmin_floor_skipped_when_params_none(tmp_path):
    fp = os.path.join(str(tmp_path), "state_noparam.npz")
    np.savez_compressed(fp, wavenumber=0.1, tolerance=1e-5, resolution=8)
    status, data = check_state(fp, params=None)
    assert status is True
    assert data is not None


def test_check_state_nmin_floor_logs_debug_not_warning(tmp_path, caplog):
    fp = os.path.join(str(tmp_path), "state_floor_log.npz")
    _write_config_state_at_resolution(fp, SimulationParams(), 64)
    with caplog.at_level(logging.DEBUG, logger="tearing_eigenmodes.io"):
        status, data = check_state(fp, params=SimulationParams(Nmin=128))
    assert status is False
    assert data is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        r.levelno == logging.DEBUG and "below" in r.getMessage()
        for r in caplog.records
    )


def test_check_state_missing_key_logs_debug_not_warning(tmp_path, caplog):
    params = SimulationParams()
    meta = compile_metadata(params)
    meta.pop("S")
    fp = os.path.join(str(tmp_path), "state_missing_key.npz")
    save_eigenmode(
        fp,
        wavenumber=0.1,
        eigenvalue=0.05 + 0.0j,
        tolerance=1e-5,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        resolution=128,
        grid=np.linspace(-5.0, 5.0, 8),
        **meta,
    )
    with caplog.at_level(logging.DEBUG, logger="tearing_eigenmodes.io"):
        status, data = check_state(fp, params=SimulationParams())
    assert status is False
    assert data is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        r.levelno == logging.DEBUG and "missing run-config key" in r.getMessage()
        for r in caplog.records
    )


def test_check_state_mismatch_logs_debug_not_warning(tmp_path, caplog):
    fp = os.path.join(str(tmp_path), "state_mismatch_log.npz")
    _write_config_state(fp, SimulationParams(S=1e4))
    with caplog.at_level(logging.DEBUG, logger="tearing_eigenmodes.io"):
        status, data = check_state(fp, params=SimulationParams(S=1e5))
    assert status is False
    assert data is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        r.levelno == logging.DEBUG and "run-config mismatch" in r.getMessage()
        for r in caplog.records
    )


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


def test_save_eigenmode_new_files_load_without_pickle(tmp_path):
    """New files carry no pickled object arrays; legacy ones still load."""
    fp = os.path.join(str(tmp_path), "state_new.npz")
    scales = {"b_key": 0.02, "a_key": 0.01}
    grid = np.linspace(-5.0, 5.0, 8)
    save_eigenmode(
        fp,
        wavenumber=0.1,
        eigenvalue=0.05 + 0.0j,
        tolerance=1e-5,
        resolution=128,
        grid=grid,
        minimum_physical_scale=0.01,
        minimum_physical_scale_key="a_key",
        minimum_scale_nodes=4,
        mode_scales=scales,
    )
    with np.load(fp, allow_pickle=False) as npz:
        assert float(npz["wavenumber"]) == 0.1
        assert np.array_equal(npz["grid"], grid)
        assert [str(k) for k in npz["mode_scale_keys"]] == sorted(scales)
        assert np.allclose(
            npz["mode_scale_values"],
            [scales[k] for k in sorted(scales)],
        )
    legacy = os.path.join(str(tmp_path), "state_legacy_obj.npz")
    np.savez_compressed(
        legacy,
        wavenumber=0.1,
        tolerance=1e-5,
        resolution=128,
        mode_scales=np.array({"a_key": 0.01}, dtype=object),
    )
    data = load_state_data(legacy)
    assert data is not None
    assert float(data["wavenumber"]) == 0.1
    # None-valued optional metadata (e.g. inner_scale/mode default None)
    # round-trips through load_state_data but is stored as pickled object
    # arrays, so such files require allow_pickle=True (which loaders keep).
    meta = compile_metadata(SimulationParams())
    assert meta["inner_scale"] is None and meta["mode"] is None
    fp_none = os.path.join(str(tmp_path), "state_none_meta.npz")
    save_eigenmode(
        fp_none,
        wavenumber=0.1,
        tolerance=1e-5,
        resolution=128,
        **meta,
    )
    data_none = load_state_data(fp_none)
    assert data_none is not None
    assert data_none["inner_scale"] is None
    assert data_none["mode"] is None
    with np.load(fp_none, allow_pickle=False) as npz:
        with pytest.raises(ValueError):
            for key in npz.files:
                npz[key]
