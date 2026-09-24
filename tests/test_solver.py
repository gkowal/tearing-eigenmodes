import pytest
import numpy as np
from tearing_eigenmodes import eigenmodes, SimulationParams
from tearing_eigenmodes.io import compile_metadata

def test_eigenmodes_static_vs_dynamic_C():
    # 1. Run standard (static C) convergence
    params_static = SimulationParams(
        alpha=0.15,
        a=1.0,
        w=0.0,
        S=1e3,
        Nmin=64,
        Nmax=512,
        Ninc=32,
        dynamic_C=False,
    )
    σ_stat, _, _, _, _, _, C_stat, N_stat, _, success_stat = eigenmodes(params_static)
    assert success_stat
    assert C_stat is not None and C_stat > 0.0
    assert N_stat is not None and N_stat >= 64

    # 2. Run dynamic C convergence
    params_dynamic = SimulationParams(
        alpha=0.15,
        a=1.0,
        w=0.0,
        S=1e3,
        Nmin=64,
        Nmax=512,
        Ninc=32,
        dynamic_C=True,
    )
    σ_dyn, _, _, _, _, _, C_dyn, N_dyn, _, success_dyn = eigenmodes(params_dynamic)
    assert success_dyn
    assert C_dyn is not None and C_dyn > 0.0
    assert N_dyn is not None and N_dyn >= 64

    # C should have updated and grown in the dynamic case
    assert C_dyn is not None and C_stat is not None and C_dyn > C_stat


def test_eigenmodes_multiscale_diagnostics_populated():
    params = SimulationParams(
        alpha=0.15,
        a=1.0,
        w=0.0,
        S=1e3,
        Pr=0.0,
        Nmin=64,
        Nmax=512,
        Ninc=32,
        dynamic_C=False,
    )
    sigma, s, e, delta_in, nin, nwa, C, N, z, success = eigenmodes(params)
    assert success
    assert delta_in is not None and np.isfinite(delta_in) and delta_in > 0
    assert nin is not None and nin > 0
    assert nwa is not None and nwa > 0


def test_eigenmodes_viscous_scale_limiting():
    # Run with high Pr where viscous layer is smaller than resistive layer
    params = SimulationParams(
        alpha=0.15,
        a=1.0,
        w=0.0,
        S=1e3,
        Pr=5.0,
        Nmin=64,
        Nmax=512,
        Ninc=32,
        dynamic_C=False,
    )
    sigma, s, e, delta_in, nin, nwa, C, N, z, success = eigenmodes(params)
    assert success
    assert delta_in is not None and np.isfinite(delta_in) and delta_in > 0
    assert nin is not None and nin > 0


def test_eigenmodes_verbose_scale_logging(caplog: pytest.LogCaptureFixture):
    import logging
    params = SimulationParams(
        alpha=0.15,
        a=1.0,
        w=0.0,
        S=1e3,
        Pr=0.0,
        Nmin=64,
        Nmax=512,
        Ninc=32,
        verbose=True,
        dynamic_C=False,
    )
    with caplog.at_level(logging.INFO):
        sigma, s, e, delta_in, nin, nwa, C, N, z, success = eigenmodes(params)
    assert success
    # Check that "mode scales:" appears in logs EXACTLY ONCE
    scale_logs = [rec.message for rec in caplog.records if "mode scales:" in rec.message]
    assert len(scale_logs) == 1
    msg = scale_logs[0]
    assert "classical.bz_induction.eta_vs_ideal=" in msg or "classical.bz_induction.eta_vs_f=" in msg
    assert "nan" not in msg


def test_eigenmodes_suppress_scale_summary(caplog: pytest.LogCaptureFixture):
    """When emit_scale_summary=False, scale summary logging must be suppressed."""
    import logging
    params = SimulationParams(
        alpha=0.15,
        a=1.0,
        w=0.0,
        S=1e3,
        Pr=0.0,
        Nmin=64,
        Nmax=512,
        Ninc=32,
        verbose=True,
        emit_scale_summary=False,
        dynamic_C=False,
    )
    with caplog.at_level(logging.INFO):
        sigma, s, e, delta_in, nin, nwa, C, N, z, success = eigenmodes(params)
    assert success
    scale_logs = [rec.message for rec in caplog.records if "mode scales:" in rec.message]
    assert len(scale_logs) == 0


def test_maxima_objective_suppresses_scale_summary(caplog: pytest.LogCaptureFixture):
    """make_objective from eigenmodes-maxima must not emit scale lines during optimizer trials."""
    import logging
    import importlib.util
    spec = importlib.util.spec_from_file_location("maxima_mod", "scripts/eigenmodes-maxima.py")
    assert spec is not None and spec.loader is not None
    maxima_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maxima_mod)

    params = SimulationParams(
        alpha=0.15,
        a=1.0,
        w=0.0,
        S=1e3,
        Pr=0.0,
        Nmin=64,
        Nmax=512,
        Ninc=32,
        verbose=True,
        dynamic_C=False,
    )
    f = maxima_mod.make_objective(params)
    with caplog.at_level(logging.INFO):
        val = f(0.15)
    assert val != 0.0
    scale_logs = [rec.message for rec in caplog.records if "mode scales:" in rec.message]
    assert len(scale_logs) == 0


def test_eigenmodes_all_invalid_scales_behavior(monkeypatch: pytest.MonkeyPatch):
    """When all candidate scales are invalid, delta_in must be nan and nin must be 0."""
    from tearing_eigenmodes.analysis import CLASSICAL_GRID_SCALE_KEYS
    def mock_measure(sys):
        return {k: float("nan") for k in CLASSICAL_GRID_SCALE_KEYS}

    monkeypatch.setattr("tearing_eigenmodes.solver.measure_eigenmode_scales", mock_measure)

    params = SimulationParams(
        alpha=0.15,
        a=1.0,
        w=0.0,
        S=1e3,
        Pr=0.0,
        Nmin=64,
        Nmax=512,
        Ninc=32,
        dynamic_C=False,
    )
    sigma, s, e, delta_in, nin, nwa, C, N, z, success = eigenmodes(params)
    assert success
    assert delta_in is not None and np.isnan(delta_in)
    assert nin == 0
    assert isinstance(nin, int)
    assert np.isnan(s["minimum_physical_scale"])
    assert s["minimum_physical_scale_key"] == ""
    assert s["minimum_scale_nodes"] == 0


def test_eigenmodes_unconverged_solve_behavior():
    """Unconverged solves (tolerance > 1.0) must produce all-nan scale dictionary and nin=0."""
    params = SimulationParams(
        alpha=0.15,
        a=1.0,
        w=0.0,
        S=1e3,
        Pr=0.0,
        Nmin=64,
        Nmax=288,  # select_NC finds Nmin=224; solver exhausts 288 without reaching atol=1e-30
        Ninc=32,
        atol=1e-30,
        rtol=1e-30,
        dynamic_C=False,
    )
    sigma, s, e, delta_in, nin, nwa, C, N, z, success = eigenmodes(params)
    assert success
    assert float(np.atleast_1d(e)[0]) > 1.0
    assert delta_in is not None and np.isnan(delta_in)
    assert nin == 0
    assert isinstance(nin, int)
    assert np.isnan(s["minimum_physical_scale"])
    assert s["minimum_physical_scale_key"] == ""
    assert s["minimum_scale_nodes"] == 0
    from tearing_eigenmodes.analysis import CLASSICAL_DIAGNOSTIC_SCALE_KEYS
    assert len(s["mode_scales"]) == len(CLASSICAL_DIAGNOSTIC_SCALE_KEYS)
    for k, val in s["mode_scales"].items():
        assert np.isnan(val)


def test_solver_converged_selects_induction_minimum_over_smaller_vorticity(monkeypatch: pytest.MonkeyPatch):
    """Converged solver must store all diagnostics but report induction minimum as delta_in."""
    def mock_measure(sys):
        return {
            "classical.bz_induction.eta_vs_ideal": 0.045,
            "classical.bz_induction.eta_vs_ideal_envelope": 0.040,
            "classical.bz_induction.eta_vs_f": 0.050,
            "classical.bz_induction.g_vs_f": np.nan,
            "classical.bz_induction.xi_vs_f": np.nan,
            "classical.uz_vorticity.nu_vs_ideal": 0.015,  # smaller vorticity diagnostic
            "classical.uz_vorticity.nu_vs_ideal_envelope": 0.012,
            "classical.uz_vorticity.nu_vs_f": 0.020,
            "classical.uz_vorticity.g_vs_f": np.nan,
            "classical.uz_vorticity.xi_vs_f": np.nan,
        }

    monkeypatch.setattr("tearing_eigenmodes.solver.measure_eigenmode_scales", mock_measure)

    params = SimulationParams(
        alpha=0.15,
        a=1.0,
        w=0.0,
        S=1e4,
        Pr=0.0,
        Nmin=64,
        Nmax=512,
        Ninc=32,
        dynamic_C=False,
    )
    sigma, s, e, delta_in, nin, nwa, C, N, z, success = eigenmodes(params)
    assert success
    # Must report induction minimum 0.045, NOT smaller vorticity 0.015:
    assert delta_in is not None and np.isclose(delta_in, 0.045)
    assert np.isclose(s["minimum_physical_scale"], 0.045)
    assert s["minimum_physical_scale_key"] == "classical.bz_induction.eta_vs_ideal"
    # All 10 diagnostics remain stored:
    assert len(s["mode_scales"]) == 10
    assert np.isclose(s["mode_scales"]["classical.uz_vorticity.nu_vs_ideal"], 0.015)


def test_format_mode_scale_summary_robustness():
    """format_mode_scale_summary must safely handle non-mappings, mixed key types, canonical key collisions, and sort deterministically."""
    from tearing_eigenmodes.printing import format_mode_scale_summary

    # 1. Non-mapping returns None
    assert format_mode_scale_summary("not-a-mapping") is None
    assert format_mode_scale_summary(None) is None
    assert format_mode_scale_summary([1, 2, 3]) is None

    # 2. Empty or all-invalid returns None
    assert format_mode_scale_summary({}) is None
    assert format_mode_scale_summary({"k1": np.nan, "k2": 0.0, "k3": -1.0, "k4": "bad", "k5": "1.25", "k6": True}) is None

    # 3. Mixed key types (int and str)
    mixed_keys_summary = format_mode_scale_summary({"b": 0.02, 1: 0.01})
    assert mixed_keys_summary == "mode scales: 1=1.0000e-02, b=2.0000e-02"

    # 4. Canonical key collisions: conflicting values are omitted independently of insertion order
    assert format_mode_scale_summary({1: 0.01, "1": 0.02}) is None
    assert format_mode_scale_summary({"1": 0.02, 1: 0.01}) is None

    # 5. Canonical key collisions: identical values are emitted once
    assert format_mode_scale_summary({1: 0.01, "1": 0.01}) == "mode scales: 1=1.0000e-02"
    assert format_mode_scale_summary({"1": 0.01, 1: 0.01}) == "mode scales: 1=1.0000e-02"

    # 6. Unrelated valid keys remain visible when one canonical group has a conflict
    unrelated_summary = format_mode_scale_summary({1: 0.01, "1": 0.02, "b": 0.05})
    assert unrelated_summary == "mode scales: b=5.0000e-02"

    # 7. Numeric and non-numeric string values must be omitted, not converted
    str_val_summary = format_mode_scale_summary({"k1": "1.25", "k2": "bad", "k3": 0.05})
    assert str_val_summary == "mode scales: k3=5.0000e-02"

    # 8. Mixed valid, inf, and invalid
    scales = {
        "z_key": 0.05,
        "a_key": np.inf,
        "b_key": np.nan,
        "c_key": "bad",
        "d_key": 0.001234,
        "e_key": True,
        "f_key": np.array([1.0, 2.0]),
        "g_key": -1.0,
    }
    summary = format_mode_scale_summary(scales)
    assert summary is not None
    # Deterministic alphabetical ordering: a_key, d_key, z_key
    assert summary == "mode scales: a_key=inf, d_key=1.2340e-03, z_key=5.0000e-02"


def test_cached_task_verbose_with_non_mapping_and_mixed_scales(capsys: pytest.CaptureFixture, tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Cached execution with verbose=True must not crash on non-mapping or raw mixed-key mode_scales."""
    import os
    import importlib.util

    # 1. Test compute script task
    spec_c = importlib.util.spec_from_file_location("compute_mod", "scripts/eigenmodes-compute.py")
    assert spec_c is not None and spec_c.loader is not None
    compute_mod = importlib.util.module_from_spec(spec_c)
    spec_c.loader.exec_module(compute_mod)

    # 1a. Non-mapping scales in compute script
    params_c1 = SimulationParams(alpha=0.1, verbose=True, data_path=str(tmp_path))
    fp_c1 = os.path.join(str(tmp_path), f"state_α{0.1:.6e}.npz")
    np.savez_compressed(
        fp_c1,
        wavenumber=0.1,
        eigenvalue=0.05 + 0.0j,
        tolerance=1e-5,
        resolution=128,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        mode_scales="not-a-mapping",
        **compile_metadata(params_c1),
    )
    capsys.readouterr()
    compute_mod.task(0.1, None, None, params_c1)
    out_c1 = capsys.readouterr().out
    scale_lines_c1 = [line for line in out_c1.splitlines() if "mode scales:" in line]
    assert len(scale_lines_c1) == 0

    # 1b. Raw mixed int/str keys in compute script
    params_c2 = SimulationParams(alpha=0.2, verbose=True, data_path=str(tmp_path))
    fp_c2 = os.path.join(str(tmp_path), f"state_α{0.2:.6e}.npz")
    np.savez_compressed(
        fp_c2,
        wavenumber=0.2,
        eigenvalue=0.05 + 0.0j,
        tolerance=1e-5,
        resolution=128,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        mode_scales=np.array({1: 0.01, "b": 0.02, "bad": "1.25"}, dtype=object),
        **compile_metadata(params_c2),
    )
    compute_mod.task(0.2, None, None, params_c2)
    out_c2 = capsys.readouterr().out
    scale_lines_c2 = [line for line in out_c2.splitlines() if "mode scales:" in line]
    assert len(scale_lines_c2) == 1
    assert scale_lines_c2[0] == "mode scales: 1=1.0000e-02, b=2.0000e-02"

    # 2. Test maxima script task
    spec_m = importlib.util.spec_from_file_location("maxima_mod", "scripts/eigenmodes-maxima.py")
    assert spec_m is not None and spec_m.loader is not None
    maxima_mod = importlib.util.module_from_spec(spec_m)
    spec_m.loader.exec_module(maxima_mod)

    def guard_fresh_optimization(*args, **kwargs):
        raise AssertionError("Fresh optimization must not be entered on cached state!")

    monkeypatch.setattr(maxima_mod, "make_objective", guard_fresh_optimization)

    # 2a. Non-mapping scales with correct signed filename
    params_m1 = SimulationParams(dependence="S", S=1e4, verbose=True, data_path=str(tmp_path))
    fp_m1 = os.path.join(str(tmp_path), f"state_S{1e4:+.6e}.npz")
    np.savez_compressed(
        fp_m1,
        scan_parameter_value=1e4,
        wavenumber=0.1,
        eigenvalue=0.05 + 0.0j,
        growth_rate=0.05,
        alpha_max=0.1,
        sigma_max=0.05,
        wavenumber_error=1e-5,
        eigenvalue_error=1e-6,
        niter=10,
        tolerance=1e-5,
        resolution=128,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        minimum_physical_scale=0.02,
        minimum_scale_nodes=5,
        mode_scales="not-a-mapping",
        **compile_metadata(params_m1),
    )
    res_m1 = maxima_mod.task(1e4, None, None, None, params_m1)
    assert res_m1[0] is not None
    out_m1 = capsys.readouterr().out
    scale_lines_m1 = [line for line in out_m1.splitlines() if "mode scales:" in line]
    assert len(scale_lines_m1) == 0

    # 2b. Raw mixed int/str keys with correct signed filename
    params_m2 = SimulationParams(dependence="S", S=2e4, verbose=True, data_path=str(tmp_path))
    fp_m2 = os.path.join(str(tmp_path), f"state_S{2e4:+.6e}.npz")
    np.savez_compressed(
        fp_m2,
        scan_parameter_value=2e4,
        wavenumber=0.1,
        eigenvalue=0.05 + 0.0j,
        growth_rate=0.05,
        alpha_max=0.1,
        sigma_max=0.05,
        wavenumber_error=1e-5,
        eigenvalue_error=1e-6,
        niter=10,
        tolerance=1e-5,
        resolution=128,
        grid_scaling_factor=1.0,
        current_sheet_nodes=20,
        minimum_physical_scale=0.02,
        minimum_scale_nodes=5,
        mode_scales=np.array({1: 0.01, "b": 0.02, "bad": "1.25"}, dtype=object),
        **compile_metadata(params_m2),
    )
    res_m2 = maxima_mod.task(2e4, None, None, None, params_m2)
    assert res_m2[0] is not None
    out_m2 = capsys.readouterr().out
    scale_lines_m2 = [line for line in out_m2.splitlines() if "mode scales:" in line]
    assert len(scale_lines_m2) == 1
    assert scale_lines_m2[0] == "mode scales: 1=1.0000e-02, b=2.0000e-02"


def test_eigenmodes_shift_invert_matches_qz(monkeypatch: pytest.MonkeyPatch):
    """gevp_method='shift-invert' reaches psecas and reproduces QZ's growth rate."""
    import warnings
    import psecas.solver

    calls = []
    original = psecas.solver._eig_shift_invert

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(psecas.solver, "_eig_shift_invert", spy)

    def params(gevp_method: str) -> SimulationParams:
        return SimulationParams(alpha=0.15, a=1.0, w=0.0, S=1e3, Nmin=64,
                                Nmax=512, Ninc=32, gevp_method=gevp_method)

    σ_qz, *_, N_qz, _, ok_qz = eigenmodes(params("qz"))
    assert ok_qz and not calls

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        σ_si, *_, N_si, _, ok_si = eigenmodes(params("shift-invert"))
    assert ok_si
    assert calls and all(calls)
    assert N_si == N_qz
    assert abs(np.atleast_1d(σ_si)[0] - np.atleast_1d(σ_qz)[0]) \
        <= 1e-8 * abs(np.atleast_1d(σ_qz)[0])


def test_eigenmodes_hall_converges_without_scale_diagnostics():
    """Hall branches have no dominance-scale diagnostics; a converged Hall
    eigenmode must still be returned, with NaN scale metadata."""
    params = SimulationParams(S=1e3, Hall=0.1, alpha=0.5, Nmin=64, Nmax=192, Ninc=32)
    σ, s, e, δin, nin, nwa, C, N, z, status = eigenmodes(params)
    assert status is True
    assert e <= 1.0
    assert σ.real > 0.0
    assert np.isnan(δin) and nin == 0
    assert np.isnan(s["minimum_physical_scale"])
    assert s["minimum_physical_scale_key"] == ""
    assert np.isnan(s["resistive_layer_thickness"])
    assert all(np.isnan(v) for v in s["mode_scales"].values())


@pytest.mark.parametrize("Nlow, Nmax, Ninc, expected", [
    (64, 128, 32, [64, 96, 128]),
    (100, 200, 32, [100, 132, 164, 196, 200]),
    (64, 100, 32, [64, 96, 100]),
    (64, 64, 32, [64]),
])
def test_resolution_sequence_never_exceeds_nmax(Nlow, Nmax, Ninc, expected):
    from tearing_eigenmodes.solver import resolution_sequence
    assert resolution_sequence(Nlow, Nmax, Ninc).tolist() == expected


def test_eigenmodes_respects_nmax_with_misaligned_range():
    params = SimulationParams(S=1e3, alpha=0.5, Nmin=64, Nmax=100, Ninc=32,
                              rtol=1e-15, atol=1e-30)
    *_, N, _, status = eigenmodes(params)
    assert status is True
    assert N == 100


def test_eigenmodes_single_resolution_reports_unconverged():
    """Nmin == Nmax leaves one resolution: the solve must return the mode
    as unconverged (infinite error) rather than fail."""
    params = SimulationParams(S=1e3, alpha=0.5, Nmin=64, Nmax=64, Ninc=32)
    σ, s, e, δin, nin, nwa, C, N, z, status = eigenmodes(params)
    assert status is True
    assert N == 64
    assert np.isinf(e)
    assert σ.real > 0.0


class _StubSolver:
    """Deterministic stand-in for psecas: three real modes whose full solves
    drift with N (so the first comparison does not converge) and whose
    guided solves reproduce the guess."""

    def __init__(self):
        from types import SimpleNamespace
        self.grid = SimpleNamespace(N=0, C=1.0, zg=np.zeros(3))
        self.system = SimpleNamespace(result={})
        self.guesses = []

    def solve_full(self):
        base = np.array([0.1, 0.3, 0.2], dtype=complex)  # unsorted on purpose
        return base * (1.0 + 0.1 / self.grid.N), np.eye(3, dtype=complex)

    def filter_modes(self, Σ, V, re_range=None, im_range=None):
        return Σ, V

    def solve_mode(self, σ0, v0=None, useOPinv=True, verbose=False):
        self.guesses.append(σ0)
        return σ0 * (1.0 + 1e-12), v0

    def prolongate_eigenvector(self, v, grid_old):
        return v

    def keep_result(self, σ, v, mode):
        self.system.result["sigma"] = σ


def _stub_iterate(**kwargs):
    from tearing_eigenmodes.solver import TearingSolver
    stub = _StubSolver()
    out = TearingSolver.iterate_solve_multimode(
        stub, [64, 96, 128, 160], orderby="real", rtol=1e-5, atol=1e-10,
        gtol=1e-2, **kwargs)
    return stub, out


def test_iterate_tracks_requested_mode_through_guess_path():
    stub, (σ, v, e) = _stub_iterate(maxmode=1)
    assert e <= 1.0
    assert σ.real == pytest.approx(0.2, rel=1e-2)
    assert stub.guesses and all(g.real == pytest.approx(0.2, rel=1e-2) for g in stub.guesses)


def test_iterate_rejects_missing_mode_instead_of_clamping():
    from tearing_eigenmodes.exceptions import DeltaError
    with pytest.raises(DeltaError, match="Requested mode 5"):
        _stub_iterate(maxmode=5)


def test_iterate_allmodes_clamps_to_available_modes():
    _, (Σ, V, errors) = _stub_iterate(maxmode=5, allmodes=True)
    assert Σ.size == 3
