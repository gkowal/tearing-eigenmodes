import pytest
import numpy as np
from psecas import ChebyshevRationalGrid, Solver
from tearing_eigenmodes.systems import TearingGyrotropicMHD
from tearing_eigenmodes.analysis import (
    evaluate_cgl_induction_terms,
    measure_cgl_scales,
    measure_eigenmode_scales,
    CGL_GRID_SCALE_KEYS,
)


class MockCGLSystem:
    def __init__(
        self,
        grid,
        kx=0.5,
        a=1.0,
        S=1e4,
        epsilon=0.0,
        ky=0.0,
    ):
        self.grid = grid
        self.kx = kx
        self.a = a
        self.S = S
        self.ϵ = epsilon
        self.ky = ky
        self.η = 1.0 / S if S > 0 else 0.0
        self.Bx = np.tanh(grid.zg / a)
        self.By = 1.0 / np.cosh(grid.zg / a)
        self.result = {}


def test_cgl_all_keys_present_and_types():
    grid = ChebyshevRationalGrid(N=64, C=1.0, max_derivative_order=4)
    system = MockCGLSystem(grid, kx=0.5, a=1.0, S=1e4)
    system.result["sigma"] = 0.05 + 0.0j
    system.result["duy"] = np.exp(-(grid.zg / 0.3)**2)
    system.result["duz"] = np.exp(-(grid.zg / 0.35)**2)
    system.result["dby"] = np.exp(-(grid.zg / 0.4)**2)
    system.result["dbz"] = np.exp(-(grid.zg / 0.45)**2)

    scales = measure_eigenmode_scales(system)
    assert len(scales) == len(CGL_GRID_SCALE_KEYS)
    for key in CGL_GRID_SCALE_KEYS:
        assert key in scales
        val = scales[key]
        assert isinstance(val, float)
        assert np.isfinite(val) or np.isnan(val) or np.isinf(val)

    # Ensure no classical keys or vorticity/pressure keys are present
    for k in scales:
        assert k.startswith("cgl.")
        assert "vorticity" not in k
        assert "pressure" not in k
        assert "g_vs_f" not in k
        assert "xi_vs_f" not in k


def test_cgl_scale_invariance_under_eigenvector_scaling():
    grid = ChebyshevRationalGrid(N=64, C=1.0, max_derivative_order=4)
    system = MockCGLSystem(grid, kx=0.5, a=1.0, S=1e4)
    system.result["sigma"] = 0.03 + 0.01j
    system.result["duy"] = np.exp(-(grid.zg / 0.25)**2) * (1.0 + 0.5j)
    system.result["duz"] = np.exp(-(grid.zg / 0.30)**2) * (0.5 - 0.2j)
    system.result["dby"] = np.exp(-(grid.zg / 0.35)**2) * (0.8 - 0.3j)
    system.result["dbz"] = np.exp(-(grid.zg / 0.40)**2) * (0.7 + 0.4j)

    scales1 = measure_cgl_scales(system)

    # Multiply eigenvector by complex constant
    c = 2.7 - 1.9j
    system.result["duy"] = system.result["duy"] * c
    system.result["duz"] = system.result["duz"] * c
    system.result["dby"] = system.result["dby"] * c
    system.result["dbz"] = system.result["dbz"] * c

    scales2 = measure_cgl_scales(system)

    for k in CGL_GRID_SCALE_KEYS:
        v1 = scales1[k]
        v2 = scales2[k]
        if np.isfinite(v1):
            assert np.isclose(v1, v2, rtol=1e-12)
        elif np.isnan(v1):
            assert np.isnan(v2)
        elif np.isinf(v1):
            assert np.isinf(v2)


def test_cgl_a_not_one_uses_kx():
    grid = ChebyshevRationalGrid(N=64, C=1.0, max_derivative_order=4)
    a = 2.0
    alpha = 0.6
    kx = alpha / a

    system = MockCGLSystem(grid, kx=kx, a=a, S=1e4)
    system.result["sigma"] = 0.05 + 0.0j
    system.result["duy"] = np.exp(-(grid.zg / 0.3)**2)
    system.result["duz"] = np.exp(-(grid.zg / 0.35)**2)
    system.result["dby"] = np.exp(-(grid.zg / 0.4)**2)
    system.result["dbz"] = np.exp(-(grid.zg / 0.45)**2)

    profiles = evaluate_cgl_induction_terms(system)
    expected_T_F_bz = kx * system.Bx * system.result["duz"]
    assert np.allclose(profiles.T_F_bz, expected_T_F_bz)

    expected_T_F_by = -kx * system.Bx * system.result["duy"] + (system.Bx * system.By / a) * system.result["duz"]
    assert np.allclose(profiles.T_F_by, expected_T_F_by)


def test_cgl_hall_rejection():
    grid = ChebyshevRationalGrid(N=64, C=1.0, max_derivative_order=4)
    system = MockCGLSystem(grid, epsilon=0.1)
    system.result["sigma"] = 0.05 + 0.0j
    system.result["duy"] = np.ones_like(grid.zg)
    system.result["duz"] = np.ones_like(grid.zg)
    system.result["dby"] = np.ones_like(grid.zg)
    system.result["dbz"] = np.ones_like(grid.zg)

    with pytest.raises(NotImplementedError, match="Hall"):
        measure_eigenmode_scales(system)


def test_cgl_equation_closure_on_converged_eigenmode():
    # Solve a modest CGL/Gyrotropic MHD eigenmode and verify induction residuals
    grid = ChebyshevRationalGrid(N=64, C=0.5, max_derivative_order=4)
    system = TearingGyrotropicMHD(grid, periodic=False, kx=0.2, a=1.0, S=1e3, Pr=0.0, β=1.0, Δβ=0.0, ϵ=0.0)
    solver = Solver(grid, system)

    sigma, v = solver.solve()
    assert np.isfinite(sigma.real)

    solver.keep_result(sigma, v, 0)
    sys_result = getattr(system, "result")
    sys_result["converged"] = True
    sys_result["error"] = 0.0

    profiles = evaluate_cgl_induction_terms(system)

    # Residuals
    # by induction: L_by - (T_F_by + T_eta_by)
    active_by = profiles.T_F_by
    if profiles.T_eta_by is not None:
        active_by = active_by + profiles.T_eta_by
    R_by = profiles.L_by - active_by

    # bz induction: L_bz - (T_F_bz + T_eta_bz)
    active_bz = profiles.T_F_bz
    if profiles.T_eta_bz is not None:
        active_bz = active_bz + profiles.T_eta_bz
    R_bz = profiles.L_bz - active_bz

    I = np.where(np.abs(grid.zg) <= 1.0)[0]

    norm_Rby = np.max(np.abs(R_by[I]))
    scale_by = np.max(np.abs(profiles.L_by[I]))
    rel_err_by = norm_Rby / max(scale_by, 1e-14)
    assert rel_err_by < 1e-3, f"CGL by induction residual too large: {rel_err_by}"

    norm_Rbz = np.max(np.abs(R_bz[I]))
    scale_bz = np.max(np.abs(profiles.L_bz[I]))
    rel_err_bz = norm_Rbz / max(scale_bz, 1e-14)
    assert rel_err_bz < 1e-3, f"CGL bz induction residual too large: {rel_err_bz}"
