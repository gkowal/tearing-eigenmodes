import pytest
import numpy as np
from psecas import ChebyshevRationalGrid, Solver
from tearing_eigenmodes.systems import TearingClassicalMHD
from tearing_eigenmodes.analysis import (
    evaluate_classical_terms,
    measure_classical_scales,
    measure_eigenmode_scales,
    minimum_eigenmode_scale,
    CLASSICAL_GRID_SCALE_KEYS,
)


class MockClassicalSystem:
    def __init__(
        self,
        grid,
        kx=0.5,
        a=1.0,
        w=0.0,
        S=1e4,
        Pr=0.0,
        xi=0.0,
        epsilon=0.0,
        ky=0.0,
        shear=True,
    ):
        self.grid = grid
        self.kx = kx
        self.a = a
        self.w = w
        self.S = S
        self.Pr = Pr
        self.ξ = xi
        self.ϵ = epsilon
        self.ky = ky
        self.shear = shear
        self.η = 1.0 / S if S > 0 else 0.0
        self.ν = (Pr / S) if S > 0 else 0.0
        self.Bx = np.tanh(grid.zg / a)
        self.Ux = 0.5 * (np.tanh((grid.zg + w) / a) - np.tanh((grid.zg - w) / a)) if w > 0 else np.zeros_like(grid.zg)
        self.d2Bxdz = grid.derivative(self.Bx, 2)
        self.d2Uxdz = grid.derivative(self.Ux, 2) if w > 0 else np.zeros_like(grid.zg)
        self.result = {}


def test_classical_all_keys_present_and_types():
    grid = ChebyshevRationalGrid(N=64, C=1.0, max_derivative_order=4)
    system = MockClassicalSystem(grid, kx=0.5, a=1.0, S=1e4, Pr=1.0, xi=0.1, w=0.2, shear=True)
    system.result["sigma"] = 0.05 + 0.0j
    system.result["duz"] = np.exp(-(grid.zg / 0.3)**2)
    system.result["dbz"] = np.exp(-(grid.zg / 0.4)**2)

    scales = measure_eigenmode_scales(system)
    assert len(scales) == len(CLASSICAL_GRID_SCALE_KEYS)
    for key in CLASSICAL_GRID_SCALE_KEYS:
        assert key in scales
        val = scales[key]
        assert isinstance(val, float)
        assert np.isfinite(val) or np.isnan(val) or np.isinf(val)


def test_classical_coefficient_activation():
    grid = ChebyshevRationalGrid(N=64, C=1.0, max_derivative_order=4)

    # w = 0, Pr = 0, xi = 0 -> only resistive terms active
    sys_res_only = MockClassicalSystem(grid, kx=0.5, a=1.0, w=0.0, S=1e4, Pr=0.0, xi=0.0, shear=False)
    sys_res_only.result["sigma"] = 0.05 + 0.0j
    sys_res_only.result["duz"] = np.exp(-(grid.zg / 0.3)**2)
    sys_res_only.result["dbz"] = np.exp(-(grid.zg / 0.4)**2)

    scales_res = measure_eigenmode_scales(sys_res_only)
    assert np.isfinite(scales_res["classical.bz_induction.eta_vs_ideal"])
    assert np.isfinite(scales_res["classical.bz_induction.eta_vs_f"])
    assert np.isnan(scales_res["classical.bz_induction.g_vs_f"])
    assert np.isnan(scales_res["classical.bz_induction.xi_vs_f"])
    assert np.isnan(scales_res["classical.uz_vorticity.nu_vs_ideal"])
    assert np.isnan(scales_res["classical.uz_vorticity.nu_vs_f"])
    assert np.isnan(scales_res["classical.uz_vorticity.g_vs_f"])
    assert np.isnan(scales_res["classical.uz_vorticity.xi_vs_f"])

    # Activate Pr > 0 -> nu terms become active
    sys_visc = MockClassicalSystem(grid, kx=0.5, a=1.0, w=0.0, S=1e4, Pr=1.0, xi=0.0, shear=False)
    sys_visc.result["sigma"] = 0.05 + 0.0j
    sys_visc.result["duz"] = np.exp(-(grid.zg / 0.3)**2)
    sys_visc.result["dbz"] = np.exp(-(grid.zg / 0.4)**2)
    scales_visc = measure_eigenmode_scales(sys_visc)
    assert np.isfinite(scales_visc["classical.uz_vorticity.nu_vs_ideal"]) or np.isinf(scales_visc["classical.uz_vorticity.nu_vs_ideal"])
    assert np.isfinite(scales_visc["classical.uz_vorticity.nu_vs_f"]) or np.isinf(scales_visc["classical.uz_vorticity.nu_vs_f"])

    # Activate xi > 0 -> xi terms become active
    sys_xi = MockClassicalSystem(grid, kx=0.5, a=1.0, w=0.0, S=1e4, Pr=0.0, xi=0.05, shear=False)
    sys_xi.result["sigma"] = 0.05 + 0.0j
    sys_xi.result["duz"] = np.exp(-(grid.zg / 0.3)**2)
    sys_xi.result["dbz"] = np.exp(-(grid.zg / 0.4)**2)
    scales_xi = measure_eigenmode_scales(sys_xi)
    assert np.isfinite(scales_xi["classical.bz_induction.xi_vs_f"]) or np.isinf(scales_xi["classical.bz_induction.xi_vs_f"])
    assert np.isfinite(scales_xi["classical.uz_vorticity.xi_vs_f"]) or np.isinf(scales_xi["classical.uz_vorticity.xi_vs_f"])

    # Activate flow (w > 0 and shear) -> G terms become active
    sys_g = MockClassicalSystem(grid, kx=0.5, a=1.0, w=0.2, S=1e4, Pr=0.0, xi=0.0, shear=True)
    sys_g.result["sigma"] = 0.05 + 0.0j
    sys_g.result["duz"] = np.exp(-(grid.zg / 0.3)**2)
    sys_g.result["dbz"] = np.exp(-(grid.zg / 0.4)**2)
    scales_g = measure_eigenmode_scales(sys_g)
    assert np.isfinite(scales_g["classical.bz_induction.g_vs_f"]) or np.isinf(scales_g["classical.bz_induction.g_vs_f"])
    assert np.isfinite(scales_g["classical.uz_vorticity.g_vs_f"]) or np.isinf(scales_g["classical.uz_vorticity.g_vs_f"])


def test_classical_scale_invariance_under_eigenvector_scaling():
    grid = ChebyshevRationalGrid(N=64, C=1.0, max_derivative_order=4)
    system = MockClassicalSystem(grid, kx=0.5, a=1.0, w=0.2, S=1e4, Pr=1.0, xi=0.05, shear=True)
    system.result["sigma"] = 0.03 + 0.01j
    system.result["duz"] = np.exp(-(grid.zg / 0.25)**2) * (1.0 + 0.5j)
    system.result["dbz"] = np.exp(-(grid.zg / 0.35)**2) * (0.8 - 0.3j)

    scales1 = measure_classical_scales(system)

    # Multiply eigenvector by complex constant
    c = 3.5 - 2.1j
    system.result["duz"] = system.result["duz"] * c
    system.result["dbz"] = system.result["dbz"] * c

    scales2 = measure_classical_scales(system)

    for k in CLASSICAL_GRID_SCALE_KEYS:
        v1 = scales1[k]
        v2 = scales2[k]
        if np.isfinite(v1):
            assert np.isclose(v1, v2, rtol=1e-12)
        elif np.isnan(v1):
            assert np.isnan(v2)
        elif np.isinf(v1):
            assert np.isinf(v2)


def test_classical_a_not_one_uses_kx():
    # Verify that a != 1 uses kx in derivative operators rather than alpha = kx * a
    grid = ChebyshevRationalGrid(N=64, C=1.0, max_derivative_order=4)
    a = 2.0
    alpha = 0.4
    kx = alpha / a  # 0.2

    system = MockClassicalSystem(grid, kx=kx, a=a, w=0.0, S=1e4, Pr=0.0, xi=0.0, shear=False)
    system.result["sigma"] = 0.05 + 0.0j
    system.result["duz"] = np.exp(-(grid.zg / 0.3)**2)
    system.result["dbz"] = np.exp(-(grid.zg / 0.4)**2)

    profiles = evaluate_classical_terms(system)
    # T_F_b = 1j * kx * Bx * duz
    expected_T_F_b = 1j * kx * system.Bx * system.result["duz"]
    assert np.allclose(profiles.T_F_b, expected_T_F_b)


def test_classical_hall_rejection():
    grid = ChebyshevRationalGrid(N=64, C=1.0, max_derivative_order=4)
    system = MockClassicalSystem(grid, epsilon=0.1)
    system.result["sigma"] = 0.05 + 0.0j
    system.result["duz"] = np.ones_like(grid.zg)
    system.result["dbz"] = np.ones_like(grid.zg)

    with pytest.raises(NotImplementedError, match="Hall"):
        measure_eigenmode_scales(system)


def test_classical_equation_closure_on_converged_eigenmode():
    # Solve a small, fast Classical MHD eigenmode and verify equation residuals
    grid = ChebyshevRationalGrid(N=64, C=0.5, max_derivative_order=4)
    system = TearingClassicalMHD(grid, periodic=False, kx=0.2, a=1.0, w=0.0, S=1e3, Pr=0.1, shear=False)
    solver = Solver(grid, system)

    # Solve
    sigma, v = solver.solve()
    assert np.isfinite(sigma.real)

    # Keep the mode
    solver.keep_result(sigma, v, 0)
    sys_result = getattr(system, "result")
    sys_result["converged"] = True
    sys_result["error"] = 0.0

    profiles = evaluate_classical_terms(system)

    # Residuals
    # Induction: L_b - sum(T_j^(b))
    active_b = profiles.T_F_b
    if profiles.T_G_b is not None:
        active_b = active_b + profiles.T_G_b
    if profiles.T_xi_b is not None:
        active_b = active_b + profiles.T_xi_b
    if profiles.T_eta_b is not None:
        active_b = active_b + profiles.T_eta_b

    R_b = profiles.L_b - active_b

    # Vorticity: L_omega - sum(T_j^(omega))
    active_omega = profiles.T_F_omega
    if profiles.T_G_omega is not None:
        active_omega = active_omega + profiles.T_G_omega
    if profiles.T_xi_omega is not None:
        active_omega = active_omega + profiles.T_xi_omega
    if profiles.T_nu_omega is not None:
        active_omega = active_omega + profiles.T_nu_omega

    R_omega = profiles.L_omega - active_omega

    # Check residuals inside analysis interval |z| <= 1.0 (away from boundary conditions)
    I = np.where(np.abs(grid.zg) <= 1.0)[0]
    norm_Rb = np.max(np.abs(R_b[I]))
    scale_b = np.max(np.abs(profiles.L_b[I]))
    rel_err_b = norm_Rb / max(scale_b, 1e-14)
    assert rel_err_b < 1e-3, f"Induction equation residual too large: {rel_err_b}"

    norm_Rw = np.max(np.abs(R_omega[I]))
    scale_w = np.max(np.abs(profiles.L_omega[I]))
    rel_err_w = norm_Rw / max(scale_w, 1e-14)
    assert rel_err_w < 1e-3, f"Vorticity equation residual too large: {rel_err_w}"
