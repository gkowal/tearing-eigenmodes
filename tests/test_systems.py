import numpy as np
import pytest
from psecas import ChebyshevRationalGrid
from tearing_eigenmodes.systems import TearingClassicalMHD, TearingGyrotropicMHD

def test_classical_mhd_init():
    grid = ChebyshevRationalGrid(N=32, C=1.0, max_derivative_order=4)
    # Test classical initialization with periodic=False (standard boundary conditions)
    system = TearingClassicalMHD(grid, kx=0.1, S=1e4, periodic=False)
    assert system.kx == 0.1
    assert system.periodic is False
    assert system.dim == 2
    assert "duz" in system.variables
    assert "dbz" in system.variables

def test_gyrotropic_mhd_init():
    grid = ChebyshevRationalGrid(N=32, C=1.0, max_derivative_order=4)
    # Test gyrotropic initialization
    system = TearingGyrotropicMHD(grid, kx=0.1, S=1e4, periodic=False)
    assert system.kx == 0.1
    assert system.periodic is False
    # By default, ɣpar=3, ɣper=2, which is not isothermal, so we expect 5 variables
    assert system.dim == 5
    assert "duz" in system.variables
    assert "dbz" in system.variables
    assert "duy" in system.variables
    assert "dby" in system.variables
    assert "ddp" in system.variables

def test_gyrotropic_periodic_matches_classical():
    # Gyrotropic periodic background must equal the Classical periodic
    # background with w=0, zeta=0, Bguide=0 (Gyrotropic has no such params).
    a, z1, z2, kx = 0.5, -1.0, 1.0, 0.3
    grid_g = ChebyshevRationalGrid(N=32, C=1.0, max_derivative_order=4)
    grid_c = ChebyshevRationalGrid(N=32, C=1.0, max_derivative_order=4)
    gyro = TearingGyrotropicMHD(grid_g, kx=kx, S=1e4, a=a, z1=z1, z2=z2,
                                periodic=True)
    clas = TearingClassicalMHD(grid_c, kx=kx, S=1e4, a=a, z1=z1, z2=z2,
                               w=0, ζ=0, Bguide=0, periodic=True)
    np.testing.assert_allclose(gyro.Bx, clas.Bx, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(gyro.By, clas.By, rtol=1e-12, atol=1e-14)

def test_gyrotropic_beta_mutation_refreshes_cgl_decay():
    grid = ChebyshevRationalGrid(N=32, C=1.0, max_derivative_order=4)
    kwargs = dict(kx=0.3, S=1e4, a=0.5, periodic=False, β=0.5, Δβ=0.2,
                  ɣpar=3.0, ɣper=2.0, σ=0.1)
    system = TearingGyrotropicMHD(grid, **kwargs)
    system.β = 1.5
    system.Δβ = 0.7
    grid_ref = ChebyshevRationalGrid(N=32, C=1.0, max_derivative_order=4)
    ref = TearingGyrotropicMHD(grid_ref, **{**kwargs, "β": 1.5, "Δβ": 0.7})
    np.testing.assert_allclose(system.A, ref.A, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(system.R0, ref.R0, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(system.λ, ref.λ, rtol=1e-14, atol=1e-15)
    # Rejected β update must leave everything unchanged
    before = (system.A, system.R0, system.λ, system.β0, system.Γβ)
    system.β = -1.0
    after = (system.A, system.R0, system.λ, system.β0, system.Γβ)
    assert system.β == 1.5
    for b, a in zip(before, after):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)

def test_classical_hall_kx_zero_ky_zero_raises():
    grid = ChebyshevRationalGrid(N=32, C=1.0, max_derivative_order=4)
    with pytest.raises(ValueError, match="> 0"):
        TearingClassicalMHD(grid, kx=0.0, ky=0.0, S=1e4, ϵ=0.1)

def test_classical_nonhall_kx_zero_constructs():
    grid = ChebyshevRationalGrid(N=32, C=1.0, max_derivative_order=4)
    system = TearingClassicalMHD(grid, kx=0.0, ky=0.0, S=1e4)
    assert system.kx == 0.0
    assert system.dim == 2

def test_classical_hall_kx_positive_constructs():
    grid = ChebyshevRationalGrid(N=32, C=1.0, max_derivative_order=4)
    system = TearingClassicalMHD(grid, kx=0.1, ky=0.0, S=1e4, ϵ=0.1)
    assert system.kx == 0.1
    assert system.dim == 4

def test_gyrotropic_kx_zero_raises():
    grid = ChebyshevRationalGrid(N=32, C=1.0, max_derivative_order=4)
    with pytest.raises(ValueError, match="kx must be > 0"):
        TearingGyrotropicMHD(grid, kx=0.0, S=1e4)

def _hall_bz_equation(**kwargs):
    grid = ChebyshevRationalGrid(N=32, C=1.0, max_derivative_order=4)
    system = TearingClassicalMHD(grid, kx=0.5, S=1e4, ϵ=0.1, periodic=False, **kwargs)
    return system.equations[system.variables.index("dbz")]

def test_classical_hall_bz_terms_appear_once():
    eq = _hall_bz_equation(w=0.5, ξ=0.1, shear=True)
    assert eq.count("-1j*kx*Ux*dbz") == 1
    assert eq.count("+ξ*dz(duz)") == 1
    assert eq.count("ξ*kx*dz(dby)") + eq.count("ξ*kx*ϵ*dz(dby)") == 1

def test_classical_hall_no_shear_drops_advection():
    eq = _hall_bz_equation(w=0.5, ξ=0.0, shear=False)
    assert "Ux" not in eq
    assert "ξ" not in eq

@pytest.mark.parametrize("zeta", [-0.1, 1.5])
def test_classical_zeta_out_of_range_raises(zeta):
    grid = ChebyshevRationalGrid(N=32, C=1.0, max_derivative_order=4)
    with pytest.raises(ValueError, match="between 0 and 1"):
        TearingClassicalMHD(grid, kx=0.1, S=1e4, ζ=zeta)

def test_classical_a_setter_refreshes_boundary_decay():
    grid = ChebyshevRationalGrid(N=32, C=1.0, max_derivative_order=4)
    system = TearingClassicalMHD(grid, kx=0.3, S=1e4, a=1.0, periodic=False)
    system.a = 0.5
    assert system.λ == pytest.approx(0.15)
    assert system.δ == 0.5

def test_classical_w_setter_rejects_equation_change():
    grid = ChebyshevRationalGrid(N=32, C=1.0, max_derivative_order=4)
    system = TearingClassicalMHD(grid, kx=0.3, S=1e4, w=0.0, periodic=False)
    with pytest.raises(ValueError, match="construct a new system"):
        system.w = 0.5
    assert system.w == 0.0
    sheared = TearingClassicalMHD(ChebyshevRationalGrid(N=32, C=1.0, max_derivative_order=4),
                                  kx=0.3, S=1e4, w=0.5, periodic=False)
    sheared.w = 0.7
    assert sheared.w == 0.7

def test_gyrotropic_a_setter_refreshes_cgl_decay():
    kwargs = dict(kx=0.3, S=1e4, periodic=False, β=0.5, Δβ=0.2, σ=0.1)
    system = TearingGyrotropicMHD(ChebyshevRationalGrid(N=32, C=1.0, max_derivative_order=4), a=1.0, **kwargs)
    system.a = 0.5
    ref = TearingGyrotropicMHD(ChebyshevRationalGrid(N=32, C=1.0, max_derivative_order=4), a=0.5, **kwargs)
    np.testing.assert_allclose(system.χ, ref.χ, rtol=1e-14)
    np.testing.assert_allclose(system.λ, ref.λ, rtol=1e-14)

def test_gyrotropic_delta_beta_setter_rejects_equation_change():
    system = TearingGyrotropicMHD(ChebyshevRationalGrid(N=32, C=1.0, max_derivative_order=4),
                                  kx=0.3, S=1e4, periodic=False, Δβ=0.0)
    with pytest.raises(ValueError, match="construct a new system"):
        system.Δβ = 0.2
    assert system.Δβ == 0.0
