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
