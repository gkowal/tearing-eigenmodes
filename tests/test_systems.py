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
