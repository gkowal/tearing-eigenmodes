import pytest
import numpy as np
from psecas import ChebyshevRationalGrid
from tearing_eigenmodes.analysis import inner_layer_thickness, find_peak_location

class MockSystem:
    def __init__(self, grid, a=1.0, w=0.0, S=1.0, kx=1.0, Bx=1.0, Ux=0.0, shear=False, xi=0.0):
        self.grid = grid
        self.a = a
        self.w = w
        self.S = S
        self.kx = kx
        self.Bx = Bx
        self.Ux = Ux
        self.shear = shear
        self.ξ = xi
        self.result = {}

def test_inner_layer_thickness_sign_change():
    grid = ChebyshevRationalGrid(N=64, C=1.0, max_derivative_order=2)
    system = MockSystem(grid, a=1.0, w=0.0, S=1.0, kx=1.0, Bx=1.0)
    
    # We want Td = Tn - Ti to change sign at z = 0.5
    # Tn = |b" - k^2 b| / S. If b = const 1.0, Tn = |0 - 1.0|/1.0 = 1.0
    # Ti = |1j * kx * a * u * Bx| = |1.0 * u|.
    # If we set u = 2.0 * grid.zg, then Ti = 2.0 * z.
    # Td = 1.0 - 2.0 * z.
    # At z = 0.5, Td = 0.0.
    system.result['dbz'] = np.ones_like(grid.zg)
    system.result['duz'] = 2.0 * grid.zg
    
    delta, nin, nwa = inner_layer_thickness(system, δtol=1e-5)
    
    # delta should be very close to 0.5
    assert np.isclose(delta, 0.5, atol=2e-2)
    assert isinstance(nin, int)
    assert isinstance(nwa, int)
    assert nin > 0
    assert nwa > 0

def test_inner_layer_thickness_no_sign_change():
    grid = ChebyshevRationalGrid(N=64, C=1.0, max_derivative_order=2)
    system = MockSystem(grid, a=1.0, w=0.0, S=1.0, kx=1.0, Bx=1.0)
    
    # Set fields such that Td = Tn - Ti is always negative
    # Tn = 1.0, Ti = 10.0
    system.result['dbz'] = np.ones_like(grid.zg)
    system.result['duz'] = 10.0 * np.ones_like(grid.zg)
    
    delta, nin, nwa = inner_layer_thickness(system)
    
    # delta should be 0.0 since there is no region where Td >= 0
    assert delta == 0.0
    assert nin == 1  # max(1, I[0].size) where I is zg <= 0.0 -> zg=0.0 is 1 point

def test_find_peak_location():
    grid = ChebyshevRationalGrid(N=64, C=1.0, max_derivative_order=2)
    
    # Create a clean Gaussian peak at z = 0.5 for u0
    # u0 = exp(-((z - 0.5) / 0.1)^2)
    u0 = np.exp(-((grid.zg - 0.5) / 0.1)**2)
    b0 = np.zeros_like(grid.zg)
    
    z_peak, n = find_peak_location(u0, b0, grid, a=1.0, w=0.0, ztol=1e-4)
    
    # The peak should be extremely close to 0.5
    assert np.isclose(z_peak, 0.5, atol=1e-3)
    assert isinstance(n, int)
    assert n > 0
