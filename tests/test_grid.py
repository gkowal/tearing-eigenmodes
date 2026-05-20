import pytest
import numpy as np
from tearing_eigenmodes.grid import select_NC
from tearing_eigenmodes.exceptions import DeltaError, ConvergenceError

def test_select_nc_default():
    # Simple default run
    params = {
        'alpha': 0.1,
        'a': 1.0,
        'w': 0.0,
        'CGL': False,
    }
    N, C = select_NC(params)
    assert isinstance(N, int)
    assert isinstance(C, float)
    assert N >= 64
    assert C > 0.0

def test_select_nc_cmean_strategies():
    base_params = {
        'alpha': 0.1,
        'a': 1.0,
        'w': 0.0,
        'CGL': False,
    }
    means = ['geometric', 'harmonic', 'average', 'lower', 'outer', 'upper', 'inner', 'unknown_mean']
    results = {}
    for m in means:
        params = base_params.copy()
        params['Cmean'] = m
        N, C = select_NC(params)
        results[m] = (N, C)

    # Check that all scaling strategies return float scaling factors and int resolutions
    for m in means:
        assert isinstance(results[m][0], int)
        assert isinstance(results[m][1], float)

def test_select_nc_cgl_success():
    params = {
        'alpha': 0.1,
        'a': 1.0,
        'w': 0.0,
        'CGL': True,
        'plasma_beta': 1.0,
        'plasma_beta_difference': 0.1,
        'parallel_index': 3.0,
        'perpendicular_index': 2.0,
    }
    N, C = select_NC(params)
    assert isinstance(N, int)
    assert isinstance(C, float)

def test_select_nc_cgl_infinite_decay():
    # Trigger D = 0
    # D = 1.0 + R + 0.5 * ((ɣpar + ɣper - 2) * β + ɣpar * Δβ)
    # Let R = 0.0 (by setting σ to None)
    # We want (ɣpar + ɣper - 2) * β + ɣpar * Δβ = -2
    # Let ɣpar = 3, ɣper = 2. Then (ɣpar + ɣper - 2) = 3.
    # If β = 0, then 3 * Δβ = -2 => Δβ = -2/3.
    params = {
        'alpha': 0.1,
        'a': 1.0,
        'w': 0.0,
        'CGL': True,
        'plasma_beta': 0.0,
        'plasma_beta_difference': -2.0 / 3.0,
        'parallel_index': 3.0,
        'perpendicular_index': 2.0,
        'sigma': None,
    }
    with pytest.raises(DeltaError, match="Infinite decaying factor"):
        select_NC(params)

def test_select_nc_cgl_imaginary_decay():
    # Trigger μ < 0 => C / D < 0
    # C = 1.0 + R - Δβ/2
    # Let σ = None => R = 0 => C = 1.0 - Δβ/2
    # Let Δβ = 3.0 => C = 1.0 - 1.5 = -0.5
    # D = 1.0 + 0.5 * (3 * β + 3 * 3) = 1.0 + 1.5 * β + 4.5
    # If β = 0, D = 5.5 > 0.
    # Since C = -0.5 < 0 and D = 5.5 > 0, μ = C / D = -0.5 / 5.5 < 0.
    params = {
        'alpha': 0.1,
        'a': 1.0,
        'w': 0.0,
        'CGL': True,
        'plasma_beta': 0.0,
        'plasma_beta_difference': 3.0,
        'parallel_index': 3.0,
        'perpendicular_index': 2.0,
        'sigma': None,
    }
    with pytest.raises(DeltaError, match="Decaying factor purely imaginary"):
        select_NC(params)

def test_select_nc_convergence_error():
    # Trigger ConvergenceError by forcing N > Ntop immediately
    params = {
        'alpha': 0.1,
        'a': 1.0,
        'w': 0.0,
        'Nmin': 64,
        'Nmax': 64,
        'Ninc': 1,
    }
    with pytest.raises(ConvergenceError, match="Insufficient N up to Nmax"):
        select_NC(params)
