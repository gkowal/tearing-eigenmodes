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
        params = dict(base_params)
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

def test_select_nc_invalid_inputs():
    # alpha <= 0
    with pytest.raises(ValueError, match="Wavenumber alpha must be positive"):
        select_NC({'alpha': 0.0})
    
    # decay_efolds <= 0
    with pytest.raises(ValueError, match="decay_efolds must be positive"):
        select_NC({'alpha': 0.1, 'decay_efolds': 0.0})

    # a <= 0
    with pytest.raises(ValueError, match="Current sheet thickness a must be positive"):
        select_NC({'alpha': 0.1, 'a': 0.0})

    # w < 0
    with pytest.raises(ValueError, match="Current sheet half-width w must be non-negative"):
        select_NC({'alpha': 0.1, 'w': -0.1})

    # Nmin <= 0
    with pytest.raises(ValueError, match="Resolution limits Nmin and Nmax must be positive"):
        select_NC({'alpha': 0.1, 'Nmin': 0})

    # Ninc <= 0
    with pytest.raises(ValueError, match="Resolution increment Ninc must be positive"):
        select_NC({'alpha': 0.1, 'Ninc': 0})

def test_select_nc_cgl_coefficient_guards():
    # C <= 0
    # C = 1.0 + R - Δβ/2. Set Δβ = 3.0 => C = -0.5
    with pytest.raises(DeltaError, match="Decaying factor purely imaginary"):
        select_NC({
            'alpha': 0.1,
            'a': 1.0,
            'w': 0.0,
            'CGL': True,
            'plasma_beta_difference': 3.0,
        })

    # D <= 0
    # D = 1.0 + R + 0.5 * ((ɣpar + ɣper - 2) * β + ɣpar * Δβ)
    # Let β = 1.0, Δβ = -2.0, ɣpar = 3.0, ɣper = 2.0.
    # D = 1.0 + 0.5 * (3 * 1.0 + 3 * (-2.0)) = 1.0 + 0.5 * (-3.0) = -0.5
    with pytest.raises(DeltaError, match="Decaying factor purely imaginary"):
        select_NC({
            'alpha': 0.1,
            'a': 1.0,
            'w': 0.0,
            'CGL': True,
            'plasma_beta': 1.0,
            'plasma_beta_difference': -2.0,
            'parallel_index': 3.0,
            'perpendicular_index': 2.0,
        })

def test_select_nc_stable_sigma_guarded():
    # If σ.real <= 0, the lσ calculation should be bypassed and not crash
    params = {
        'alpha': 0.1,
        'a': 1.0,
        'w': 0.0,
        'xi': 1.0,
        'sigma': complex(-0.05, 0.0), # σ.real <= 0
    }
    # This should complete successfully because the stable σ.real is bypassed
    N, C = select_NC(params)
    assert N >= 64
    assert C > 0.0

