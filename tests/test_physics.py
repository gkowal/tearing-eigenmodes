import pytest
from tearing_eigenmodes import SimulationParams, estimate_max
from tearing_eigenmodes.exceptions import DeltaError

def test_estimate_max_default():
    params = SimulationParams(
        CGL=False,
        S=1e4,
        Pr=0.0,
        plasma_beta=0.0,
        plasma_beta_difference=0.0,
        parallel_index=3.0,
        perpendicular_index=2.0,
    )
    alpha = estimate_max(params)
    assert alpha > 0.0
    assert isinstance(alpha, float)

def test_estimate_max_invalid_inputs():
    # Lundquist number S <= 0
    with pytest.raises(ValueError, match="Lundquist number S must be positive"):
        estimate_max(SimulationParams(S=0.0))

    # Prandtl number Pr < 0
    with pytest.raises(ValueError, match="Prandtl number Pr cannot be negative"):
        estimate_max(SimulationParams(Pr=-0.1))

    # Parallel index <= 0
    with pytest.raises(ValueError, match="Parallel adiabatic index.*must be positive"):
        estimate_max(SimulationParams(CGL=True, parallel_index=0.0))

    # Perpendicular index <= 0
    with pytest.raises(ValueError, match="Perpendicular adiabatic index.*must be positive"):
        estimate_max(SimulationParams(CGL=True, perpendicular_index=0.0))

def test_estimate_max_anisotropy_guards():
    # A <= 0 (requires Δβ < 2)
    # Let Δβ = 2.0 => A = 0.0 <= 0
    with pytest.raises(DeltaError, match="Stable or unphysical regime: coefficient A.*<= 0"):
        estimate_max(SimulationParams(
            CGL=True,
            plasma_beta_difference=2.0,
        ))

    # Under CGL, stable bounds (Δβ <= -(2 + (ɣpar + ɣper - 2) * β) / ɣpar)
    # Let β = 0.0, ɣpar = 3.0, ɣper = 2.0. Limit is -(2 + 0)/3 = -2/3.
    # Set Δβ = -1.0 => Δβ < -2/3
    with pytest.raises(DeltaError, match="purely imaginary or stable: decay coefficient"):
        estimate_max(SimulationParams(
            CGL=True,
            plasma_beta=0.0,
            plasma_beta_difference=-1.0,
            parallel_index=3.0,
            perpendicular_index=2.0,
        ))


def test_calculate_inner_factors_classical():
    from tearing_eigenmodes import calculate_inner_factors
    A, R0, q = calculate_inner_factors(beta=1.0, delta_beta=0.5, CGL=False)
    assert A == 1.0
    assert R0 == 1.0
    assert q == 1.0


def test_calculate_inner_factors_gyrotropic():
    import numpy as np
    from tearing_eigenmodes import calculate_inner_factors

    # Classical limit under CGL (zero beta and delta_beta)
    A, R0, q = calculate_inner_factors(beta=0.0, delta_beta=0.0, CGL=True)
    assert A == 1.0
    assert R0 == 1.0
    assert q == 1.0

    # Non-trivial Gyrotropic parameters
    # A = 1 - 0.4 / 2 = 0.8
    # R0 = 1 + 0.5 * ((3 + 2 - 2) * 1.0 + 3 * 0.4) = 1 + 0.5 * (3.0 + 1.2) = 3.1
    # q = sqrt(0.8 / 3.1)
    A, R0, q = calculate_inner_factors(
        beta=1.0,
        delta_beta=0.4,
        gamma_par=3.0,
        gamma_per=2.0,
        CGL=True,
    )
    assert np.isclose(A, 0.8)
    assert np.isclose(R0, 3.1)
    assert np.isclose(q, np.sqrt(0.8 / 3.1))

    # A <= 0 guard (delta_beta >= 2.0)
    with pytest.raises(DeltaError, match="Gyrotropic inner factors require A > 0"):
        calculate_inner_factors(delta_beta=2.0, CGL=True)

    # R0 <= 0 guard
    # R0 = 1 + 0.5 * (3 * 0.0 + 3 * (-1.0)) = 1 - 1.5 = -0.5
    with pytest.raises(DeltaError, match="Gyrotropic inner factors require A > 0 and R0 > 0"):
        calculate_inner_factors(beta=0.0, delta_beta=-1.0, gamma_par=3.0, gamma_per=2.0, CGL=True)


def test_model_delta_prime():
    import numpy as np
    from tearing_eigenmodes import model_delta_prime

    # Classical limit (q = 1)
    # Delta' = 2 * (1 / 0.2 - 0.2) = 2 * (5.0 - 0.2) = 9.6
    dp = model_delta_prime(0.2, q=1.0)
    assert np.isclose(dp, 9.6)

    # Marginal cutoff at alpha = q => Delta' = 0
    assert np.isclose(model_delta_prime(1.5, q=1.5), 0.0)

    # Beyond marginal cutoff alpha > q => Delta' < 0
    assert model_delta_prime(2.0, q=1.0) < 0.0

    # Invalid input guards
    with pytest.raises(ValueError, match="Wavenumber alpha must be strictly positive"):
        model_delta_prime(0.0)

    with pytest.raises(ValueError, match="Parameter q must be strictly positive"):
        model_delta_prime(0.1, q=0.0)


def test_estimate_inner_scale_classical_and_gyrotropic():
    import numpy as np
    from tearing_eigenmodes import estimate_inner_scale

    params_classical = SimulationParams(
        alpha=0.1,
        a=1.0,
        S=1e4,
        Pr=0.0,
        CGL=False,
    )
    scale_classical = estimate_inner_scale(params_classical)
    assert scale_classical > 0.0
    assert isinstance(scale_classical, float)

    # Classical limit under Gyrotropic model gives identical scale
    params_gyrotropic_limit = SimulationParams(
        alpha=0.1,
        a=1.0,
        S=1e4,
        Pr=0.0,
        CGL=True,
        plasma_beta=0.0,
        plasma_beta_difference=0.0,
        parallel_index=3.0,
        perpendicular_index=2.0,
    )
    scale_gyro_lim = estimate_inner_scale(params_gyrotropic_limit)
    assert np.isclose(scale_classical, scale_gyro_lim)

    # Higher S reduces inner scale
    params_high_S = SimulationParams(alpha=0.1, a=1.0, S=1e6, Pr=0.0, CGL=False)
    scale_high_S = estimate_inner_scale(params_high_S)
    assert scale_high_S < scale_classical

    # Safety factor scales grid scale inversely
    params_safe = SimulationParams(alpha=0.1, a=1.0, S=1e4, Pr=0.0, CGL=False)
    setattr(params_safe, 'inner_resolution_safety', 2.0)
    scale_safe = estimate_inner_scale(params_safe)
    assert np.isclose(scale_safe, scale_classical / 2.0)

    # Dimensional scaling with a
    params_a2 = SimulationParams(alpha=0.1, a=2.0, S=1e4, Pr=0.0, CGL=False)
    scale_a2 = estimate_inner_scale(params_a2)
    assert np.isclose(scale_a2, 2.0 * scale_classical)

    # Beyond marginal cutoff alpha > 1.0 (Delta' <= 0) falls back to Coppi branch safely
    scale_cutoff = estimate_inner_scale(params_classical, alpha=1.5)
    assert scale_cutoff > 0.0


def test_calculate_anisotropy_scale_growth_estimate():
    import numpy as np
    from tearing_eigenmodes import calculate_anisotropy_scale, SimulationParams
    from tearing_eigenmodes.physics import estimate_growth_rate

    # Positive beta_bar with NO initial sigma supplied -> uses estimate_growth_rate
    params = SimulationParams(
        CGL=True,
        alpha=0.1,
        a=1.0,
        plasma_beta=1.0,
        plasma_beta_difference=0.1,
        parallel_index=3.0,
        perpendicular_index=2.0,
        S=1e4,
        sigma=None,
    )
    scale = calculate_anisotropy_scale(params)
    assert scale > 0.0

    gamma_hat = estimate_growth_rate(params, alpha=0.1)
    beta_bar = 0.5 * ((3.0 + 2.0 - 2.0) * 1.0 + (3.0 - 1.0) * 0.1)  # 1.6
    expected_scale = gamma_hat / (0.1 * np.sqrt(beta_bar))
    assert np.isclose(scale, expected_scale)


def test_calculate_anisotropy_scale_negative_beta_bar(caplog):
    import logging
    from tearing_eigenmodes import calculate_anisotropy_scale, SimulationParams

    # Negative beta_bar branch: beta = 0.1, delta_beta = -0.5
    # beta_bar = 0.5 * (3 * 0.1 + 2 * (-0.5)) = 0.5 * (0.3 - 1.0) = -0.35 < 0
    params = SimulationParams(
        CGL=True,
        alpha=0.1,
        a=1.0,
        plasma_beta=0.1,
        plasma_beta_difference=-0.5,
        parallel_index=3.0,
        perpendicular_index=2.0,
        S=1e4,
    )
    with caplog.at_level(logging.DEBUG):
        scale = calculate_anisotropy_scale(params)

    assert scale == 0.0
    assert "central delta_q estimate does not cover off-center pressure structures" in caplog.text


def test_modified_case_scale_broadening():
    import numpy as np
    from tearing_eigenmodes import (
        modified_case_scale_broadening,
        estimate_modified_inner_scale,
        SimulationParams,
    )

    base = 0.05
    # w=0, xi=0 returns exact base theoretical scale
    assert np.isclose(modified_case_scale_broadening(base, w=0.0, xi=0.0, a=1.0), base)

    # Monotone non-decreasing in w
    scale_w1 = modified_case_scale_broadening(base, w=0.5, xi=0.0, a=1.0)
    scale_w2 = modified_case_scale_broadening(base, w=1.0, xi=0.0, a=1.0)
    assert scale_w1 > base
    assert scale_w2 > scale_w1

    # Monotone non-decreasing in xi
    scale_xi1 = modified_case_scale_broadening(base, w=0.0, xi=0.5, a=1.0)
    scale_xi2 = modified_case_scale_broadening(base, w=0.0, xi=1.0, a=1.0)
    assert scale_xi1 > base
    assert scale_xi2 > scale_xi1

    # Combined parameters
    scale_comb = modified_case_scale_broadening(base, w=0.5, xi=0.5, a=1.0)
    assert scale_comb > scale_w1
    assert scale_comb > scale_xi1

    # Integrated modified inner scale helper
    params = SimulationParams(alpha=0.1, a=1.0, S=1e4, Pr=0.0, CGL=False, w=0.5, xi=0.2)
    mod_scale = estimate_modified_inner_scale(params)
    assert mod_scale > base

