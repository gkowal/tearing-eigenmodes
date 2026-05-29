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
    alpha, Delta = estimate_max(params)
    assert alpha > 0.0
    assert isinstance(alpha, float)
    assert isinstance(Delta, float)

def test_estimate_max_invalid_inputs():
    # Lundquist number S <= 0
    with pytest.raises(ValueError, match="Lundquist number S must be positive"):
        estimate_max(SimulationParams(S=0.0))

    # Prandtl number Pr < 0
    with pytest.raises(ValueError, match="Prandtl number Pr cannot be negative"):
        estimate_max(SimulationParams(Pr=-0.1))

    # Parallel index <= 0
    with pytest.raises(ValueError, match="Parallel adiabatic index.*must be positive"):
        estimate_max(SimulationParams(parallel_index=0.0))

    # Perpendicular index <= 0
    with pytest.raises(ValueError, match="Perpendicular adiabatic index.*must be positive"):
        estimate_max(SimulationParams(perpendicular_index=0.0))

def test_estimate_max_anisotropy_guards():
    # A <= 0 (requires Δβ < 2)
    # Let Δβ = 2.0 => A = 0.0 <= 0
    with pytest.raises(DeltaError, match="Stable or unphysical regime: coefficient A.*<= 0"):
        estimate_max(SimulationParams(
            CGL=False,
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
