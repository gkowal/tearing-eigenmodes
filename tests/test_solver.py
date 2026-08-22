import pytest
import numpy as np
from tearing_eigenmodes import eigenmodes, SimulationParams

def test_eigenmodes_static_vs_dynamic_C():
    # 1. Run standard (static C) convergence
    params_static = SimulationParams(
        alpha=0.15,
        a=1.0,
        w=0.0,
        S=1e3,
        Nmin=64,
        Nmax=512,
        Ninc=32,
        dynamic_C=False,
    )
    σ_stat, _, _, _, _, _, C_stat, N_stat, _, success_stat = eigenmodes(params_static)
    assert success_stat
    assert C_stat is not None and C_stat > 0.0
    assert N_stat is not None and N_stat >= 64

    # 2. Run dynamic C convergence
    params_dynamic = SimulationParams(
        alpha=0.15,
        a=1.0,
        w=0.0,
        S=1e3,
        Nmin=64,
        Nmax=512,
        Ninc=32,
        dynamic_C=True,
    )
    σ_dyn, _, _, _, _, _, C_dyn, N_dyn, _, success_dyn = eigenmodes(params_dynamic)
    assert success_dyn
    assert C_dyn is not None and C_dyn > 0.0
    assert N_dyn is not None and N_dyn >= 64

    # C should have updated and grown in the dynamic case
    assert C_dyn is not None and C_stat is not None and C_dyn > C_stat
