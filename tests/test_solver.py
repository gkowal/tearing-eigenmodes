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


def test_eigenmodes_multiscale_diagnostics_populated():
    params = SimulationParams(
        alpha=0.15,
        a=1.0,
        w=0.0,
        S=1e3,
        Pr=0.0,
        Nmin=64,
        Nmax=512,
        Ninc=32,
        dynamic_C=False,
    )
    sigma, s, e, delta_in, nin, nwa, C, N, z, success = eigenmodes(params)
    assert success
    assert delta_in is not None and np.isfinite(delta_in) and delta_in > 0
    assert nin is not None and nin > 0
    assert nwa is not None and nwa > 0


def test_eigenmodes_viscous_scale_limiting():
    # Run with high Pr where viscous layer is smaller than resistive layer
    params = SimulationParams(
        alpha=0.15,
        a=1.0,
        w=0.0,
        S=1e3,
        Pr=5.0,
        Nmin=64,
        Nmax=512,
        Ninc=32,
        dynamic_C=False,
    )
    sigma, s, e, delta_in, nin, nwa, C, N, z, success = eigenmodes(params)
    assert success
    assert delta_in is not None and np.isfinite(delta_in) and delta_in > 0
    assert nin is not None and nin > 0


def test_eigenmodes_verbose_scale_logging(caplog):
    import logging
    params = SimulationParams(
        alpha=0.15,
        a=1.0,
        w=0.0,
        S=1e3,
        Pr=0.0,
        Nmin=64,
        Nmax=512,
        Ninc=32,
        verbose=True,
        dynamic_C=False,
    )
    with caplog.at_level(logging.INFO):
        sigma, s, e, delta_in, nin, nwa, C, N, z, success = eigenmodes(params)
    assert success
    # Check that "mode scales:" appears in logs with deterministic keys
    scale_logs = [rec.message for rec in caplog.records if "mode scales:" in rec.message]
    assert len(scale_logs) >= 1
    msg = scale_logs[0]
    assert "classical.bz_induction.eta_vs_ideal=" in msg or "classical.bz_induction.eta_vs_f=" in msg
    assert "nan" not in msg
