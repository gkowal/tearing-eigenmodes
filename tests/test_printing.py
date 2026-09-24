import logging

import numpy as np

from tearing_eigenmodes.params import SimulationParams
from tearing_eigenmodes.printing import print_info


def test_print_info_classical_hides_cgl_fields(caplog):
    with caplog.at_level(logging.INFO, logger="tearing_eigenmodes.printing"):
        print_info(SimulationParams(CGL=False))
    assert "Equation of State" not in caplog.text
    assert "Plasma-β (β)" not in caplog.text
    assert "Plasma-β difference (Δβ)" not in caplog.text
    assert "Parallel adiabatic index" not in caplog.text
    assert "Perpendicular adiabatic index" not in caplog.text
    assert "Anisotropy scale points" not in caplog.text
    assert "Equations" in caplog.text
    assert "Classical MHD" in caplog.text
    assert "Lundquist number" in caplog.text
    assert "Magnetic transverse field (ξ)" in caplog.text
    assert "Current sheet half-width (w)" in caplog.text
    assert "Shear parameter (ζ)" in caplog.text
    assert "Velocity shear" in caplog.text


def test_print_info_cgl_shows_cgl_fields(caplog):
    with caplog.at_level(logging.INFO, logger="tearing_eigenmodes.printing"):
        print_info(SimulationParams(CGL=True))
    assert "Equation of State" in caplog.text
    assert "Plasma-β (β)" in caplog.text
    assert "Plasma-β difference (Δβ)" in caplog.text
    assert "Parallel adiabatic index" in caplog.text
    assert "Perpendicular adiabatic index" in caplog.text
    assert "Anisotropy scale points" in caplog.text
    assert "Gyrotropic MHD" in caplog.text
    assert "Magnetic transverse field (ξ)" not in caplog.text
    assert "Current sheet half-width (w)" not in caplog.text
    assert "Shear parameter (ζ)" not in caplog.text
    assert "Velocity shear" not in caplog.text


def _fake_eigenmodes(*args, **kwargs):
    n = 1
    v = np.zeros(n)
    alpha = np.array([0.2])
    sigma = np.array([0.04 + 0.0j])
    e = np.array([1e-5])
    delta = np.array([0.09])
    c = np.array([1.2])
    nin = np.array([22])
    nwa = np.array([50])
    N = np.array([128])
    return v, alpha, sigma, e, delta, c, nin, nwa, N


def _read_dat_header(tmp_path, monkeypatch, cgl):
    import tearing_eigenmodes.io as io_mod

    monkeypatch.setattr(io_mod, "load_eigenmodes", _fake_eigenmodes)
    params = SimulationParams(CGL=cgl, data_path=str(tmp_path), mode=0)
    io_mod.write_results(params, delta_time=1.0)
    with open(f"{tmp_path}.dat") as f:
        return f.read()


def test_write_results_header_classical(tmp_path, monkeypatch):
    content = _read_dat_header(tmp_path, monkeypatch, cgl=False)
    assert "Magnetic transverse field" in content
    assert "half-width" in content
    assert "Shear parameter" in content
    assert "Equation of State" not in content
    assert "Plasma-β (β)" not in content
    assert "Δβ" not in content
    assert "adiabatic index" not in content
    assert "Anisotropy scale points" not in content


def test_write_results_header_cgl(tmp_path, monkeypatch):
    content = _read_dat_header(tmp_path, monkeypatch, cgl=True)
    assert "Equation of State" in content
    assert "Plasma-β (β)" in content
    assert "Δβ" in content
    assert "adiabatic index" in content
    assert "Anisotropy scale points" in content
    assert "Magnetic transverse field" not in content
    assert "half-width" not in content
    assert "Shear parameter" not in content


def test_write_results_header_explains_unconverged_rows(tmp_path, monkeypatch):
    content = _read_dat_header(tmp_path, monkeypatch, cgl=False)
    assert "tolerance > 1 did not converge" in content
    rows = [l for l in content.splitlines() if l and not l.startswith("#")]
    assert len(rows) == 1 and len(rows[0].split()) == 9
