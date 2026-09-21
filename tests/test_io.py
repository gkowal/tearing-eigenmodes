from tearing_eigenmodes.io import build_dpath
from tearing_eigenmodes.params import SimulationParams


def test_model_tag_distinguishes_classical_and_cgl():
    classical = SimulationParams(
        CGL=False,
        plasma_beta=None,
        plasma_beta_difference=None,
        zeta=None,
    )
    cgl = SimulationParams(
        CGL=True,
        plasma_beta=None,
        plasma_beta_difference=None,
        eos=None,
        parallel_index=None,
        perpendicular_index=None,
    )
    classical_dpath = build_dpath(classical)
    cgl_dpath = build_dpath(cgl)
    assert classical_dpath != cgl_dpath
    assert "MHD" in classical_dpath
    assert "CGL" in cgl_dpath


def test_cgl_eos_distinguishes_dirs():
    adiabatic = SimulationParams(CGL=True, eos="adiabatic")
    isothermal = SimulationParams(CGL=True, eos="isothermal")
    assert build_dpath(adiabatic) != build_dpath(isothermal)
    assert "adiabatic" in build_dpath(adiabatic)
    assert "isothermal" in build_dpath(isothermal)


def test_cgl_gamma_indices_distinguish_dirs():
    default = SimulationParams(
        CGL=True, parallel_index=3.0, perpendicular_index=2.0
    )
    custom = SimulationParams(
        CGL=True, parallel_index=1.5, perpendicular_index=1.0
    )
    parallel_only = SimulationParams(
        CGL=True, parallel_index=1.5, perpendicular_index=2.0
    )
    assert build_dpath(default) != build_dpath(custom)
    assert build_dpath(default) != build_dpath(parallel_only)


def test_noshear_distinguishes_dirs():
    shear = SimulationParams(CGL=False, noshear=False)
    noshear = SimulationParams(CGL=False, noshear=True)
    shear_dpath = build_dpath(shear)
    noshear_dpath = build_dpath(noshear)
    assert shear_dpath != noshear_dpath
    assert "noshear" not in shear_dpath
    assert "noshear" in noshear_dpath


def test_zeta_distinguishes_dirs():
    low = SimulationParams(CGL=False, zeta=1.0)
    high = SimulationParams(CGL=False, zeta=2.0)
    assert build_dpath(low) != build_dpath(high)
    assert f"ζ{1.0:.3e}" in build_dpath(low)
    assert f"ζ{2.0:.3e}" in build_dpath(high)


def test_swept_none_params_omitted():
    swept = SimulationParams(CGL=False, S=None, zeta=None)
    swept_dpath = build_dpath(swept)
    assert f"S{1e4:.3e}" not in swept_dpath
    assert "ζ" not in swept_dpath

    swept_cgl = SimulationParams(
        CGL=True, eos=None, parallel_index=None, perpendicular_index=None
    )
    swept_cgl_dpath = build_dpath(swept_cgl)
    assert "ɣpar" not in swept_cgl_dpath
    assert "ɣper" not in swept_cgl_dpath
    assert "adiabatic" not in swept_cgl_dpath


def test_suffix_preserved():
    classical = SimulationParams(CGL=False, suffix="_test123")
    cgl = SimulationParams(CGL=True, suffix="_test123")
    assert build_dpath(classical).endswith("_test123")
    assert build_dpath(cgl).endswith("_test123")


def test_delta_beta_distinguishes_dirs_scientific_notation():
    low = SimulationParams(CGL=True, plasma_beta_difference=0.001)
    high = SimulationParams(CGL=True, plasma_beta_difference=0.004)
    low_dpath = build_dpath(low)
    high_dpath = build_dpath(high)
    assert low_dpath != high_dpath
    assert f"Δβ{0.001:+.3e}" in low_dpath
    assert f"Δβ{0.004:+.3e}" in high_dpath

    lower = SimulationParams(CGL=True, plasma_beta_difference=0.005)
    upper = SimulationParams(CGL=True, plasma_beta_difference=0.014)
    lower_dpath = build_dpath(lower)
    upper_dpath = build_dpath(upper)
    assert lower_dpath != upper_dpath
    assert f"Δβ{0.005:+.3e}" in lower_dpath
    assert f"Δβ{0.014:+.3e}" in upper_dpath


def test_default_classical_dir_contains_core_segments():
    dpath = build_dpath(SimulationParams())
    assert f"S{1e4:.3e}" in dpath
    assert f"Pr{0.0:.3e}" in dpath
    assert f"a{1.0:.3e}" in dpath
    assert f"w{0.0:.3e}" in dpath
