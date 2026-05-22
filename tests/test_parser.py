import pytest
import sys
import argparse
from tearing_eigenmodes.parser import parser_setup, validate_parameters, build_params
from tearing_eigenmodes.exceptions import ParameterError

def test_validate_parameters_default_valid():
    parser = parser_setup()
    args = parser.parse_args([])
    # Default parameters should validate successfully without raising any exceptions
    validate_parameters(args)

def test_validate_parameters_lundquist():
    parser = parser_setup()
    args = parser.parse_args([])
    args.Lundquist_number = 0.0
    with pytest.raises(ParameterError, match="Lundquist number must be > 0"):
        validate_parameters(args)

    args.Lundquist_number = -5.0
    with pytest.raises(ParameterError, match="Lundquist number must be > 0"):
        validate_parameters(args)

def test_validate_parameters_prandtl():
    parser = parser_setup()
    args = parser.parse_args([])
    args.Prandtl_number = -0.1
    with pytest.raises(ParameterError, match="Prandtl number cannot be negative"):
        validate_parameters(args)

def test_validate_parameters_beta():
    parser = parser_setup()
    args = parser.parse_args([])
    args.plasma_beta = -1.0
    with pytest.raises(ParameterError, match="Plasma‑β must be ≥ 0"):
        validate_parameters(args)

def test_validate_parameters_parallel_beta():
    parser = parser_setup()
    args = parser.parse_args([])
    args.plasma_beta = 1.0
    args.plasma_beta_difference = -2.0
    # β∥ = β⊥ + Δβ = 1.0 - 2.0 = -1.0 < 0
    with pytest.raises(ParameterError, match="Parallel plasma‑β.*must be ≥ 0"):
        validate_parameters(args)

def test_validate_parameters_transverse_field():
    parser = parser_setup()
    args = parser.parse_args([])
    args.magnetic_transverse_field = -0.1
    with pytest.raises(ParameterError, match="Magnetic transverse field strength .* must be ≥ 0"):
        validate_parameters(args)

def test_validate_parameters_hall():
    parser = parser_setup()
    args = parser.parse_args([])
    args.Hall_parameter = -0.1
    with pytest.raises(ParameterError, match="Hall parameter .* must be ≥ 0"):
        validate_parameters(args)

def test_validate_parameters_thickness_width():
    parser = parser_setup()
    args = parser.parse_args([])
    args.thickness = 0.0
    with pytest.raises(ParameterError, match="Current sheet thickness .* must be > 0"):
        validate_parameters(args)

    args.thickness = 1.0
    args.width = -0.1
    with pytest.raises(ParameterError, match="Current sheet half‑width .* must be ≥ 0"):
        validate_parameters(args)

def test_validate_parameters_eos_custom():
    parser = parser_setup()
    args = parser.parse_args([])
    args.eos = "custom"

    # If one or both gamma parameters are missing
    if hasattr(args, "gamma_parallel"):
        delattr(args, "gamma_parallel")
    with pytest.raises(ParameterError, match="Both --gamma-parallel .* must be provided"):
        validate_parameters(args)

def test_validate_parameters_resolution_range():
    parser = parser_setup()
    args = parser.parse_args([])

    # Nmin > Nmax
    args.resolution_range = [128, 64, 32]
    with pytest.raises(ParameterError, match="Resolution range start .* must be ≤ end"):
        validate_parameters(args)

    # Ninc <= 0
    args.resolution_range = [64, 128, 0]
    with pytest.raises(ParameterError, match="Resolution increment .* must be positive"):
        validate_parameters(args)

    # Ninc > Nmin
    args.resolution_range = [64, 128, 128]
    with pytest.raises(ParameterError, match="cannot exceed the minimum resolution"):
        validate_parameters(args)

def test_validate_parameters_tolerances():
    parser = parser_setup()
    args = parser.parse_args([])

    args.absolute_tolerance = 0.0
    with pytest.raises(ParameterError, match="absolute tolerance .* must be > 0"):
        validate_parameters(args)

    args = parser.parse_args([])
    args.relative_tolerance = -1e-5
    with pytest.raises(ParameterError, match="relative tolerance .* must be > 0"):
        validate_parameters(args)

def test_validate_parameters_growth_rate():
    parser = parser_setup()
    args = parser.parse_args([])

    # gmin > gmax
    args.real_part_range = [1.0, 0.5]
    with pytest.raises(ParameterError, match="Growth‑rate lower bound .* must be ≤ upper bound"):
        validate_parameters(args)

    # gmin < 0
    args.real_part_range = [-0.1, 1.0]
    with pytest.raises(ParameterError, match="Growth‑rate lower bound cannot be negative"):
        validate_parameters(args)

def test_validate_parameters_inner_points():
    parser = parser_setup()
    args = parser.parse_args([])

    args.n_inner = 2
    with pytest.raises(ParameterError, match="Minimum number of inner collocation points .* must be >= 3"):
        validate_parameters(args)

    args = parser.parse_args([])
    args.n_resistivity = 2
    with pytest.raises(ParameterError, match="Minimum number of resistivity layer collocation points .* must be >= 3"):
        validate_parameters(args)



def test_validate_parameters_conflicting_scaling():
    parser = parser_setup()
    args = parser.parse_args([])

    args.scaling_factor = 1.0
    args.resistive_scale = 0.1
    with pytest.raises(ParameterError, match="Only one of these options may be set at a time"):
        validate_parameters(args)

    args = parser.parse_args([])
    args.resistive_scale = -0.1
    with pytest.raises(ParameterError, match="Resistive scale .* must be > 0"):
        validate_parameters(args)

def test_build_params_dispersion(monkeypatch):
    # Simulate arguments passed to eigenmodes-compute.py
    monkeypatch.setattr("sys.argv", [
        "eigenmodes-compute.py",
        "-S", "1.5e4",
        "-Pr", "0.2",
        "-a", "2.0",
        "-K", "0.1", "0.5", "0.05",
        "--eos", "isothermal"
    ])

    params = build_params(parser_type="dispersion")

    assert params["S"] == 1.5e4
    assert params["Pr"] == 0.2
    assert params["a"] == 2.0
    assert params["kmin"] == 0.1
    assert params["kmax"] == 0.5
    assert params["kinc"] == 0.05
    # Isothermal EOS -> parallel/perpendicular indices are 1.0, 1.0
    assert params["parallel_index"] == 1.0
    assert params["perpendicular_index"] == 1.0

def test_build_params_maximum(monkeypatch):
    # Simulate arguments passed to eigenmodes-maxima.py
    monkeypatch.setattr("sys.argv", [
        "eigenmodes-maxima.py",
        "-d", "S",
        "-R", "1e3", "1e5", "1e4",
        "--CGL"
    ])

    params = build_params(parser_type="maximum")

    assert params["dependence"] == "S"
    assert params["vmin"] == 1e3
    assert params["vmax"] == 1e5
    assert params["vinc"] == 1e4
    assert params["CGL"] is True
    # The swept parameter 'S' should be set to None in build_params
    assert params["S"] is None
