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

    args = parser.parse_args([])
    args.n_equilibrium = 2
    with pytest.raises(ParameterError, match="Minimum number of inner collocation points .* must be >= 3"):
        validate_parameters(args)

    args = parser.parse_args([])
    args.n_inner_scale = 2
    with pytest.raises(ParameterError, match="Minimum number of resistivity layer collocation points .* must be >= 3"):
        validate_parameters(args)


def test_validate_parameters_safety_factor():
    parser = parser_setup()
    args = parser.parse_args([])
    args.inner_resolution_safety = 0.5
    with pytest.raises(ParameterError, match="Inner resolution safety factor .* must be >= 1.0"):
        validate_parameters(args)


def test_validate_parameters_conflicting_scaling():
    parser = parser_setup()
    args = parser.parse_args([])

    args.scaling_factor = 1.0
    args.inner_scale = 0.1
    with pytest.raises(ParameterError, match="Only one of these options may be set at a time"):
        validate_parameters(args)

    args = parser.parse_args([])
    args.inner_scale = -0.1
    with pytest.raises(ParameterError, match="Resistive scale .* must be > 0"):
        validate_parameters(args)


def test_build_params_inner_scale_aliases(monkeypatch):
    # Test canonical options
    monkeypatch.setattr("sys.argv", [
        "eigenmodes-compute.py",
        "--inner-scale", "0.025",
        "--inner-resolution-safety", "1.5",
        "--n-equilibrium", "7",
        "--n-inner-scale", "9",
    ])
    params = build_params(parser_type="dispersion")
    assert params.inner_scale == 0.025
    assert params.delta == 0.025
    assert params.inner_resolution_safety == 1.5
    assert params.n_equilibrium == 7
    assert params.n_inner == 7
    assert params.n_inner_scale == 9
    assert params.n_resistivity == 9

    # Test legacy aliases
    monkeypatch.setattr("sys.argv", [
        "eigenmodes-compute.py",
        "--resistive-scale", "0.035",
        "--n-inner", "11",
        "--n-resistivity", "13",
    ])
    params_legacy = build_params(parser_type="dispersion")
    assert params_legacy.inner_scale == 0.035
    assert params_legacy.delta == 0.035
    assert params_legacy.n_equilibrium == 11
    assert params_legacy.n_inner == 11
    assert params_legacy.n_inner_scale == 13
    assert params_legacy.n_resistivity == 13

def test_build_params_dispersion(monkeypatch):
    # Simulate arguments passed to eigenmodes-compute.py
    monkeypatch.setattr("sys.argv", [
        "eigenmodes-compute.py",
        "-S", "1.5e4",
        "-Pr", "0.2",
        "-a", "2.0",
        "-K", "0.1", "0.5", "0.05",
        "--eos", "isothermal",
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
    assert "Cmean" not in params

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

def test_build_params_plot_2d(monkeypatch):
    # Simulate arguments passed to eigenmodes-maps.py or eigenmodes-profiles.py with 2D options
    monkeypatch.setattr("sys.argv", [
        "eigenmodes-maps.py",
        "--nx", "150",
        "--nperiods", "2.5",
        "--zmin", "-1.0",
        "--zmax", "1.0"
    ])

    params = build_params(parser_type="plot")

    assert params["nx"] == 150
    assert params["nperiods"] == 2.5
    assert params["zmin"] == -1.0
    assert params["zmax"] == 1.0

def test_validate_parameters_plot_2d():
    from tearing_eigenmodes.parser import parser_setup, validate_parameters
    parser = parser_setup()
    
    # Try invalid nx <= 0
    args = parser.parse_args([])
    args.nx = 0
    with pytest.raises(ParameterError, match="Number of x points .* must be > 0"):
        validate_parameters(args)

    # Try invalid nperiods <= 0
    args = parser.parse_args([])
    args.nx = 100
    args.nperiods = -1.0
    with pytest.raises(ParameterError, match="Number of periods .* must be > 0"):
        validate_parameters(args)


def test_build_params_log_extrapolation(monkeypatch):
    """--log-extrapolation defaults to False and maps to params when set."""
    # Default: flag absent -> False
    monkeypatch.setattr("sys.argv", [
        "eigenmodes-maxima.py",
        "-d", "S",
        "-R", "1e3", "1e5", "1e4",
    ])
    params = build_params(parser_type="maximum")
    assert params["log_extrapolation"] is False

    # Explicit flag -> True
    monkeypatch.setattr("sys.argv", [
        "eigenmodes-maxima.py",
        "-d", "S",
        "-R", "1e3", "1e5", "1e4",
        "--log-extrapolation",
    ])
    params_log = build_params(parser_type="maximum")
    assert params_log["log_extrapolation"] is True

def test_build_params_gevp_method(monkeypatch):
    """--gevp-method follows --step unless given explicitly."""
    base = ["eigenmodes-maxima.py", "-d", "S", "-R", "1e3", "1e5", "1e4"]

    monkeypatch.setattr("sys.argv", base)
    assert build_params(parser_type="maximum")["gevp_method"] == "qz"

    monkeypatch.setattr("sys.argv", base + ["--step"])
    assert build_params(parser_type="maximum")["gevp_method"] == "shift-invert"

    monkeypatch.setattr("sys.argv", base + ["--step", "--gevp-method", "qz"])
    assert build_params(parser_type="maximum")["gevp_method"] == "qz"

    monkeypatch.setattr("sys.argv", base + ["--gevp-method", "shift-invert"])
    assert build_params(parser_type="maximum")["gevp_method"] == "shift-invert"

def test_build_params_dispersion_gevp_method(monkeypatch):
    """The dispersion parser keeps --gevp-method as requested, None if absent."""
    base = ["eigenmodes-compute.py", "-K", "0.1", "0.5", "0.1"]

    monkeypatch.setattr("sys.argv", base)
    assert build_params(parser_type="dispersion")["gevp_method"] is None

    monkeypatch.setattr("sys.argv", base + ["--gevp-method", "shift-invert"])
    assert build_params(parser_type="dispersion")["gevp_method"] == "shift-invert"


def test_resolve_gevp_method():
    from tearing_eigenmodes.parser import resolve_gevp_method

    assert resolve_gevp_method(None, True) == "shift-invert"
    assert resolve_gevp_method(None, False) == "qz"
    assert resolve_gevp_method("qz", True) == "qz"
    assert resolve_gevp_method("shift-invert", False) == "shift-invert"

def test_inner_resolution_safety_default_and_parsing():
    """Inner resolution safety factor must default to 1.01 and parse explicit values correctly."""
    from tearing_eigenmodes import SimulationParams
    from tearing_eigenmodes.parser import parser_setup, build_params

    # 1. SimulationParams dataclass default
    p = SimulationParams()
    assert p.inner_resolution_safety == 1.01

    # 2. CLI parser default
    parser = parser_setup()
    args_default = parser.parse_args([])
    assert args_default.inner_resolution_safety == 1.01

    # 3. Explicit 1.0 (no margin)
    args_10 = parser.parse_args(["--inner-resolution-safety", "1.0"])
    assert args_10.inner_resolution_safety == 1.0

    # 4. Explicit 1.5
    args_15 = parser.parse_args(["--inner-resolution-safety", "1.5"])
    assert args_15.inner_resolution_safety == 1.5

    # 5. Invalid safety factor < 1.0
    args_invalid = parser.parse_args(["--inner-resolution-safety", "0.9"])
    with pytest.raises(ParameterError, match="Inner resolution safety factor .* must be >= 1.0"):
        validate_parameters(args_invalid)




def test_build_params_plot_with_dependence(monkeypatch):
    """Plot scripts accept --dependence/--value without a sweep --range."""
    monkeypatch.setattr("sys.argv", [
        "eigenmodes-profiles.py", "-d", "S", "--value", "1e4",
    ])
    params = build_params(parser_type="plot")
    assert params["dependence"] == "S"
    assert params["value_plot"] == 1e4
    assert params["S"] is None


@pytest.mark.parametrize("argv", [
    ["--CGL", "-ξ", "0.1"],
    ["--CGL", "-w", "0.5"],
])
def test_validate_parameters_cgl_rejects_transverse_field_and_width(argv):
    args = parser_setup().parse_args(argv)
    with pytest.raises(ParameterError, match="not supported with --CGL"):
        validate_parameters(args)


def test_validate_parameters_cgl_rejects_width_sweep():
    parser = parser_setup()
    parser.add_argument("--dependence", default=None)
    parser.add_argument("--range", type=float, nargs=3, default=[0, 1, 0.1])
    args = parser.parse_args(["--CGL", "--dependence", "w", "--range", "0", "1", "0.5"])
    with pytest.raises(ParameterError, match="half‑width"):
        validate_parameters(args)


def test_build_params_zeta(monkeypatch):
    monkeypatch.setattr("sys.argv", ["eigenmodes-compute.py"])
    assert build_params(parser_type="dispersion")["zeta"] == 1.0
    monkeypatch.setattr("sys.argv", ["eigenmodes-compute.py", "--zeta", "0.25"])
    assert build_params(parser_type="dispersion")["zeta"] == 0.25


@pytest.mark.parametrize("zeta", ["-0.1", "1.5"])
def test_validate_parameters_zeta_range(zeta):
    args = parser_setup().parse_args(["--zeta", zeta])
    with pytest.raises(ParameterError, match="ζ"):
        validate_parameters(args)


def test_build_parser_maximum_has_description(monkeypatch):
    from tearing_eigenmodes.parser import build_parser
    captured = {}
    import argparse as _argparse
    original = _argparse.ArgumentParser.parse_args

    def spy(self, *a, **k):
        captured["description"] = self.description
        return original(self, *a, **k)

    monkeypatch.setattr(_argparse.ArgumentParser, "parse_args", spy)
    monkeypatch.setattr("sys.argv", ["eigenmodes-maxima.py", "-R", "1e3", "1e5", "1e4"])
    build_parser(parser_type="maximum")
    assert "maximum growth rate" in captured["description"]


def test_validate_parameters_imag_range_order():
    args = parser_setup().parse_args(["-I", "1", "-1"])
    with pytest.raises(ParameterError, match="Imaginary"):
        validate_parameters(args)


def test_validate_parameters_cgl_delta_beta_below_two():
    args = parser_setup().parse_args(["--CGL", "-Δβ", "2.0"])
    with pytest.raises(ParameterError, match="< 2"):
        validate_parameters(args)
    validate_parameters(parser_setup().parse_args(["-Δβ", "2.0"]))  # Classical ignores Δβ


@pytest.mark.parametrize("argv, dest", [
    (["-K", "0", "1", "0"], "wavenumber_range"),
    (["-R", "0", "1", "-0.1"], "range"),
])
def test_validate_parameters_sweep_increment_positive(argv, dest):
    parser = parser_setup()
    if dest == "wavenumber_range":
        parser.add_argument("--wavenumber-range", "-K", type=float, nargs=3, default=[0, 1, 0.01])
    else:
        parser.add_argument("--range", "-R", type=float, nargs=3, default=[0, 1, 0.1])
    args = parser.parse_args(argv)
    with pytest.raises(ParameterError, match="increment"):
        validate_parameters(args)
