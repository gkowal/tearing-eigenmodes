import tearing_eigenmodes

EXPECTED_ROOT_API = frozenset({
    "DeltaError",
    "build_dpath",
    "build_params",
    "check_state",
    "compile_metadata",
    "eigenmodes",
    "estimate_max",
    "load_eigenmodes",
    "print_info",
    "refine_eigenvalues",
    "refine_inner_scale",
    "refine_wavenumber_bracket",
    "save_eigenmode",
    "setup_logging",
    "SimulationParams",
    "validate_and_fix_file",
    "write_results",
})


def test_root_all_matches_contract():
    assert set(tearing_eigenmodes.__all__) == set(EXPECTED_ROOT_API)


def test_root_names_importable_from_root():
    for name in EXPECTED_ROOT_API:
        assert getattr(tearing_eigenmodes, name) is not None
