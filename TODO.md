# Project TODO List

This list tracks planned improvements for the `tearing-eigenmodes` project, categorized by impact area.

## 1. Robustness & Numerical Stability
- [x] **Atomic State Saving**: Prevent `.npz` corruption by using temporary files and `os.replace()` for updates. (Implemented in `io.save_eigenmode`)
- [x] **Full Reproducibility in State Files**: Save all physical and numerical parameters directly into `.npz` files to make them self-describing.
- [x] **I/O Performance**: Optimize `load_eigenmodes` in `io.py` by caching the `ChebyshevRationalGrid` object or its nodes to avoid repeated re-instantiation. (Obsolete: grid is saved directly in state files and loaded as a raw array)
- [ ] **Refinement I/O Caching**: Cache result tables loaded during parameter sweep initialization in `refinement.py` to avoid reading `.npz` files four times sequentially.
- [ ] **CGL Factor Consolidation**: Consolidate duplicated $C$ and $\mu$ formulas from `grid.py` and `physics.py` into a single central function in `physics.py`.
- [ ] **Explicit Numerical Guards**: Add checks for potential division-by-zero or `log(negative)` in physical scaling laws to provide clearer error messages.

## 2. Code Quality & Maintenance
- [ ] **Typed Parameters**: Transition the `params` dictionary to a `dataclass` or `NamedTuple` for better type safety and IDE support.
- [ ] **Eliminate Script Duplication**: Extract shared state-management logic from the `task()` functions in CLI scripts into `io.py`.
- [x] **Library-based Logging**: Move `SmartStreamHandler` into the library (e.g., `tearing_eigenmodes.logging_utils`) to avoid code duplication across scripts.
- [x] **Type Hinting**: Expand type hints across all modules, particularly in `solver.py` and `refinement.py`.
- [ ] **Type Hinting for Scripts**: Add type annotations to CLI scripts under `scripts/` to expand mypy static analysis coverage.
- [ ] **Automated Static Analysis**: Run static type verification with `mypy` as part of the test runner or CI checks.

## 3. Execution & User Experience
- [x] **Graceful Interrupts**: Improve `multiprocessing` handling to ensure clean termination on `Ctrl+C`. (Implemented in `scripts/eigenmodes-compute.py` and `scripts/eigenmodes-maxima.py`)
- [x] **Advanced Logging**:
    *   Add a `--log-file` argument to capture detailed `DEBUG` output to disk.
    *   Include timestamps in log messages for long-running sweeps.
- [x] **Range Validation**: Add checks to ensure generated sweep ranges (wavenumbers or dependent parameters) are not empty before starting calculations. (Implemented in `scripts/eigenmodes-compute.py` and `scripts/eigenmodes-maxima.py`)

## 4. Testing
- [x] **Verification Suite**: Add a `tests/` directory with `pytest` cases for:
    *   Grid determination logic (`select_NC`).
    *   Layer thickness diagnostics.
    *   Parameter parsing and validation.
