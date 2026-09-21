"""Dependency declaration regression test (F9).

Asserts that third-party imports used by the package and scripts are
declared in pyproject.toml, and that the key undeclared-previously
imports resolve in this environment.
"""

import importlib
import pathlib
import tomllib


def _pyproject() -> dict:
    path = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _dep_names(deps: list) -> list:
    names = []
    for dep in deps:
        names.append(dep.split(";")[0].split("[")[0].split("=")[0].split(">")[0].split("<")[0].strip().lower())
    return names


def test_core_dependencies_declared() -> None:
    project = _pyproject()["project"]
    names = _dep_names(project.get("dependencies", []))
    assert "sympy" in names
    assert "matplotlib" in names


def test_dev_extras_declared() -> None:
    dev = _pyproject()["project"].get("optional-dependencies", {}).get("dev", [])
    names = _dep_names(dev)
    assert "pytest" in names
    assert "mypy" in names


def test_declared_imports_resolve() -> None:
    assert importlib.import_module("sympy") is not None
    assert importlib.import_module("matplotlib.pyplot") is not None
