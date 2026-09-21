"""Tests for Extrapolator log-space mode (scripts/eigenmodes-maxima.py).

Log mode fits the polynomial to log(y) over positive-history entries only
and exponentiates the prediction. It is exact when log(y) is polynomial in
x (e.g. exponential histories), not when y itself is polynomial in x: for
a power law y=x^2, log(y)=2*log(x) is not a polynomial in x, so log mode
is only approximate there while linear mode at deg=2 is exact. The tests
below assert the mathematically true behavior (see test_log_mode_power_law).
"""

import importlib.util

import numpy as np
import pytest


def _load_extrapolator():
    spec = importlib.util.spec_from_file_location(
        "maxima_extrap_mod", "scripts/eigenmodes-maxima.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Extrapolator


def test_log_mode_exponential_exact_while_linear_errs():
    """Log mode is (near-)exact for y=2^x at deg>=1; linear mode errs."""
    Extrapolator = _load_extrapolator()
    for deg in (1, 2, 3):
        log_ex = Extrapolator(maxdeg=deg, ymin=None, log_y=True)
        lin_ex = Extrapolator(maxdeg=deg, ymin=None, log_y=False)
        for x in (1.0, 2.0, 3.0, 4.0):
            log_ex.add(x, 2.0**x)
            lin_ex.add(x, 2.0**x)
        assert log_ex.predict(5.0) == pytest.approx(32.0, rel=1e-9)
        assert abs(lin_ex.predict(5.0) - 32.0) > 1.0


def test_log_mode_power_law_is_approximate_not_exact():
    """For y=x^2, log(y) is not polynomial in x: log mode approximates.

    Measured with centered/scaled numerics: history 1,2,3 -> predict at 4
    gives ~29.7 (deg=1) and ~11.39 (deg=2) vs true 16, while linear deg=2
    is exact. Log mode must still return a positive finite value.
    """
    Extrapolator = _load_extrapolator()
    log_ex = Extrapolator(maxdeg=1, ymin=None, log_y=True)
    lin_ex = Extrapolator(maxdeg=2, ymin=None, log_y=False)
    for x in (1.0, 2.0, 3.0):
        log_ex.add(x, x**2)
        lin_ex.add(x, x**2)
    pred_log = log_ex.predict(4.0)
    assert pred_log is not None and np.isfinite(pred_log) and pred_log > 0.0
    assert abs(pred_log - 16.0) > 1.0  # not exact: documented behavior
    assert lin_ex.predict(4.0) == pytest.approx(16.0, rel=1e-9)


def test_log_mode_filters_nonpositive_history():
    """Non-positive entries are skipped; too few positives -> None."""
    Extrapolator = _load_extrapolator()
    ex = Extrapolator(maxdeg=1, ymin=None, log_y=True)
    ex.add(1.0, 0.0)
    ex.add(2.0, -3.0)
    ex.add(3.0, 4.0)
    # Only one positive entry < minpoints=2 -> caller falls back to bracket
    assert ex.predict(4.0) is None

    ex2 = Extrapolator(maxdeg=1, ymin=None, log_y=True)
    ex2.add(1.0, 0.0)  # filtered
    ex2.add(2.0, 2.0)
    ex2.add(3.0, 4.0)
    pred = ex2.predict(4.0)
    assert pred is not None and np.isfinite(pred) and pred > 0.0


def test_log_mode_ymin_clamp_applies_after_exponentiation():
    """The ymin clamp applies to the exponentiated prediction."""
    Extrapolator = _load_extrapolator()
    ex = Extrapolator(maxdeg=1, ymin=1e-6, log_y=True)
    for x in (1.0, 2.0, 3.0):
        ex.add(x, 10.0 ** (-x))  # decaying history extrapolates downward
    pred = ex.predict(10.0)
    assert pred is not None
    assert pred >= 1e-6


def test_log_mode_insufficient_history_returns_none():
    """Fewer than minpoints total entries -> None, as in linear mode."""
    Extrapolator = _load_extrapolator()
    ex = Extrapolator(maxdeg=2, minpoints=2, ymin=None, log_y=True)
    ex.add(1.0, 1.0)
    assert ex.predict(2.0) is None


def test_log_mode_legacy_predictions_reproduced_exactly():
    """log_y=False (the default) is byte-identical to legacy behavior."""
    Extrapolator = _load_extrapolator()
    xs = (1e3, 2e3, 3e3, 4e3)
    ys = (0.05, 0.04, 0.032, 0.027)
    default_ex = Extrapolator(maxdeg=2, ymin=1e-6)
    explicit_ex = Extrapolator(maxdeg=2, ymin=1e-6, log_y=False)
    for x, y in zip(xs, ys):
        default_ex.add(x, y)
        explicit_ex.add(x, y)
    pred_default = default_ex.predict(5e3)
    pred_explicit = explicit_ex.predict(5e3)
    assert pred_default is not None and pred_explicit is not None
    assert pred_default == pred_explicit

    # Cross-check against the direct legacy formula (centered/scaled polyfit)
    arr_xs = np.array(xs)
    arr_ys = np.array(ys)
    x0 = arr_xs.mean()
    xscale = (arr_xs.max() - arr_xs.min()) or 1.0
    coeffs = np.polyfit((arr_xs - x0) / xscale, arr_ys, 2)
    expected = max(float(np.polyval(coeffs, (5e3 - x0) / xscale)), 1e-6)
    assert pred_default == expected
