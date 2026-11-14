import numpy as np
import pytest

from pneumonia_cxr.metrics import (
    binary_metrics,
    bootstrap_ci,
    expected_calibration_error,
    fit_temperature,
    sigmoid,
    threshold_for_sensitivity,
    youden_threshold,
)


def test_binary_metrics_known_values():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.6, 0.4, 0.7])
    m = binary_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (3, 1, 3, 1)
    assert m["sensitivity"] == pytest.approx(0.75)
    assert m["specificity"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["auroc"] == pytest.approx(0.875)


def test_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.1, 0.9, 1.0])
    m = binary_metrics(y, p, youden_threshold(y, p))
    assert m["auroc"] == 1.0 and m["accuracy"] == 1.0


def test_threshold_for_sensitivity_reaches_target():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.3 * y + rng.normal(0.35, 0.2, 2000), 0, 1)
    t = threshold_for_sensitivity(y, p, 0.95)
    assert binary_metrics(y, p, t)["sensitivity"] >= 0.95


def test_ece_is_zero_for_calibrated_bins_and_large_when_overconfident():
    y = np.array([1] * 30 + [0] * 70)
    assert expected_calibration_error(y, np.full(100, 0.3)) == pytest.approx(0.0)
    assert expected_calibration_error(y, np.full(100, 0.99)) > 0.6


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(1)
    true_logits = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < sigmoid(true_logits)).astype(int)
    overconfident = true_logits * 4.0
    t = fit_temperature(overconfident, y)
    assert t == pytest.approx(4.0, rel=0.1)
    assert expected_calibration_error(y, sigmoid(overconfident / t)) < expected_calibration_error(
        y, sigmoid(overconfident)
    )


def test_bootstrap_ci_contains_point_estimate_and_supports_groups():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    p = np.clip(0.4 * y + rng.random(400) * 0.6, 0, 1)
    groups = np.repeat(np.arange(100), 4)
    point = binary_metrics(y, p, 0.5)["auroc"]
    for g in (None, groups):
        lo, hi = bootstrap_ci(y, p, 0.5, n_boot=200, groups=g)["auroc"]
        assert lo <= point <= hi
