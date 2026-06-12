import numpy as np
import pytest
from test import bootstrap_ci, welch_ttest


def test_bootstrap_ci_returns_tuple_of_two():
    rng = np.random.default_rng(42)
    rewards = rng.normal(200, 30, 150).tolist()
    lo, hi = bootstrap_ci(rewards, n_bootstrap=1000, ci=0.95, seed=42)
    assert lo < hi


def test_bootstrap_ci_contains_true_mean():
    rng = np.random.default_rng(0)
    rewards = rng.normal(200, 30, 150).tolist()
    lo, hi = bootstrap_ci(rewards, n_bootstrap=5000, ci=0.95, seed=0)
    assert lo < 200 < hi


def test_bootstrap_ci_wider_with_more_variance():
    rng = np.random.default_rng(1)
    low_var = rng.normal(200, 5, 150).tolist()
    high_var = rng.normal(200, 50, 150).tolist()
    lo_l, hi_l = bootstrap_ci(low_var, n_bootstrap=2000, ci=0.95, seed=1)
    lo_h, hi_h = bootstrap_ci(high_var, n_bootstrap=2000, ci=0.95, seed=1)
    assert (hi_l - lo_l) < (hi_h - lo_h)


def test_welch_ttest_significant_difference():
    rng = np.random.default_rng(42)
    a = rng.normal(210, 5, 150).tolist()
    b = rng.normal(180, 30, 150).tolist()
    p = welch_ttest(a, b)
    assert p < 0.05


def test_welch_ttest_no_significant_difference():
    rng = np.random.default_rng(42)
    a = rng.normal(200, 30, 150).tolist()
    b = rng.normal(202, 30, 150).tolist()
    p = welch_ttest(a, b)
    assert p > 0.05


def test_welch_ttest_returns_float_between_0_and_1():
    rng = np.random.default_rng(42)
    a = rng.normal(200, 30, 50).tolist()
    b = rng.normal(180, 30, 50).tolist()
    p = welch_ttest(a, b)
    assert 0.0 <= p <= 1.0
