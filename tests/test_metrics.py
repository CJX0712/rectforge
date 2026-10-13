"""评测指标：KL 协议地板、MMD 无偏性、sliced W1、错误路径。"""
import numpy as np
import pytest

from rectforge.core.errors import MetricError
from rectforge.data.densities import make_density
from rectforge.eval.metrics import (
    kde_log_density,
    kl_p_q,
    mmd2,
    silverman_bandwidth,
    sliced_w1,
)


def test_silverman_formula():
    X = np.random.default_rng(0).normal(size=(1000, 2))
    h = silverman_bandwidth(X)
    expect = X.std(axis=0).mean() * (4 * 1000 / 4) ** (-1 / 6)
    assert h == pytest.approx(expect)


def test_silverman_zero_variance_raises():
    with pytest.raises(MetricError, match="E501"):
        silverman_bandwidth(np.zeros((10, 2)))


def test_kl_self_near_zero():
    """协议地板：同分布两批样本 KL≈0（|KL|<0.05，G1b 容差口径）。"""
    d = make_density("gauss2gauss")
    a = d.sample(np.random.default_rng(1), 3000)
    b = d.sample(np.random.default_rng(2), 3000)
    assert abs(kl_p_q(d.log_prob, a, b)) < 0.05


def test_kl_positive_for_shifted():
    d = make_density("gauss2gauss")
    a = d.sample(np.random.default_rng(1), 3000)
    b = d.sample(np.random.default_rng(2), 3000) + np.array([1.0, 0.0])
    assert kl_p_q(d.log_prob, a, b) > 0.5


def test_kde_integrates_to_one():
    X = np.random.default_rng(0).normal(size=(500, 2))
    h = silverman_bandwidth(X)
    g = np.linspace(-4, 4, 201)
    gx, gy = np.meshgrid(g, g)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    lp = kde_log_density(X, h, pts)
    area = np.trapezoid(np.trapezoid(np.exp(lp).reshape(201, 201), g, axis=1), g)
    assert abs(area - 1.0) < 0.02


def test_mmd2_self_near_zero():
    X = np.random.default_rng(0).normal(size=(400, 2))
    assert abs(mmd2(X, X)) < 0.01


def test_mmd2_detects_shift():
    X = np.random.default_rng(0).normal(size=(400, 2))
    Y = np.random.default_rng(1).normal(size=(400, 2)) + 2.0
    assert mmd2(X, Y) > 0.1


def test_mmd2_needs_two_samples():
    with pytest.raises(MetricError, match="E501"):
        mmd2(np.zeros((1, 2)), np.zeros((5, 2)))


def test_sliced_w1_self_near_zero_and_shift_positive():
    X = np.random.default_rng(0).normal(size=(600, 2))
    assert sliced_w1(X, X) == pytest.approx(0.0, abs=1e-12)
    Y = X + 2.0
    assert sliced_w1(X, Y) > 0.5


def test_kl_shape_mismatch_raises():
    d = make_density("gauss2gauss")
    a = d.sample(np.random.default_rng(1), 10)
    with pytest.raises(MetricError, match="E501"):
        kl_p_q(lambda x: np.zeros(x.shape[0] + 1), a, a)
