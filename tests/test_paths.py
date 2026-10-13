"""路径与耦合：端点/边界不变量、时间采样支撑、OT 配对最优性。"""
import numpy as np
import pytest

from rectforge.paths.coupling import available_scipy, independent_pair, make_coupling, ot_pair
from rectforge.paths.interpolants import (
    LinearPath,
    SIPath,
    VPPath,
    make_path,
    sample_time,
)


# ---- 路径 ------------------------------------------------------------
def test_linear_path_endpoints(rng):
    x0, x1 = rng.normal(size=(16, 2)), rng.normal(size=(16, 2))
    p = LinearPath()
    xt0, _ = p.conditional(x0, x1, np.zeros(16), rng)
    xt1, _ = p.conditional(x0, x1, np.ones(16), rng)
    assert np.allclose(xt0, x0) and np.allclose(xt1, x1)


def test_vp_path_endpoints(rng):
    x0, x1 = rng.normal(size=(16, 2)), rng.normal(size=(16, 2))
    p = VPPath()
    xt0, _ = p.conditional(x0, x1, np.zeros(16), rng)
    xt1, _ = p.conditional(x0, x1, np.ones(16), rng)
    assert np.allclose(xt0, x0) and np.allclose(xt1, x1)
    # 速度目标 = dx_t/dt
    t = np.full(16, 0.5)
    _, v = p.conditional(x0, x1, t, rng)
    h = 1e-6
    xp, _ = p.conditional(x0, x1, t + h, rng)
    xm, _ = p.conditional(x0, x1, t - h, rng)
    assert np.allclose(v, (xp - xm) / (2 * h), atol=1e-6)


def test_si_path_gamma_zero_at_ends(rng):
    x0, x1 = rng.normal(size=(16, 2)), rng.normal(size=(16, 2))
    p = SIPath(sigma=0.25)
    assert np.allclose(p.gamma(np.array([0.0, 1.0])), 0.0)
    for t in (0.0, 1.0):
        xt, _ = p.conditional(x0, x1, np.full(16, t), rng)
        assert np.allclose(xt, x0 if t == 0.0 else x1)


def test_si_path_velocity_matches_finite_diff(rng):
    x0, x1 = rng.normal(size=(32, 2)), rng.normal(size=(32, 2))
    p = SIPath(sigma=0.3)
    r = np.random.default_rng(0)
    t = np.full(32, 0.4)
    _, v = p.conditional(x0, x1, t, r)
    # 固定同批 eps 的条件下 v = dx_t/dt
    h = 1e-6
    r2 = np.random.default_rng(0)
    xtp, _ = p.conditional(x0, x1, t + h, r2)
    r3 = np.random.default_rng(0)
    xtm, _ = p.conditional(x0, x1, t - h, r3)
    assert np.allclose(v, (xtp - xtm) / (2 * h), atol=1e-5)


def test_sample_time_support():
    r = np.random.default_rng(0)
    tu = sample_time("uniform", 100_000, r)
    assert tu.min() >= 0.0 and tu.max() < 1.0
    tl = sample_time("logit_normal", 100_000, r)
    assert tl.min() > 0.0 and tl.max() < 1.0


def test_sample_time_unknown_raises():
    with pytest.raises(ValueError, match="unknown time sampling"):
        sample_time("beta", 4, np.random.default_rng(0))


def test_make_path_unknown_raises():
    with pytest.raises(ValueError, match="unknown path"):
        make_path("cubic")


# ---- 耦合 ------------------------------------------------------------
def test_ot_pair_is_permutation(rng):
    x0 = rng.normal(size=(64, 2))
    x1 = rng.normal(size=(64, 2))
    paired = ot_pair(x0, x1)
    assert paired.shape == x1.shape
    assert np.array_equal(np.sort(paired, axis=0), np.sort(x1, axis=0))


def test_ot_pair_cost_not_worse_than_independent(rng):
    x0 = rng.normal(size=(128, 2))
    x1 = rng.normal(size=(128, 2))

    def cost(a, b):
        return float(((a - b) ** 2).sum())

    assert cost(x0, ot_pair(x0, x1)) <= cost(x0, x1) + 1e-9


def test_ot_pair_matches_lsa_directly(rng):
    from scipy.optimize import linear_sum_assignment

    x0 = rng.normal(size=(32, 2))
    x1 = rng.normal(size=(32, 2))
    cost = ((x0[:, None, :] - x1[None, :, :]) ** 2).sum(-1)
    _, col = linear_sum_assignment(cost)
    assert np.allclose(ot_pair(x0, x1), x1[col])


def test_make_coupling_registry():
    assert callable(make_coupling("independent"))
    assert (make_coupling("ot") is independent_pair) != available_scipy()
    with pytest.raises(ValueError, match="unknown coupling"):
        make_coupling("gromov")
