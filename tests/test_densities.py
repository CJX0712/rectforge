"""密度：归一化自检、采样-log_prob 同源、解析速度场金标准。"""
import numpy as np
import pytest

from rectforge.core.errors import DensitySpecError
from rectforge.data.densities import ALL_DENSITIES, gaussian_transfer_velocity, make_density


def test_normalization_all_densities():
    """G1a：全部密度梯形积分归一化（checker 盒界不连续，容差 2e-3）。"""
    for name in (*ALL_DENSITIES, "gauss2gauss"):
        d = make_density(name)
        err = d.normalization_error()
        tol = 2e-3 if name == "checker" else 1e-4
        assert err < tol, f"{name}: norm err {err:.3e} >= {tol:.1e}"


def test_check_normalization_returns_error():
    d = make_density("gmm8")
    err = d.check_normalization()
    assert 0.0 <= err < 1e-4


@pytest.mark.parametrize("name", ["gmm8", "ring", "wave", "checker", "gauss2gauss"])
def test_sample_log_prob_consistency(name):
    """sample 与 log_prob 同源：大样本平均 log p 应接近熵的负值量级（有限、非偏）。"""
    d = make_density(name)
    x = d.sample(np.random.default_rng(0), 4000)
    assert x.shape == (4000, d.dim)
    lp = d.log_prob(x)
    assert np.all(np.isfinite(lp))
    # 样本应落在 bounds 内
    lo, hi = d.bounds
    assert np.all(x >= lo - 1e-9) and np.all(x <= hi + 1e-9)


def test_unknown_density_raises():
    with pytest.raises(DensitySpecError, match="E101"):
        make_density("nope")


def test_gmm8_moments():
    """全局矩闭式：Var = 4*mean(cos^2)+sigma^2 = 2+sigma^2。"""
    d = make_density("gmm8")
    x = d.sample(np.random.default_rng(1), 200_000)
    assert np.abs(x.mean(axis=0)).max() < 0.05
    expect_var = 2.0 + 0.22**2
    assert np.abs(x.var(axis=0) - expect_var).max() < 0.05


def test_gmm_log_prob_vs_scipy():
    from scipy.stats import norm

    d = make_density("gmm8")
    x = np.array([[2.0, 0.0], [0.0, 0.0], [1.5, -1.0]])
    # 手算：等权 8 分量对角高斯
    lp = []
    for xi in x:
        comp = np.log(d.weights) + np.array(
            [np.log(norm.pdf(xi[0], m[0], s[0]) * norm.pdf(xi[1], m[1], s[1]))
             for m, s in zip(d.means, d.stds, strict=True)]
        )
        from scipy.special import logsumexp

        lp.append(logsumexp(comp))
    assert np.allclose(d.log_prob(x), lp, atol=1e-12)


def test_gaussian_transfer_velocity_transport():
    """金标准：解析速度场 ODE 输运 N(0,I) -> N(mu,cov)，矩误差 < 0.02。"""
    from rectforge.sampling.ode import NFECounter, rk4_integrate

    d = make_density("gauss2gauss")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16384, 2))

    def u(x_, t):
        return gaussian_transfer_velocity(x_, t, d.mu, d.cov, np.zeros(2), np.eye(2))

    y = rk4_integrate(u, x, 100, NFECounter())
    assert np.abs(y.mean(axis=0) - d.mu).max() < 0.02
    assert np.abs(np.cov(y.T) - d.cov).max() < 0.02


def test_gaussian_transfer_velocity_matches_exact_posterior(rng):
    """b_t(x) 与闭式后验均值 E[x1-x0|x_t] 一致（independent 高斯解析）。"""
    d = make_density("gauss2gauss")
    m0, s0 = d.mu, d.cov
    m1, s1 = np.zeros(2), np.eye(2)
    for t in (0.1, 0.4, 0.7, 0.95):
        x = rng.normal(size=(64, 2)) * 2
        b = gaussian_transfer_velocity(x, t, m0, s0, m1, s1)
        mt = (1 - t) * m0 + t * m1
        st = (1 - t) ** 2 * s0 + t**2 * s1
        cross = -(1 - t) * s0 + t * s1
        expect = (m1 - m0) + np.linalg.solve(st, (x - mt).T).T @ cross.T
        assert np.allclose(b, expect, atol=1e-12)
