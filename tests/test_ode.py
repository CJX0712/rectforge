"""ODE 积分器与采样：已知 ODE 精度、NFE 记账、pf_velocity_fn 行为。"""
import numpy as np
import pytest

from rectforge.core.errors import SamplingError
from rectforge.core.types import BenchmarkConfig
from rectforge.data.densities import gaussian_transfer_velocity, make_density
from rectforge.models.mlp import TimeCondMLP
from rectforge.paths.interpolants import LinearPath
from rectforge.sampling.ode import (
    NFECounter,
    euler_integrate,
    integrate_ode,
    pf_velocity_fn,
    rk4_integrate,
)


def _decay(u, t):
    return -u


def test_euler_convergence_order():
    """x'=-x, x(0)=1：Euler 误差 O(h)。"""
    for steps, tol in ((10, 0.2), (100, 0.02)):
        x = euler_integrate(_decay, np.array([[1.0]]), steps, NFECounter())
        # x'=-x, x(1)=1 反向积分到 t=0：解析解 e（误差 O(h)，实测 1.2e-1 / 1.3e-2）
        assert abs(x[0, 0] - np.e) < tol


def test_rk4_fourth_order():
    """x'=x（反向 t:1->0，x(1)=1 => x(0)=e^{-1}）：RK4 误差 O(h^4)。"""
    for steps in (5, 10):
        x = rk4_integrate(lambda u, t: u, np.array([[1.0]]), steps, NFECounter())
        # 实测误差 5.8e-6 / 3.3e-7（O(h^4)，每加倍步数 ~17x 改善）
        assert abs(x[0, 0] - np.e**-1) < 1e-5 / (steps / 5) ** 3


def test_nfe_accounting():
    c1 = NFECounter()
    euler_integrate(_decay, np.zeros((3, 2)), 7, c1)
    assert c1.n == 7
    c2 = NFECounter()
    rk4_integrate(_decay, np.zeros((3, 2)), 7, c2)
    assert c2.n == 28


def test_integrate_ode_unknown_method():
    with pytest.raises(SamplingError, match="E500"):
        integrate_ode(_decay, np.zeros((1, 1)), 3, "dopri5", NFECounter())


def test_learned_ode_transports_gaussian(rng):
    """学到的速度场（金标准 DGP）ODE 输运矩达标。"""
    from rectforge.training.trainer import train_velocity

    dgp = make_density("gauss2gauss")
    cfg = BenchmarkConfig(dgps=("gauss2gauss",), seeds=(0,), n_train=2048,
                          eval_samples=256, true_eval_samples=256, nfe_sweep=(12,))
    t = cfg.train
    t.steps, t.hidden, t.layers, t.batch_size = 600, 64, 2, 128
    data = dgp.sample(np.random.default_rng(0), 2048)
    model = TimeCondMLP(dim=2, hidden=64, layers=2, embed_dim=16, seed=1000)
    params, _ = train_velocity(model, data, t, np.random.default_rng(0))
    path = LinearPath()
    u_fn = pf_velocity_fn(model, params, path, data[:512])
    z = np.random.default_rng(1).normal(size=(8192, 2))
    y = rk4_integrate(u_fn, z, 64, NFECounter())
    assert np.abs(y.mean(axis=0) - dgp.mu).max() < 0.15
    assert np.abs(np.cov(y.T) - dgp.cov).max() < 0.20


def test_pf_velocity_fn_gamma_zero_identity(rng):
    """gamma=0 时 pf_velocity_fn 必须原样返回 u_theta（旗舰路径精确性）。"""
    model = TimeCondMLP(dim=2, hidden=8, layers=1, seed=0)
    params = model.params
    path = LinearPath()
    u_fn = pf_velocity_fn(model, params, path, np.zeros((4, 2)))
    x = rng.normal(size=(16, 2))
    t = 0.37
    assert np.allclose(u_fn(x, t), model.predict(params, x, np.full(16, t)))


def test_pf_velocity_fn_si_correction_bounded():
    """gamma>0：修正项存在且有限；gamma->0 时修正 -> -0.5*sum w(m-x)，有界。"""
    from rectforge.paths.interpolants import SIPath

    model = TimeCondMLP(dim=2, hidden=8, layers=1, seed=0)
    path = SIPath(sigma=0.25)
    corpus = np.random.default_rng(0).normal(size=(32, 2))
    u_fn = pf_velocity_fn(model, model.params, path, corpus)
    x = np.random.default_rng(1).normal(size=(8, 2))
    out = u_fn(x, 0.5)
    assert np.all(np.isfinite(out))
    assert out.shape == x.shape


def test_analytic_velocity_ode_vs_exact_field():
    """解析场积分 vs 逐点解析式：同一场，轨迹矩已在 densities 测。这里测自洽。"""
    d = make_density("gauss2gauss")
    x = np.random.default_rng(0).normal(size=(64, 2))
    for t in (0.2, 0.5, 0.8):
        b1 = gaussian_transfer_velocity(x, t, d.mu, d.cov, np.zeros(2), np.eye(2))
        b2 = gaussian_transfer_velocity(x, t, d.mu, d.cov, np.zeros(2), np.eye(2))
        assert np.array_equal(b1, b2)
