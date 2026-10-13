"""DDPM/DDIM 基线：调度数学、采样形状、NFE、确定性。"""
import numpy as np

from rectforge.baselines.ddpm import DiffusionSchedule, ddim_sample, ddpm_sample
from rectforge.models.mlp import TimeCondMLP
from rectforge.sampling.ode import NFECounter


def test_schedule_cosine_monotone():
    s = DiffusionSchedule(T=1000)
    assert s.ab.shape == (1001,)  # 含 t=0 哨兵
    ab = s.ab[1:]
    assert np.all(ab > 0) and np.all(ab <= 1.0)
    assert np.all(np.diff(ab) < 0)  # 单调下降


def test_schedule_endpoints():
    s = DiffusionSchedule(T=1000)
    assert s.ab[1] > 0.99  # t=1 几乎无噪
    assert s.ab[-1] < 0.05  # t=T 几乎纯噪


def _model():
    return TimeCondMLP(dim=2, hidden=16, layers=1, embed_dim=16, seed=0)


def test_ddpm_sample_shape_nfe():
    m = _model()
    s = DiffusionSchedule(T=100)
    c = NFECounter()
    out = ddpm_sample(m, m.params, s, 32, np.random.default_rng(0), c)
    assert out.shape == (32, 2)
    assert c.n == 100
    assert np.all(np.isfinite(out))


def test_ddim_sample_shape_nfe():
    m = _model()
    s = DiffusionSchedule(T=100)
    c = NFECounter()
    out = ddim_sample(m, m.params, s, 32, 20, np.random.default_rng(0), c)
    assert out.shape == (32, 2)
    assert c.n == 20


def test_ddim_deterministic_given_rng():
    m = _model()
    s = DiffusionSchedule(T=100)
    a = ddim_sample(m, m.params, s, 16, 10, np.random.default_rng(3), NFECounter())
    b = ddim_sample(m, m.params, s, 16, 10, np.random.default_rng(3), NFECounter())
    assert np.array_equal(a, b)


def test_ddim_direction_noise_to_data():
    """回归测试（爆炸 bug）：时间网格必须从噪声端走向数据端。

    训练后的模型 DDIM 采样必须有界且量级正确——方向反转会让
    x0 估计发散到 1e14 量级。
    """
    from rectforge.core.types import TrainConfig
    from rectforge.data.densities import make_density
    from rectforge.training.trainer import train_ddpm

    dgp = make_density("gauss2gauss")
    data = dgp.sample(np.random.default_rng(0), 1024)
    sched = DiffusionSchedule(T=200)
    cfg = TrainConfig(steps=150, hidden=32, layers=2, batch_size=128)
    m = _model()
    params, _ = train_ddpm(m, data, sched, cfg, np.random.default_rng(0))
    out = ddim_sample(m, params, sched, 256, 24, np.random.default_rng(1), NFECounter())
    assert np.all(np.isfinite(out))
    # 方向 bug 的爆炸量级是 1e14+；修复后小模型 t≈1 端 x0 估计仍有
    # 1/sqrt(ab[T]) 放大的重尾（DDIM 低 NFE 固有性质），阈值取实测 1.2e3 的 10 倍。
    assert np.abs(out).max() < 1e4
    assert out.std() < 1e4
