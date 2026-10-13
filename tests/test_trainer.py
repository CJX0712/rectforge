"""训练器：Adam 数学性质、lr 调度、收敛、E112 禁忌组合。"""
import numpy as np
import pytest

from rectforge.baselines.ddpm import DiffusionSchedule
from rectforge.core.errors import ConfigError
from rectforge.core.types import BenchmarkConfig, TrainConfig
from rectforge.data.densities import make_density
from rectforge.models.mlp import TimeCondMLP
from rectforge.training.trainer import Adam, lr_at, train_ddpm, train_velocity


# ---- Adam ------------------------------------------------------------
def test_adam_minimizes_quadratic():
    params = {"w": np.array([3.0, -2.0])}
    opt = Adam(params)
    for _ in range(500):
        g = 2.0 * params["w"]
        opt.step(params, {"w": g}, lr=0.05)
    assert np.abs(params["w"]).max() < 1e-2


def test_adam_bias_correction_first_step():
    """第一步步长应≈lr（偏差修正生效）。"""
    params = {"w": np.zeros(1)}
    opt = Adam(params)
    before = params["w"].copy()
    opt.step(params, {"w": np.array([1.0])}, lr=0.1)
    assert np.allclose(params["w"] - before, -0.1, atol=1e-6)


# ---- lr 调度 ---------------------------------------------------------
def _cfg(**kw):
    t = TrainConfig()
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def test_lr_at_warmup():
    t = _cfg(steps=100, lr=2e-3, warmup=10)
    assert lr_at(0, t) < lr_at(5, t) < t.lr  # 线性上升
    assert lr_at(9, t) == pytest.approx(t.lr)


def test_lr_at_cosine_decay():
    t = _cfg(steps=100, lr=2e-3, warmup=10, lr_min_ratio=0.05)
    mid = lr_at(55, t)
    end = lr_at(99, t)
    assert t.lr > mid > end > t.lr * t.lr_min_ratio * 0.9


# ---- train_velocity --------------------------------------------------
def _small_cfg(**kw):
    cfg = BenchmarkConfig(dgps=("gauss2gauss",), seeds=(0,), n_train=2048,
                          eval_samples=256, true_eval_samples=256, nfe_sweep=(12,))
    t = cfg.train
    t.steps, t.hidden, t.layers, t.batch_size = 400, 64, 2, 128
    for k, v in kw.items():
        setattr(t, k, v)
    return cfg


def test_train_velocity_converges_on_gauss2gauss():
    """金标准 DGP 上损失逼近条件方差下界。

    独立耦合的解析下界（按 t 平均 Var(v|x_t)）≈ 1.18；
    400 步小模型实测 ~1.5，断言显著低于初始损失即可（防训练回归）。
    """
    dgp = make_density("gauss2gauss")
    data = dgp.sample(np.random.default_rng(0), 2048)
    model = TimeCondMLP(dim=2, hidden=64, layers=2, embed_dim=16, seed=1000)
    _, hist = train_velocity(model, data, _small_cfg().train, np.random.default_rng(0))
    assert len(hist) == 400
    assert hist[-1] < 0.7 * hist[0]
    assert hist[-1] < 1.8  # 解析下界 1.18 的 1.5 倍以内


def test_train_velocity_deterministic():
    dgp = make_density("gauss2gauss")
    data = dgp.sample(np.random.default_rng(0), 1024)
    cfg = _small_cfg(steps=120).train
    m1 = TimeCondMLP(dim=2, hidden=32, layers=2, seed=1000)
    p1, _ = train_velocity(m1, data, cfg, np.random.default_rng(7))
    m2 = TimeCondMLP(dim=2, hidden=32, layers=2, seed=1000)
    p2, _ = train_velocity(m2, data, cfg, np.random.default_rng(7))
    for k in p1:
        assert np.array_equal(p1[k], p2[k])


def test_train_velocity_rejects_si_with_ot():
    cfg = _small_cfg(path="si", coupling="ot").train
    model = TimeCondMLP(dim=2, hidden=16, layers=1, seed=0)
    with pytest.raises(ConfigError, match="E112"):
        train_velocity(model, np.zeros((4, 2)), cfg, np.random.default_rng(0))


def test_train_velocity_si_with_independent_runs():
    dgp = make_density("gauss2gauss")
    data = dgp.sample(np.random.default_rng(0), 1024)
    cfg = _small_cfg(path="si", coupling="independent", steps=100).train
    model = TimeCondMLP(dim=2, hidden=32, layers=2, seed=0)
    _params, hist = train_velocity(model, data, cfg, np.random.default_rng(0))
    assert len(hist) == 100 and np.all(np.isfinite(hist))


# ---- train_ddpm ------------------------------------------------------
def test_train_ddpm_runs_and_loss_finite():
    dgp = make_density("gauss2gauss")
    data = dgp.sample(np.random.default_rng(0), 1024)
    cfg = _small_cfg(steps=100).train
    model = TimeCondMLP(dim=2, hidden=32, layers=2, seed=0)
    _params, hist = train_ddpm(
        model, data, DiffusionSchedule(T=200), cfg, np.random.default_rng(0)
    )
    assert len(hist) == 100
    assert np.all(np.isfinite(hist))
    assert hist[-1] < hist[0]
