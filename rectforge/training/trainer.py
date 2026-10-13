"""训练器：速度场（FM 家族）+ DDPM，纯 numpy Adam。

确定性约定：所有随机性来自显式传入的 rng；数据子批用 rng.integers 取下标
（有放回抽样，避免 epoch 边界效应）；BLAS 线程由 core.seed 钉 1。
"""
from __future__ import annotations

import numpy as np

from rectforge.core.types import TrainConfig
from rectforge.paths.coupling import make_coupling
from rectforge.paths.interpolants import make_path, sample_time


class Adam:
    """对 params dict 逐键 Adam（与 MLP 参数结构一一对应）。"""

    def __init__(self, params: dict, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict, lr: float) -> None:
        self.t += 1
        bc1 = 1.0 - self.b1**self.t
        bc2 = 1.0 - self.b2**self.t
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * grads[k] ** 2
            params[k] -= lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + self.eps)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """线性 warmup + 余弦衰减到 lr_min_ratio*lr。"""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / max(cfg.warmup, 1)
    progress = (step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + np.cos(np.pi * progress))
    return cfg.lr * (cfg.lr_min_ratio + (1.0 - cfg.lr_min_ratio) * cos)


def _global_norm(grads: dict) -> float:
    return float(np.sqrt(sum(float(np.sum(g**2)) for g in grads.values())))


def train_velocity(model, data: np.ndarray, cfg: TrainConfig, rng: np.random.Generator):
    """训练速度场 v_theta(x,t) ~ b_t。返回 (params, history)。"""
    if cfg.path == "si" and cfg.coupling == "ot":
        from rectforge.core.errors import ConfigError

        raise ConfigError(
            "E112: si path with ot coupling is not supported "
            "(exact PF-ODE correction requires independent coupling)"
        )
    path = make_path(cfg.path, si_sigma=cfg.si_sigma)
    couple = make_coupling(cfg.coupling)
    params = {k: v.copy() for k, v in model.params.items()}
    opt = Adam(params)
    n = data.shape[0]
    dim = model.dim
    history: list[float] = []
    for step in range(cfg.steps):
        idx = rng.integers(0, n, cfg.batch_size)
        x0 = data[idx]
        x1 = rng.normal(size=(cfg.batch_size, dim))
        if cfg.coupling == "ot" and step % max(cfg.ot_refresh, 1) == 0:
            x1 = couple(x0, x1, cost_scale=cfg.ot_cost_scale)
        t = sample_time(cfg.time_sampling, cfg.batch_size, rng)
        xt, v = path.conditional(x0, x1, t, rng)
        loss, grads = model.loss_and_grad(params, xt, t, v)
        norm = _global_norm(grads)
        if norm > cfg.grad_clip:
            scale = cfg.grad_clip / (norm + 1e-12)
            grads = {k: g * scale for k, g in grads.items()}
        opt.step(params, grads, lr_at(step, cfg))
        history.append(loss)
    return params, history


def train_ddpm(model, data: np.ndarray, schedule, cfg: TrainConfig, rng: np.random.Generator):
    """训练 eps_theta(x,t)（Ho et al. 2020）。返回 (params, history)。"""
    params = {k: v.copy() for k, v in model.params.items()}
    opt = Adam(params)
    n = data.shape[0]
    ab = schedule.ab
    history: list[float] = []
    for step in range(cfg.steps):
        idx = rng.integers(0, n, cfg.batch_size)
        x0 = data[idx]
        t_idx = rng.integers(1, schedule.T + 1, size=cfg.batch_size)
        eps = rng.normal(size=x0.shape)
        ab_t = ab[t_idx][:, None]
        xt = np.sqrt(ab_t) * x0 + np.sqrt(1.0 - ab_t) * eps
        t_cont = t_idx.astype(np.float64) / schedule.T
        loss, grads = model.loss_and_grad(params, xt, t_cont, eps)
        norm = _global_norm(grads)
        if norm > cfg.grad_clip:
            scale = cfg.grad_clip / (norm + 1e-12)
            grads = {k: g * scale for k, g in grads.items()}
        opt.step(params, grads, lr_at(step, cfg))
        history.append(loss)
    return params, history
