"""DDPM / DDIM 基线（Ho et al. 2020；Song et al. 2021）。

- 噪声日程：cosine schedule（Nichol & Dhariwal 2021），T=1000。
-  ancestral DDPM 采样（1000 步，参考口径，不计入等预算对比）。
- DDIM 确定性采样（eta=0），步数可调 —— 等 NFE 预算对比的强基线。
"""
from __future__ import annotations

import numpy as np


class DiffusionSchedule:
    def __init__(self, T: int = 1000, s: float = 0.008):
        self.T = int(T)
        t = np.arange(self.T + 1, dtype=np.float64) / self.T
        f = np.cos((t + s) / (1.0 + s) * np.pi / 2.0) ** 2
        raw = f / f[0]  # ab[k] = alpha_bar after k steps, raw[0]=1
        # Nichol & Dhariwal 2021 标准做法：从截断 beta 重建 ab。
        # cosine 的 raw[T]=cos(pi/2)^2=0 精确为零 -> x0 估计除零发散。
        beta = np.clip(1.0 - raw[1:] / raw[:-1], 1e-4, 0.999)
        self.ab = np.concatenate([[1.0], np.cumprod(1.0 - beta)])
        self.alpha = self.ab[1:] / self.ab[:-1]
        self.beta = beta
        # posterior variance（Ho eq.7，数值截断防 0/0）
        self.post_var = np.clip(
            self.beta * (1.0 - self.ab[:-1]) / np.maximum(1.0 - self.ab[1:], 1e-12),
            1e-12,
            None,
        )


class NFECounter:
    __slots__ = ("n",)

    def __init__(self):
        self.n = 0

    def bump(self, k: int = 1) -> None:
        self.n += k


def ddpm_sample(model, params, schedule: DiffusionSchedule, n: int, rng, counter: NFECounter):
    """ancestral DDPM 采样，T 步。"""
    T = schedule.T
    ab = schedule.ab
    x = rng.normal(size=(n, model.dim))
    for t in range(T - 1, -1, -1):
        eps_hat = model.predict(params, x, np.full(n, (t + 1) / T))
        counter.bump()
        ab_t = ab[t + 1]
        ab_prev = ab[t]
        x0 = (x - np.sqrt(1.0 - ab_t) * eps_hat) / np.sqrt(ab_t)
        alpha_t = schedule.alpha[t]
        beta_t = schedule.beta[t]
        coef1 = np.sqrt(ab_prev) * beta_t / max(1.0 - ab_t, 1e-12)
        coef2 = np.sqrt(alpha_t) * (1.0 - ab_prev) / max(1.0 - ab_t, 1e-12)
        mean = coef1 * x0 + coef2 * x
        x = mean + (np.sqrt(schedule.post_var[t]) * rng.normal(size=x.shape) if t > 0 else 0.0)
    return x


def ddim_sample(model, params, schedule: DiffusionSchedule, n: int, steps: int, rng, counter: NFECounter):
    """DDIM 确定性采样（eta=0），steps 步。

    时间网格从噪声端 t=T-1 走向数据端 t=0（与训练约定 t=1 噪声一致）；
    最后一步 ab_prev=ab[0]=1 => x=x0，纯净落地。
    """
    T = schedule.T
    ab = schedule.ab
    ts = np.linspace(T - 1, 0, steps).astype(int)
    x = rng.normal(size=(n, model.dim))
    for i in range(steps):
        t_cur = int(ts[i])
        t_prev = int(ts[i + 1]) if i + 1 < steps else -1
        eps_hat = model.predict(params, x, np.full(n, (t_cur + 1) / T))
        counter.bump()
        ab_cur = ab[t_cur + 1]
        ab_prev = ab[t_prev + 1] if t_prev >= 0 else 1.0
        x0 = (x - np.sqrt(1.0 - ab_cur) * eps_hat) / np.sqrt(ab_cur)
        x = np.sqrt(ab_prev) * x0 + np.sqrt(1.0 - ab_prev) * eps_hat
    return x
