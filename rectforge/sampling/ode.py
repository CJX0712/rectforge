"""ODE 采样器：Euler / RK4，带 NFE 记账；SI 路径的精确概率流修正。

正确性基础：
- gamma == 0：dx/dt = u_theta(x,t)（u_theta ~ b_t）即概率流 ODE，精确。
- gamma > 0（仅 independent 耦合）：概率流 ODE 为
    dx/dt = u_theta(x,t) - 0.5*gamma(t)^2 * grad log p_t(x)
  其中 p_t 是高斯混合：中心 m_i=(1-t)x0^i，方差 s^2 = t^2 + gamma^2。
  修正项 0.5*gamma^2*grad log p_t = 0.5*(gamma^2/s^2) * sum_i w_i (m_i - x)，
  w_i = softmax(-||x-m_i||^2 / (2 s^2))。gamma->0 时该项有界（gamma^2/s^2->1）。
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from rectforge.core.errors import SamplingError


class NFECounter:
    """网络调用次数（batch-of-1 口径统一：每次 u_fn 调用计 1）。"""

    __slots__ = ("n",)

    def __init__(self):
        self.n = 0

    def bump(self, k: int = 1) -> None:
        self.n += k


def euler_integrate(u_fn, x0: np.ndarray, steps: int, counter: NFECounter) -> np.ndarray:
    ts = np.linspace(1.0, 0.0, steps + 1)
    x = np.asarray(x0, dtype=np.float64).copy()
    for i in range(steps):
        h = ts[i + 1] - ts[i]
        x = x + h * u_fn(x, float(ts[i]))
        counter.bump()
    return x


def rk4_integrate(u_fn, x0: np.ndarray, steps: int, counter: NFECounter) -> np.ndarray:
    ts = np.linspace(1.0, 0.0, steps + 1)
    x = np.asarray(x0, dtype=np.float64).copy()
    for i in range(steps):
        h = ts[i + 1] - ts[i]
        t0 = float(ts[i])
        k1 = u_fn(x, t0)
        k2 = u_fn(x + 0.5 * h * k1, t0 + 0.5 * h)
        k3 = u_fn(x + 0.5 * h * k2, t0 + 0.5 * h)
        k4 = u_fn(x + h * k3, t0 + h)
        x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        counter.bump(4)
    return x


def integrate_ode(u_fn, x0: np.ndarray, steps: int, method: str, counter: NFECounter) -> np.ndarray:
    if method == "euler":
        return euler_integrate(u_fn, x0, steps, counter)
    if method == "rk4":
        return rk4_integrate(u_fn, x0, steps, counter)
    raise SamplingError(f"E500: unknown ode method {method!r}")


def pf_velocity_fn(model, params, path, data: np.ndarray):
    """构造概率流速度场 u_pf(x,t)。

    gamma==0：直接返回 u_theta。
    gamma>0（independent 耦合）：加 -0.5*gamma^2*grad log p_t 闭式修正。
    """
    sigma = getattr(path, "sigma", 0.0)

    def u_fn(x: np.ndarray, t: float) -> np.ndarray:
        batch_t = np.full(x.shape[0], t)
        u = model.predict(params, x, batch_t)
        if sigma <= 0.0:
            return u
        g = sigma * t * (1.0 - t)
        # s2 下限防 0/0：RK4 中间级可能恰好取 t=0（g=0 且 t^2=0）
        s2 = max(t * t + g * g, 1e-12)
        m = (1.0 - t) * data  # (n,d) 混合中心
        # w_i = softmax(-||x-m_i||^2/(2 s2))，修正 = 0.5*(g^2/s2)*sum_i w_i (m_i-x)
        d2 = ((x[:, None, :] - m[None, :, :]) ** 2).sum(axis=2)  # (B,n)
        logits = -d2 / (2.0 * s2)
        w = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        correction = 0.5 * (g * g / s2) * (w @ m - x)
        return u - correction

    return u_fn
