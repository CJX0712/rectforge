"""概率路径（插值算子）：给定 (x0, x1, t) 产出 (x_t, 条件速度目标)。

三条路径全部来自已发表工作：
- LinearPath：rectified flow（Liu et al., ICLR 2023）
- VPPath：cosine VP 路径的 conditional flow matching（Lipman et al., ICLR 2023）
- SIPath：stochastic interpolant（Albergo et al., ICLR 2023），gamma(t)=sigma*t*(1-t)

数学事实（采样正确性的基础）：
- gamma == 0 时插值过程是确定性的，dx/dt = b_t(x)（b_t = E[v_cond|x_t]）
  即概率流 ODE，ODE 采样精确——flagship 走这条路。
- gamma > 0 时概率流 ODE 需要额外分数修正项 -0.5*gamma^2*grad log p_t；
  该项对 independent 耦合有闭式（p_t 是以 (1-t)x0^i 为中心、方差 t^2+gamma^2
  的高斯混合），见 sampling/ode.py 的 pf_velocity_fn。
"""
from __future__ import annotations

import numpy as np


def sample_time(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """时间采样：uniform（均匀）或 logit_normal（Esser 2024, SD3 同款）。"""
    if kind == "uniform":
        return rng.random(n)
    if kind == "logit_normal":
        z = rng.normal(size=n)
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(f"unknown time sampling {kind!r}")


class LinearPath:
    """x_t = (1-t)x0 + t x1, v = x1 - x0。"""

    name = "linear"

    def conditional(self, x0, x1, t, rng):
        tt = np.asarray(t, dtype=np.float64)[:, None]
        xt = (1.0 - tt) * x0 + tt * x1
        v = x1 - x0
        return xt, v


class VPPath:
    """x_t = cos(pi t/2) x0 + sin(pi t/2) x1（cosine VP 路径）。"""

    name = "vp"

    def conditional(self, x0, x1, t, rng):
        tt = np.asarray(t, dtype=np.float64)[:, None]
        a = np.cos(np.pi * tt / 2.0)
        s = np.sin(np.pi * tt / 2.0)
        da = -(np.pi / 2.0) * np.sin(np.pi * tt / 2.0)
        ds = (np.pi / 2.0) * np.cos(np.pi * tt / 2.0)
        xt = a * x0 + s * x1
        v = da * x0 + ds * x1
        return xt, v


class SIPath:
    """x_t = (1-t)x0 + t x1 + gamma(t) eps, gamma = sigma*t*(1-t)。"""

    name = "si"

    def __init__(self, sigma: float = 0.25):
        if sigma < 0:
            raise ValueError("si_sigma must be >= 0")
        self.sigma = float(sigma)

    def gamma(self, t):
        t = np.asarray(t, dtype=np.float64)
        return self.sigma * t * (1.0 - t)

    def gamma_prime(self, t):
        t = np.asarray(t, dtype=np.float64)
        return self.sigma * (1.0 - 2.0 * t)

    def conditional(self, x0, x1, t, rng):
        tt = np.asarray(t, dtype=np.float64)[:, None]
        g = self.gamma(tt.reshape(-1))[:, None]
        gp = self.gamma_prime(tt.reshape(-1))[:, None]
        eps = rng.normal(size=x0.shape)
        xt = (1.0 - tt) * x0 + tt * x1 + g * eps
        v = x1 - x0 + gp * eps
        return xt, v


def make_path(name: str, si_sigma: float = 0.25):
    if name == "linear":
        return LinearPath()
    if name == "vp":
        return VPPath()
    if name == "si":
        return SIPath(sigma=si_sigma)
    raise ValueError(f"unknown path {name!r}")
