"""跨模块接口契约（Protocol）。"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Density(Protocol):
    """合成密度：解析 log p + 可复现采样器。"""

    name: str
    dim: int
    bounds: tuple[float, float]

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray: ...

    def log_prob(self, x: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class VelocityField(Protocol):
    """速度场 v_theta(x, t)：FM/DDPM 统一接口。"""

    dim: int

    def predict(self, params, x: np.ndarray, t: np.ndarray) -> np.ndarray: ...

    def loss_and_grad(self, params, x: np.ndarray, t: np.ndarray, target: np.ndarray): ...


@runtime_checkable
class Interpolant(Protocol):
    """概率路径：给定 (x0, x1, t) 产出 (x_t, 条件速度目标)。"""

    name: str

    def conditional(
        self, x0: np.ndarray, x1: np.ndarray, t: np.ndarray, rng
    ) -> tuple[np.ndarray, np.ndarray]: ...
