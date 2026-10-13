"""耦合：independent / minibatch-OT（Tong et al. 2023）。

OT 耦合用 scipy.optimize.linear_sum_assignment 精确求解
min ||x0_i - x1_{pi(i)}||^2（批量版）。不可用时降级 independent。
"""
from __future__ import annotations

import numpy as np


def available_scipy() -> bool:
    try:
        import scipy.optimize  # noqa: F401

        return True
    except Exception:
        return False


def independent_pair(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    return x1


def _squared_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = (a**2).sum(axis=1)[:, None]
    b2 = (b**2).sum(axis=1)[None, :]
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    return np.maximum(d2, 0.0)


def ot_pair(x0: np.ndarray, x1: np.ndarray, cost_scale: float = 1.0) -> np.ndarray:
    """把 x1 重排列成 x0 的 OT 配对（返回重排后的 x1）。"""
    from scipy.optimize import linear_sum_assignment

    cost = cost_scale * _squared_dists(x0, x1)
    _row, col = linear_sum_assignment(cost)
    return x1[col]


def make_coupling(kind: str):
    if kind == "independent":
        return independent_pair
    if kind == "ot":
        if not available_scipy():
            return independent_pair
        return ot_pair
    raise ValueError(f"unknown coupling {kind!r}")
