"""时间条件 MLP：v_theta(x, t)，纯 numpy 解析反向传播。

设计要点：
- 时间嵌入用正弦特征（DDPM/FM 通用惯例），t in [0,1]（1=噪声端）。
- 最后一层小初始化（final_init_scale）：训练初期 v≈0，ODE 采样稳定。
- 梯度可中心差分校验（tests/test_models.py 硬门禁 ~1e-7）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rectforge.core.errors import ModelError


def sinusoidal_embedding(t: np.ndarray, dim: int) -> np.ndarray:
    """t: (n,) in [0,1] -> (n, dim) 正弦/余弦特征。"""
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    half = dim // 2
    freqs = 2.0 * np.pi * np.logspace(0.0, 8.0, half, base=2.0)
    ang = t[:, None] * freqs[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


def _silu(z: np.ndarray) -> np.ndarray:
    return z / (1.0 + np.exp(-z))


def _silu_grad(z: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-z))
    return s + z * s * (1.0 - s)


@dataclass
class MLPWeights:
    weights: list
    biases: list


class TimeCondMLP:
    """[x, sinEmb(t)] -> MLP(SiLU) -> out_dim。params 为 dict（Adam 直接操作）。"""

    def __init__(
        self,
        dim: int,
        out_dim: int | None = None,
        hidden: int = 96,
        layers: int = 3,
        embed_dim: int = 32,
        final_init_scale: float = 0.01,
        seed: int = 0,
    ):
        self.dim = int(dim)
        self.out_dim = int(out_dim if out_dim is not None else dim)
        self.embed_dim = int(embed_dim)
        rng = np.random.default_rng(seed)
        dims = [self.dim + self.embed_dim] + [int(hidden)] * int(layers) + [self.out_dim]
        self.sizes = dims
        weights, biases = [], []
        n_layers = len(dims) - 1
        for i in range(n_layers):
            fan_in, fan_out = dims[i], dims[i + 1]
            # He init（配 SiLU）；输出层小初始化（训练初期 v≈0，ODE 采样稳定）
            scale = (
                np.sqrt(2.0 / fan_in)
                if i < n_layers - 1
                else final_init_scale * np.sqrt(1.0 / fan_in)
            )
            weights.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            biases.append(np.zeros(fan_out))
        self.params: dict[str, np.ndarray] = {}
        for i, (w, b) in enumerate(zip(weights, biases, strict=True)):
            self.params[f"W{i}"] = w
            self.params[f"b{i}"] = b

    # ---- 前向 -------------------------------------------------------
    def predict(self, params: dict, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        out, _ = self._forward_with(params, x, t)
        return out

    def _forward_with(self, params: dict, x: np.ndarray, t: np.ndarray):
        emb = sinusoidal_embedding(t, self.embed_dim)
        h = np.concatenate([x, emb], axis=1)
        cache = [h]
        n_layers = len(self.sizes) - 1
        for i in range(n_layers):
            z = h @ params[f"W{i}"] + params[f"b{i}"]
            h = _silu(z) if i < n_layers - 1 else z
            cache.append(h)
        return cache[-1], cache

    # ---- 损失与梯度 -------------------------------------------------
    def loss_and_grad(self, params: dict, x: np.ndarray, t: np.ndarray, target: np.ndarray):
        """MSE(params(x,t), target)。返回 (loss, grads dict)。"""
        x = np.asarray(x, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        target = np.asarray(target, dtype=np.float64)
        if x.shape[0] != t.shape[0] or x.shape[0] != target.shape[0]:
            raise ModelError("E300: batch size mismatch among x/t/target")
        out, cache = self._forward_with(params, x, t)
        diff = out - target
        n, d_out = diff.shape
        loss = float(np.mean(diff**2))
        grad_out = 2.0 * diff / (n * d_out)  # dL/da_{n_layers}（输出层线性 => = dL/dz）
        grads: dict[str, np.ndarray] = {}
        n_layers = len(self.sizes) - 1
        for i in range(n_layers - 1, -1, -1):
            a_i = cache[i]  # 第 i 层输入
            if i < n_layers - 1:
                # 隐藏层：a_{i+1} = silu(z_i) => dL/dz_i = dL/da_{i+1} * silu'(z_i)
                z_i = a_i @ params[f"W{i}"] + params[f"b{i}"]
                g = grad_out * _silu_grad(z_i)
            else:
                g = grad_out  # 输出层线性，dL/dz = dL/da
            grads[f"W{i}"] = a_i.T @ g
            grads[f"b{i}"] = g.sum(axis=0)
            if i > 0:
                grad_out = g @ params[f"W{i}"].T  # dL/da_i
        return loss, grads

    def num_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))
