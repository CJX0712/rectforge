"""模型：时间条件 MLP——梯度检验硬门禁 + 前向形状 + 嵌入有界。"""
import numpy as np
import pytest

from rectforge.core.errors import ModelError
from rectforge.models.mlp import TimeCondMLP, sinusoidal_embedding


def _num_grad(model, x, t, target, eps=1e-6):
    """中心差分（ loss 对每个参数分量）。返回 {key: array}。"""
    f = lambda: model.loss_and_grad(model.params, x, t, target)[0]
    num = {}
    for k, p in model.params.items():
        g = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = p[idx]
            p[idx] = orig + eps
            lp = f()
            p[idx] = orig - eps
            lm = f()
            p[idx] = orig
            g[idx] = (lp - lm) / (2.0 * eps)
            it.iternext()
        num[k] = g
    return num


def test_gradient_check_small(rng):
    """硬金标准：解析反向 vs 中心差分。断言「最大绝对误差」（FD 噪声口径）。"""
    model = TimeCondMLP(dim=2, out_dim=2, hidden=16, layers=2, embed_dim=16, seed=3)
    x = rng.normal(size=(32, 2))
    t = rng.uniform(0, 1, size=32)
    target = rng.normal(size=(32, 2))
    _, grads = model.loss_and_grad(model.params, x, t, target)
    num = _num_grad(model, x, t, target)
    for k in num:
        assert np.abs(grads[k] - num[k]).max() < 1e-6, f"grad mismatch at {k}"


def test_gradient_check_deeper(rng):
    model = TimeCondMLP(dim=3, out_dim=2, hidden=24, layers=4, embed_dim=16, seed=5)
    x = rng.normal(size=(24, 3))
    t = rng.uniform(0, 1, size=24)
    target = rng.normal(size=(24, 2))
    _, grads = model.loss_and_grad(model.params, x, t, target)
    num = _num_grad(model, x, t, target)
    for k in num:
        assert np.abs(grads[k] - num[k]).max() < 1e-6


def test_output_layer_grad_exact(rng):
    """输出层小初始化下最后层梯度应精确（<1e-8）。"""
    model = TimeCondMLP(dim=2, hidden=8, layers=1, embed_dim=16, seed=1)
    x = rng.normal(size=(16, 2))
    t = rng.uniform(0, 1, size=16)
    target = rng.normal(size=(16, 2))
    _, grads = model.loss_and_grad(model.params, x, t, target)
    num = _num_grad(model, x, t, target)
    assert np.abs(grads["W0"] - num["W0"]).max() < 1e-8


def test_forward_shapes(rng):
    model = TimeCondMLP(dim=2, hidden=32, layers=2, embed_dim=16, seed=0)
    out = model.predict(model.params, rng.normal(size=(7, 2)), np.full(7, 0.5))
    assert out.shape == (7, 2)
    assert model.num_params() > 0


def test_batch_mismatch_raises(rng):
    model = TimeCondMLP(dim=2, hidden=8, layers=1, seed=0)
    with pytest.raises(ModelError, match="E300"):
        model.loss_and_grad(model.params, rng.normal(size=(5, 2)), np.full(6, 0.5),
                            rng.normal(size=(5, 2)))


def test_sinusoidal_embedding_bounds():
    e = sinusoidal_embedding(np.linspace(0, 1, 33), dim=16)
    assert e.shape == (33, 16)
    assert np.all(e >= -1.0 - 1e-12) and np.all(e <= 1.0 + 1e-12)
    # t=0 处 sin 项全 0、cos 项全 1
    assert np.allclose(e[0, :8], 0.0) and np.allclose(e[0, 8:], 1.0)
