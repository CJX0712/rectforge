"""梯度校验 + 训练诊断：确认 mlp.py 反向传播正确、修复后训练质量达标。"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"

import time

import numpy as np

from rectforge.models.mlp import TimeCondMLP


def grad_check(model, rng, eps=1e-6, batch=64):
    x = rng.normal(size=(batch, model.dim))
    t = rng.uniform(0.0, 1.0, size=batch)
    target = rng.normal(size=(batch, model.out_dim))
    _, grads = model.loss_and_grad(model.params, x, t, target)
    worst = 0.0
    worst_info = ""
    worst_abs = 0.0
    for k in sorted(model.params):
        p = model.params[k]
        num = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = p[idx]
            p[idx] = orig + eps
            lp, _ = model.loss_and_grad(model.params, x, t, target)
            p[idx] = orig - eps
            lm, _ = model.loss_and_grad(model.params, x, t, target)
            p[idx] = orig
            num[idx] = (lp - lm) / (2.0 * eps)
            it.iternext()
        ana = grads[k]
        err = np.abs(ana - num)
        denom = np.maximum(1e-8, np.abs(ana) + np.abs(num))
        rel = err / denom
        i = int(np.argmax(rel))
        if rel.flat[i] > worst:
            worst = float(rel.flat[i])
            worst_info = (f"{k}[{np.unravel_index(i, rel.shape)}] "
                          f"ana={ana.flat[i]:.3e} num={num.flat[i]:.3e}")
        worst_abs = max(worst_abs, float(err.max()))
    return worst, worst_info, worst_abs


def train_diagnostic(steps, hidden, layers, seed=0, n_train=4096, n_test=4096):
    """rectfuse 旗舰配置小规模训练，返回 (losses, mode_coverage, KL, 秒数)。"""
    from rectforge.core.types import BenchmarkConfig, TrainConfig
    from rectforge.data.densities import make_density
    from rectforge.pipeline.benchmark import (
        flagship_train_cfg, train_models, sample_method, evaluate_point,
    )

    cfg = BenchmarkConfig(dgps=("gmm8",), seeds=(seed,), n_train=n_train,
                          eval_samples=n_test, true_eval_samples=n_test,
                          nfe_sweep=(48,))
    t = cfg.train
    t.steps, t.hidden, t.layers = steps, hidden, layers

    density = make_density("gmm8")
    rng_data = np.random.default_rng(seed)
    data = density.sample(rng_data, n_train)
    x_true = density.sample(np.random.default_rng(seed + 900_000), n_test)
    corpus = data[np.random.default_rng(seed + 1_000_000).integers(0, len(data), 2048)]

    t0 = time.perf_counter()
    models = train_models(density, data, cfg, train_seed=seed)
    train_sec = time.perf_counter() - t0

    hist = models["rectfuse"]["final_loss"]
    rng = np.random.default_rng(seed + 500_000)
    samples, nfe = sample_method("rectfuse", models, corpus, cfg, n_test, rng, 48)
    kl, mm, w2 = evaluate_point(density, x_true, samples)

    # 模态覆盖：距最近模态中心 < 0.6
    centers = density.means
    d2 = ((samples[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    frac = float((d2.min(axis=1) < 0.36).mean())
    modes_hit = int((d2 < 0.36).any(axis=0).sum())
    return (models["rectfuse"]["final_loss"], frac, modes_hit, centers.shape[0],
            kl, mm, w2, train_sec, nfe)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("== gradient checks (central diff, eps=1e-6) ==")
    for hidden, layers in [(48, 3), (32, 1), (64, 4), (16, 2)]:
        m = TimeCondMLP(dim=2, out_dim=2, hidden=hidden, layers=layers, embed_dim=16, seed=7)
        worst, info, worst_abs = grad_check(m, rng)
        print(f"  hidden={hidden:3d} layers={layers}: worst rel={worst:.3e} "
              f"worst abs={worst_abs:.3e}  ({info})")

    print("\n== training diagnostic (gmm8, rectfuse flagship cfg) ==")
    for steps, hidden, layers in [(400, 96, 3), (800, 96, 3), (1600, 96, 3)]:
        fl, frac, hit, k, kl, mm, w2, sec, nfe = train_diagnostic(steps, hidden, layers)
        print(f"  steps={steps:5d} hidden={hidden}: final_loss={fl:.4f} | "
              f"modes {hit}/{k} cover={frac:.3f} | KL={kl:.4f} MMD2={mm:.5f} W2={w2:.4f} "
              f"| train(3 models)={sec:.1f}s nfe={nfe}")
