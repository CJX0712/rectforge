"""金标准诊断：gmm8 的精确条件速度 E[v|x_t]（独立耦合、线性路径）有闭式解。

若训练充分，模型损失应收敛到 exact-optimal loss（条件方差下界），
且学到的速度场应与闭式场逐点一致。用于区分「训练欠拟合」与「采样/配置问题」。
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy.special import logsumexp

from rectforge.data.densities import make_density
from rectforge.models.mlp import TimeCondMLP


def exact_cond_velocity(x_t, t, means, stds, weights):
    """E[x1-x0 | x_t]，独立耦合、x0~GMM、x1~N(0,I)、x_t=(1-t)x0+t x1。"""
    means = np.asarray(means, float)
    stds = np.asarray(stds, float)
    k = means.shape[0]
    w = np.asarray(weights, float)
    s2 = stds[:, 0] ** 2  # (k,) 各分量方差（对角）
    # 责任度：p(x_t) 混合 N((1-t)m_k, ((1-t)^2 s2_k + t^2) I)
    var_t = (1.0 - t)[:, None] ** 2 * s2[None, :] + t[:, None] ** 2  # (B,k)
    d2 = ((x_t[:, None, :] - (1.0 - t)[:, None, None] * means[None, :, :]) ** 2).sum(-1)
    logr = np.log(w)[None, :] - 0.5 * (d2 / var_t + np.log(var_t).sum(-1, keepdims=True)
                                       + x_t.shape[1] * np.log(2 * np.pi))
    logr -= logsumexp(logr, axis=1, keepdims=True)
    r = np.exp(logr)  # (B,k)
    # 每分量后验：x0|x_t,k ~ N(mu_k, tau_k^2)，tau_k^2 = 1/(1/s2_k + (1-t)^2/t^2)
    tt = np.asarray(t, float)[:, None, None]  # (B,1,1)
    tau2 = 1.0 / (1.0 / s2[None, :, None] + (1.0 - tt) ** 2 / tt ** 2)  # (B,k,1)
    mu_post = tau2 * (
        means[None, :, :] / s2[None, :, None]
        + (1.0 - tt) * x_t[:, None, :] / tt ** 2
    )  # (B,k,d)
    ex0 = (r[:, :, None] * mu_post).sum(axis=1)  # (B,d)
    tt = np.asarray(t, float)[:, None]  # (B,1)
    ex1 = (x_t - (1.0 - tt) * ex0) / tt
    return ex1 - ex0


def main(steps=1600, hidden=96, layers=3, seed=0, n_train=8192):
    from rectforge.core.types import BenchmarkConfig
    from rectforge.pipeline.benchmark import train_models

    dgp = make_density("gmm8")
    cfg = BenchmarkConfig(dgps=("gmm8",), seeds=(seed,), n_train=n_train,
                          eval_samples=1024, true_eval_samples=1024, nfe_sweep=(48,))
    cfg.train.steps, cfg.train.hidden, cfg.train.layers = steps, hidden, layers

    rng = np.random.default_rng(seed)
    data = dgp.sample(rng, n_train)
    models = train_models(dgp, data, cfg, train_seed=seed)
    model, params = models["rectfuse"]["model"], models["rectfuse"]["params"]

    # ---- 1) 精确最优损失 vs 模型损失 ----
    rng2 = np.random.default_rng(12345)
    x0 = dgp.sample(rng2, 8192)
    x1 = rng2.normal(size=x0.shape)
    t = rng2.uniform(0.02, 0.98, size=8192)
    tt = t[:, None]
    xt = (1 - tt) * x0 + tt * x1
    v = x1 - x0
    v_star = exact_cond_velocity(xt, t, dgp.means, dgp.stds, dgp.weights)
    loss_star = float(np.mean((v - v_star) ** 2))
    # 模型在同一 (x_t,t) 上的损失（逐 batch 防内存峰值）
    preds = []
    for i in range(0, len(xt), 2048):
        preds.append(model.predict(params, xt[i:i + 2048], t[i:i + 2048]))
    pred = np.concatenate(preds)
    loss_model = float(np.mean((v - pred) ** 2))
    print(f"[loss] exact-optimal E||v-E[v|xt]||^2/d = {loss_star:.5f}")
    print(f"[loss] trained model                          = {loss_model:.5f}")
    print(f"[loss] ratio = {loss_model / loss_star:.3f}")

    # ---- 2) 学到的速度场 vs 闭式场（网格 + 按 t 分层） ----
    g = np.linspace(-3.6, 3.6, 41)
    XX, YY = np.meshgrid(g, g)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    for tv in (0.1, 0.3, 0.5, 0.7, 0.9):
        vs = exact_cond_velocity(grid, np.full(len(grid), tv), dgp.means, dgp.stds, dgp.weights)
        vm = model.predict(params, grid, np.full(len(grid), tv))
        rel = np.linalg.norm(vm - vs) / (np.linalg.norm(vs) + 1e-12)
        print(f"[field] t={tv:.1f}: rel L2 err learned vs exact = {rel:.4f}")

    # ---- 2b) 按 t 分层的损失（训练分布 vs 均匀分布） ----
    print("[loss-by-t]  t-bin   n_train(logit_normal)   model/exact ratio")
    z = np.random.default_rng(3).normal(size=200000)
    t_ln = 1.0 / (1.0 + np.exp(-z))
    t_un = np.random.default_rng(4).random(200000)
    for lo, hi in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]:
        m_ln = ((t_ln >= lo) & (t_ln < hi)).mean()
        rng3 = np.random.default_rng(100)
        x0b = dgp.sample(rng3, 4096)
        x1b = rng3.normal(size=x0b.shape)
        tb = rng3.uniform(lo, hi, size=4096)
        ttb = tb[:, None]
        xtb = (1 - ttb) * x0b + ttb * x1b
        vb = x1b - x0b
        vsb = exact_cond_velocity(xtb, tb, dgp.means, dgp.stds, dgp.weights)
        pb = []
        for i in range(0, len(xtb), 2048):
            pb.append(model.predict(params, xtb[i:i + 2048], tb[i:i + 2048]))
        pb = np.concatenate(pb)
        lb = float(np.mean((vb - pb) ** 2))
        ls = float(np.mean((vb - vsb) ** 2))
        print(f"           [{lo:.1f},{hi:.1f})   {m_ln:.3f}                "
              f"{lb:.4f}/{ls:.4f} = {lb / ls:.3f}")

    # ---- 3) 输运矩：学到的 ODE 把 N(0,I) 推到哪 ----
    from rectforge.pipeline.benchmark import sample_method
    cfg2 = BenchmarkConfig(dgps=("gmm8",), seeds=(seed,), n_train=n_train,
                           eval_samples=4096, true_eval_samples=4096, nfe_sweep=(48,))
    cfg2.train.steps, cfg2.train.hidden, cfg2.train.layers = steps, hidden, layers
    data2 = dgp.sample(np.random.default_rng(seed), n_train)
    corpus = data2[np.random.default_rng(seed + 1_000_000).integers(0, len(data2), 2048)]
    models2 = train_models(dgp, data2, cfg2, train_seed=seed)
    rng3 = np.random.default_rng(seed + 500_000)
    samples, nfe = sample_method("rectfuse", models2, corpus, cfg2, 4096, rng3, 48)
    d2 = ((samples[:, None, :] - dgp.means[None, :, :]) ** 2).sum(-1)
    print(f"[sample] modes hit {int((d2 < 0.36).any(axis=0).sum())}/8, "
          f"coverage={float((d2.min(axis=1) < 0.36).mean()):.3f}, nfe={nfe}")
    print(f"[sample] sample mean={samples.mean(axis=0).round(3)}, "
          f"true mean={dgp.means.mean(axis=0).round(3)}, "
          f"sample std={samples.std(axis=0).round(3)}")


if __name__ == "__main__":
    import sys
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1600
    main(steps=steps)
