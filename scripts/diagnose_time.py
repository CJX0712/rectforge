"""时间采样扫描：logit_normal(std=1) vs uniform vs logit_normal(std=0.5)。

诊断结论（上一轮）：模型在中段 t 已近最优（ratio~1.01），但 t<0.1 / t>0.9
严重欠拟合（ratio 1.67 / 1.28）——logit-normal(std=1) 只给两端各 1.4% 训练质量，
而 t≈0 的速度场决定 ODE 终点是否落进数据流形。本脚本量化各方案。
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy.special import logsumexp

from rectforge.data.densities import make_density
from rectforge.core.types import BenchmarkConfig
from rectforge.pipeline.benchmark import train_models, sample_method, evaluate_point
from rectforge.models.mlp import TimeCondMLP
from rectforge.training.trainer import train_velocity, Adam, lr_at
from rectforge.paths.interpolants import make_path
from rectforge.paths.coupling import make_coupling


def exact_cond_velocity(x_t, t, means, stds, weights):
    means = np.asarray(means, float); stds = np.asarray(stds, float)
    w = np.asarray(weights, float); s2 = stds[:, 0] ** 2
    tv = np.asarray(t, float)
    tt = tv[:, None]            # (B,1)
    tt3 = tv[:, None, None]     # (B,1,1)
    var_t = (1.0 - tt) ** 2 * s2[None, :] + tt ** 2                      # (B,k)
    d2 = ((x_t[:, None, :] - (1.0 - tt3) * means[None, :, :]) ** 2).sum(-1)
    logr = np.log(w)[None, :] - 0.5 * (d2 / var_t + np.log(var_t).sum(-1, keepdims=True)
                                       + x_t.shape[1] * np.log(2 * np.pi))
    logr -= logsumexp(logr, axis=1, keepdims=True)
    r = np.exp(logr)                                                     # (B,k)
    tau2 = 1.0 / (1.0 / s2[None, :, None] + (1.0 - tt3) ** 2 / tt3 ** 2)  # (B,k,1)
    mu_post = tau2 * (means[None, :, :] / s2[None, :, None]
                      + (1.0 - tt3) * x_t[:, None, :] / tt3 ** 2)        # (B,k,d)
    ex0 = (r[:, :, None] * mu_post).sum(axis=1)                          # (B,d)
    ex1 = (x_t - (1.0 - tt) * ex0) / tt
    return ex1 - ex0


def _sample_time(kind, n, rng):
    if kind == "uniform":
        return rng.random(n)
    if kind == "logit_normal":
        return 1.0 / (1.0 + np.exp(-rng.normal(size=n)))
    if kind == "logit_normal_0.5":
        return 1.0 / (1.0 + np.exp(-0.5 * rng.normal(size=n)))
    raise ValueError(kind)


def train_one(dgp, data, cfg, train_seed, time_kind):
    """flagship 训练流，但时间采样可替换（含 logit_normal 的 std 参数）。"""
    tcfg = cfg.train
    path = make_path("linear")
    couple = make_coupling("ot")
    model = TimeCondMLP(dim=dgp.dim, hidden=tcfg.hidden, layers=tcfg.layers,
                        embed_dim=tcfg.embed_dim,
                        final_init_scale=tcfg.final_init_scale, seed=1000)
    params = {k: v.copy() for k, v in model.params.items()}
    opt = Adam(params)
    rng = np.random.default_rng(train_seed)
    n = data.shape[0]
    hist = []
    for step in range(tcfg.steps):
        idx = rng.integers(0, n, tcfg.batch_size)
        x0 = data[idx]
        x1 = rng.normal(size=(tcfg.batch_size, dgp.dim))
        if step % max(tcfg.ot_refresh, 1) == 0:
            x1 = couple(x0, x1, cost_scale=tcfg.ot_cost_scale)
        if time_kind == "uniform":
            t = rng.random(tcfg.batch_size)
        elif time_kind == "logit_normal":
            t = 1.0 / (1.0 + np.exp(-rng.normal(size=tcfg.batch_size)))
        elif time_kind == "logit_normal_0.5":
            t = 1.0 / (1.0 + np.exp(-0.5 * rng.normal(size=tcfg.batch_size)))
        xt, v = path.conditional(x0, x1, t, rng)
        loss, grads = model.loss_and_grad(params, xt, t, v)
        norm = float(np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values())))
        if norm > tcfg.grad_clip:
            sc = tcfg.grad_clip / (norm + 1e-12)
            grads = {k: g * sc for k, g in grads.items()}
        opt.step(params, grads, lr_at(step, tcfg))
        hist.append(loss)
    return model, params, hist


def run(steps=1600, hidden=96, layers=3, seed=0, n_train=8192):
    dgp = make_density("gmm8")
    results = {}
    for kind in ("logit_normal", "uniform", "logit_normal_0.5"):
        cfg = BenchmarkConfig(dgps=("gmm8",), seeds=(seed,), n_train=n_train,
                              eval_samples=4096, true_eval_samples=4096, nfe_sweep=(48,))
        cfg.train.steps, cfg.train.hidden, cfg.train.layers = steps, hidden, layers
        data = dgp.sample(np.random.default_rng(seed), n_train)
        model, params, hist = train_one(dgp, data, cfg, seed, kind)
        x_true = dgp.sample(np.random.default_rng(seed + 900_000), 4096)
        corpus = data[np.random.default_rng(seed + 1_000_000).integers(0, len(data), 2048)]

        # 分层损失比
        rows = []
        for lo, hi in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]:
            rng2 = np.random.default_rng(100)
            x0b = dgp.sample(rng2, 4096)
            x1b = rng2.normal(size=x0b.shape)
            tb = rng2.uniform(lo, hi, size=4096)
            xtb = (1 - tb[:, None]) * x0b + tb[:, None] * x1b
            vb = x1b - x0b
            vsb = exact_cond_velocity(xtb, tb, dgp.means, dgp.stds, dgp.weights)
            pb = np.concatenate([model.predict(params, xtb[i:i + 2048], tb[i:i + 2048])
                                 for i in range(0, len(xtb), 2048)])
            rows.append(float(np.mean((vb - pb) ** 2)) / float(np.mean((vb - vsb) ** 2)))
        # 采样质量（RK4 @ NFE=48 与 Euler @ 48）
        from rectforge.pipeline.benchmark import _make_model  # noqa
        fake = {"rectfuse": {"model": model, "params": params, "tcfg": cfg.train,
                             "schedule": None}}
        out = {}
        for nfe in (12, 24, 48, 96):
            rng3 = np.random.default_rng(seed + 500_000)
            samples, anfe = sample_method("rectfuse", fake, corpus, cfg, 4096, rng3, nfe)
            kl, mm, w2 = evaluate_point(dgp, x_true, samples)
            d2 = ((samples[:, None, :] - dgp.means[None, :, :]) ** 2).sum(-1)
            out[nfe] = (kl, mm, w2, float((d2.min(axis=1) < 0.36).mean()),
                        int((d2 < 0.36).any(axis=0).sum()), samples.std())
        results[kind] = {"hist_tail": hist[-1], "by_t": rows, "nfe": out}
        print(f"\n=== time_sampling={kind} (steps={steps}) final_loss={hist[-1]:.4f} ===")
        print("  loss/exact by t-bin: " + " ".join(f"{r:.2f}" for r in rows))
        for nfe, (kl, mm, w2, cov, hit, sd) in out.items():
            print(f"  nfe={nfe:3d}: KL={kl:.4f} MMD2={mm:.5f} W2={w2:.4f} "
                  f"cover={cov:.3f} modes={hit}/8 std={sd:.3f}")
    return results


if __name__ == "__main__":
    import sys
    run(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 1600)
