"""容量×步数扫描（uniform 时间采样）：目标让学到的速度场足够精确。

上一轮结论：logit-normal(std=1) 饿死两端 t（ratio 1.67/1.28），uniform 全面更优；
但即便 uniform，场的相对误差 ~30%，ODE 输运后样本 std=1.43（真值 2.01）——瓶颈
已是纯函数逼近精度。本脚本扫 steps × hidden × layers，用：
  - 场相对 L2 误差（vs 独立耦合闭式最优）
  - ODE 输运矩（std / 均值）
  - KL / 模态覆盖
定位达到 KL<=0.05 所需配置。
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy.special import logsumexp

from rectforge.data.densities import make_density
from rectforge.core.types import BenchmarkConfig
from rectforge.pipeline.benchmark import sample_method, evaluate_point
from rectforge.models.mlp import TimeCondMLP
from rectforge.training.trainer import Adam, lr_at
from rectforge.paths.interpolants import make_path
from rectforge.paths.coupling import make_coupling


def exact_cond_velocity(x_t, t, means, stds, weights):
    means = np.asarray(means, float); stds = np.asarray(stds, float)
    w = np.asarray(weights, float); s2 = stds[:, 0] ** 2
    tv = np.asarray(t, float)
    tt = tv[:, None]; tt3 = tv[:, None, None]
    var_t = (1.0 - tt) ** 2 * s2[None, :] + tt ** 2
    d2 = ((x_t[:, None, :] - (1.0 - tt3) * means[None, :, :]) ** 2).sum(-1)
    logr = np.log(w)[None, :] - 0.5 * (d2 / var_t + np.log(var_t).sum(-1, keepdims=True)
                                       + x_t.shape[1] * np.log(2 * np.pi))
    logr -= logsumexp(logr, axis=1, keepdims=True)
    r = np.exp(logr)
    tau2 = 1.0 / (1.0 / s2[None, :, None] + (1.0 - tt3) ** 2 / tt3 ** 2)
    mu_post = tau2 * (means[None, :, :] / s2[None, :, None]
                      + (1.0 - tt3) * x_t[:, None, :] / tt3 ** 2)
    ex0 = (r[:, :, None] * mu_post).sum(axis=1)
    ex1 = (x_t - (1.0 - tt) * ex0) / tt
    return ex1 - ex0


def train_one(dgp, data, cfg, train_seed, coupling="ot", time_kind="uniform"):
    tcfg = cfg.train
    path = make_path("linear")
    couple = make_coupling(coupling)
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
        if coupling == "ot" and step % max(tcfg.ot_refresh, 1) == 0:
            x1 = couple(x0, x1, cost_scale=tcfg.ot_cost_scale)
        t = rng.random(tcfg.batch_size) if time_kind == "uniform" else \
            1.0 / (1.0 + np.exp(-rng.normal(size=tcfg.batch_size)))
        xt, v = path.conditional(x0, x1, t, rng)
        loss, grads = model.loss_and_grad(params, xt, t, v)
        norm = float(np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values())))
        if norm > tcfg.grad_clip:
            sc = tcfg.grad_clip / (norm + 1e-12)
            grads = {k: g * sc for k, g in grads.items()}
        opt.step(params, grads, lr_at(step, tcfg))
        hist.append(loss)
    return model, params, hist


def field_rel_err(model, params, dgp, rng):
    x0 = dgp.sample(rng, 8192)
    x1 = rng.normal(size=x0.shape)
    t = rng.uniform(0.02, 0.98, size=8192)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    v = x1 - x0
    vs = exact_cond_velocity(xt, t, dgp.means, dgp.stds, dgp.weights)
    vh = np.concatenate([model.predict(params, xt[i:i + 2048], t[i:i + 2048])
                         for i in range(0, len(xt), 2048)])
    e2 = float(np.mean((vh - vs) ** 2))
    n2 = float(np.mean(vs ** 2))
    return e2 / max(n2, 1e-12), float(np.mean((v - vh) ** 2)), float(np.mean((v - vs) ** 2))


def run(steps, hidden, layers, seed=0, n_train=8192, coupling="ot"):
    dgp = make_density("gmm8")
    cfg = BenchmarkConfig(dgps=("gmm8",), seeds=(seed,), n_train=n_train,
                          eval_samples=4096, true_eval_samples=4096, nfe_sweep=(48,))
    cfg.train.steps, cfg.train.hidden, cfg.train.layers = steps, hidden, layers
    data = dgp.sample(np.random.default_rng(seed), n_train)
    model, params, hist = train_one(dgp, data, cfg, seed, coupling=coupling)
    rng = np.random.default_rng(4242)
    rel2, lm, ls = field_rel_err(model, params, dgp, rng)
    x_true = dgp.sample(np.random.default_rng(seed + 900_000), 4096)
    corpus = data[np.random.default_rng(seed + 1_000_000).integers(0, len(data), 2048)]
    fake = {"rectfuse": {"model": model, "params": params, "tcfg": cfg.train, "schedule": None}}
    rng3 = np.random.default_rng(seed + 500_000)
    samples, nfe = sample_method("rectfuse", fake, corpus, cfg, 4096, rng3, 48)
    kl, mm, w2 = evaluate_point(dgp, x_true, samples)
    d2 = ((samples[:, None, :] - dgp.means[None, :, :]) ** 2).sum(-1)
    cov = float((d2.min(axis=1) < 0.36).mean())
    hit = int((d2 < 0.36).any(axis=0).sum())
    print(f"steps={steps:5d} hidden={hidden:3d} L={layers} coup={coupling[:3]}: "
          f"loss={hist[-1]:.4f} field_rel2={rel2:.4f} (rel={np.sqrt(rel2):.3f}) "
          f"| KL={kl:.4f} MMD2={mm:+.5f} W2={w2:.4f} cover={cov:.3f} modes={hit}/8 "
          f"std={samples.std():.3f}", flush=True)


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "grid"
    if mode == "grid":
        for steps, hidden, layers in [
            (1600, 96, 3), (4800, 96, 3), (4800, 256, 4), (9600, 256, 4),
            (9600, 256, 5), (16000, 256, 5),
        ]:
            run(steps, hidden, layers)
    elif mode == "indep":
        for steps, hidden, layers in [(4800, 256, 4), (9600, 256, 5)]:
            run(steps, hidden, layers, coupling="independent")
