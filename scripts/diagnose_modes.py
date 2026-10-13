"""定位剩余 KL：逐模态质量/展宽 + 模态间污染 + 基线对照。"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

from rectforge.data.densities import make_density
from rectforge.core.types import BenchmarkConfig
from rectforge.pipeline.benchmark import train_models, sample_method, evaluate_point


def mode_stats(samples, dgp):
    d2 = ((samples[:, None, :] - dgp.means[None, :, :]) ** 2).sum(-1)
    nearest = d2.argmin(axis=1)
    near_d = d2.min(axis=1)
    rows = []
    for k in range(dgp.means.shape[0]):
        m = samples[nearest == k]
        inside = m[near_d[nearest == k] < 0.36]
        rows.append((len(m) / len(samples), len(inside) / max(len(m), 1),
                     float(m.std(axis=0).mean()) if len(m) else float("nan")))
    inter = float((near_d >= 0.36).mean())
    return rows, inter


def main():
    dgp = make_density("gmm8")
    cfg = BenchmarkConfig(dgps=("gmm8",), seeds=(0,), n_train=8192,
                          eval_samples=4096, true_eval_samples=4096, nfe_sweep=(12, 24, 48))
    cfg.train.steps, cfg.train.hidden, cfg.train.layers = 9600, 256, 4
    data = dgp.sample(np.random.default_rng(0), 8192)
    x_true = dgp.sample(np.random.default_rng(900_000), 4096)
    corpus = data[np.random.default_rng(1_000_000).integers(0, len(data), 2048)]
    models = train_models(dgp, data, cfg, train_seed=0)

    tr, inter_t = mode_stats(x_true, dgp)
    print(f"TRUE  per-mode (mass, infrac, std): "
          + " ".join(f"({a:.3f},{b:.2f},{c:.3f})" for a, b, c in tr)
          + f"  inter-mode={inter_t:.4f}")

    for method, nfe in (("rectfuse", 48), ("cfm_vp", 48), ("ddim", 48)):
        rng = np.random.default_rng(500_000)
        s, anfe = sample_method(method, models, corpus, cfg, 4096, rng, nfe)
        kl, mm, w1 = evaluate_point(dgp, x_true, s)
        rows, inter = mode_stats(s, dgp)
        print(f"{method:9s} nfe={anfe:4d}: KL={kl:.4f} MMD2={mm:+.5f} W1={w1:.4f} inter={inter:.4f}")
        print("          " + " ".join(f"({a:.3f},{b:.2f},{c:.3f})" for a, b, c in rows))

    # 数据的 MC 自检：同尺寸两批真样本的 KL（协议地板）
    from rectforge.eval.metrics import kl_p_q
    a = dgp.sample(np.random.default_rng(11), 4096)
    b = dgp.sample(np.random.default_rng(12), 4096)
    print(f"[floor] KL(p||p) 两批独立真样本 4096 = {kl_p_q(dgp.log_prob, a, b):.4f}")


if __name__ == "__main__":
    main()
