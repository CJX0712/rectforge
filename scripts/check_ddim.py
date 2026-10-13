"""DDIM 步数收敛验证：若 nfe 增大时 KL 逼近 ddpm@1000，则实现正确。"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

from rectforge.baselines.ddpm import DiffusionSchedule, ddpm_sample, ddim_sample
from rectforge.core.types import BenchmarkConfig
from rectforge.data.densities import make_density
from rectforge.models.mlp import TimeCondMLP
from rectforge.pipeline.benchmark import evaluate_point, train_models
from rectforge.sampling.ode import NFECounter


def main():
    dgp = make_density("gmm8")
    cfg = BenchmarkConfig(dgps=("gmm8",), seeds=(7,), n_train=8192,
                          eval_samples=2048, true_eval_samples=4096, nfe_sweep=(48,))
    cfg.train.steps, cfg.train.hidden, cfg.train.layers = 2400, 128, 3
    data = dgp.sample(np.random.default_rng(7), 8192)
    x_true = dgp.sample(np.random.default_rng(7 + 900_000), 4096)
    models = train_models(dgp, data, cfg, train_seed=7)
    entry = models["ddpm"]
    for steps in (12, 48, 100, 250, 500):
        out = ddim_sample(entry["model"], entry["params"], entry["schedule"],
                          2048, steps, np.random.default_rng(500_007), NFECounter())
        kl, mm, w1 = evaluate_point(dgp, x_true, out)
        print(f"ddim steps={steps:4d}: KL={kl:.4f} MMD2={mm:+.5f} W1={w1:.4f}")
    out = ddpm_sample(entry["model"], entry["params"], entry["schedule"],
                      2048, np.random.default_rng(500_007), NFECounter())
    kl, mm, w1 = evaluate_point(dgp, x_true, out)
    print(f"ddpm T=1000 : KL={kl:.4f} MMD2={mm:+.5f} W1={w1:.4f}")


if __name__ == "__main__":
    main()
