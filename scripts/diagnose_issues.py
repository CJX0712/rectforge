"""诊断 checker 负 KL 与 si 变体 NaN。"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

from rectforge.data.densities import make_density
from rectforge.eval.metrics import kl_p_q
from rectforge.core.types import BenchmarkConfig
from rectforge.pipeline.benchmark import train_models, sample_method, evaluate_point

# 1) checker 协议地板（两批真样本）
d = make_density("checker")
for n in (1024, 2048, 4096, 8192):
    a = d.sample(np.random.default_rng(1), n)
    b = d.sample(np.random.default_rng(2), n)
    print(f"checker KL(p||p) n={n}: {kl_p_q(d.log_prob, a, b):.4f}")

# wave 对照
w = make_density("wave")
a = w.sample(np.random.default_rng(1), 2048)
b = w.sample(np.random.default_rng(2), 2048)
print(f"wave   KL(p||p) n=2048: {kl_p_q(w.log_prob, a, b):.4f}")

# 2) si 变体 NaN 复现
cfg = BenchmarkConfig(dgps=("gmm8",), seeds=(7,), n_train=4096,
                      eval_samples=1024, true_eval_samples=2048, nfe_sweep=(48,))
cfg.train.steps, cfg.train.hidden, cfg.train.layers = 300, 64, 2
cfg.train.path, cfg.train.coupling, cfg.train.time_sampling, cfg.train.si_sigma = \
    "si", "independent", "uniform", 0.25
density = make_density("gmm8")
data = density.sample(np.random.default_rng(7), 4096)
from rectforge.models.mlp import TimeCondMLP
from rectforge.training.trainer import train_velocity
from rectforge.paths.interpolants import make_path
from rectforge.sampling.ode import NFECounter, integrate_ode, pf_velocity_fn

model = TimeCondMLP(dim=2, hidden=64, layers=2, embed_dim=32, seed=1000)
params, hist = train_velocity(model, data, cfg.train, np.random.default_rng(7))
print("si train loss: first", hist[0], "last", hist[-1], "any nan:", any(np.isnan(h) for h in hist))
path = make_path("si", si_sigma=0.25)
corpus = data[:2048]
u_fn = pf_velocity_fn(model, params, path, corpus)
x0 = np.random.default_rng(500_007).normal(size=(1024, 2))
samples = integrate_ode(u_fn, x0, 12, "rk4", NFECounter())
print("si samples finite:", np.all(np.isfinite(samples)), "max abs:", np.abs(samples).max())
kl, mm, w1 = evaluate_point(density, density.sample(np.random.default_rng(900_007), 2048), samples)
print(f"si KL={kl:.4f} MMD2={mm:+.5f} W1={w1:.4f}")
