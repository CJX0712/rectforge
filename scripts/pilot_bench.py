"""Pilot：DDIM 修复验证 + 等预算对比（rectfuse vs ddim vs cfm_vp）。"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"

import time

import numpy as np

from rectforge.core.types import BenchmarkConfig
from rectforge.pipeline.benchmark import run_benchmark


def main():
    cfg = BenchmarkConfig(
        dgps=("gmm8", "ring"),
        seeds=(7, 42),
        n_train=8192,
        eval_samples=2048,
        true_eval_samples=4096,
        nfe_sweep=(12, 24, 48),
    )
    t = cfg.train
    t.steps, t.hidden, t.layers, t.batch_size = 2400, 128, 3, 256
    t0 = time.perf_counter()
    result = run_benchmark(cfg, verbose=True)
    dt = time.perf_counter() - t0
    print(f"\n== pilot total {dt:.1f}s ==")
    for name, m in result["methods"].items():
        print(f"{name:9s} KL={m['kl_mean']:.4f}+/-{m['kl_std']:.4f} "
              f"MMD2={m['mmd2_mean']:+.5f} W1={m['w1_mean']:.4f} nfe={m['nfe']}")
    print("\ngates:")
    for g in result["gates"]:
        print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {g['name']}: {g['detail']}")


if __name__ == "__main__":
    main()
