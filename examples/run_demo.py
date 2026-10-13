"""端到端演示：合成密度 -> 多算法基准 -> 门禁 -> 落盘 benchmark.json。

用法：
    python examples/run_demo.py            # 完整基准（默认 <= 60s 目标）
    python examples/run_demo.py --fast     # 冒烟配置

输出：examples/benchmark.json（每个数字来自真实运行）。
"""
from __future__ import annotations

import os
import sys

# 线程钉 1 必须在 numpy 导入前（确定性前提）
for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_k] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402
import time  # noqa: E402

from rectforge.core.config import load_config  # noqa: E402
from rectforge.pipeline.benchmark import (  # noqa: E402
    run_ablation,
    run_benchmark,
    run_determinism_check,
)


def _print_methods(result: dict) -> None:
    header = f"{'method':12s} {'NFE':>6s} {'KL mean':>10s} {'KL std':>10s} {'MMD2':>10s} {'W1':>10s}"
    print(header)


def _print_method_rows(result: dict) -> None:
    for name in ("rectfuse", "cfm_vp", "ddim", "ddpm"):
        if name not in result["methods"]:
            continue
        m = result["methods"][name]
        print(
            f"{name:12s} {m['nfe']:>6d} {m['kl_mean']:>10.5f} {m['kl_std']:>10.5f} "
            f"{m['mmd2_mean']:>10.5f} {m['w1_mean']:>10.5f}"
        )


def _print_report(result: dict) -> int:
    """打印方法聚合 / 门禁 / 消融。返回 0（全过）或 1。"""
    print("\n" + "=" * 74)
    print(" METHOD AGGREGATES (KL: lower is better; paired across DGP x seed)")
    print("=" * 74)
    _print_methods(result)
    _print_method_rows(result)
    print("\n GATES")
    all_pass = True
    for g in result["gates"]:
        mark = "PASS" if g["passed"] else "FAIL"
        all_pass = all_pass and g["passed"]
        print(f"  [{mark}] {g['name']}: {g['detail']}")
    det = result["determinism"]
    mark = "PASS" if det["bit_identical"] else "FAIL"
    all_pass = all_pass and det["bit_identical"]
    print(f"  [{mark}] G3_determinism: max|delta|={det['max_abs_diff']:.3e}")
    print("\n ABLATION (KL at NFE budget; lower is better)")
    for vname, row in result["ablation"]["variants"].items():
        print(f"  {vname:26s} KL {row['kl_mean']:.5f} +/- {row['kl_std']:.5f} (nfe={row['nfe']})")
    print("RESULT: " + ("ALL GATES PASS" if all_pass else "SOME GATES FAILED"))
    return 0 if all_pass else 1


def main() -> int:
    fast = "--fast" in sys.argv
    cfg = load_config()
    if fast:
        cfg.dgps = ("gmm8",)
        cfg.seeds = (7,)
        cfg.train.steps = 200
        cfg.train.hidden = 64
        cfg.train.layers = 2
        cfg.nfe_sweep = (12,)
        cfg.eval_samples = 512
        cfg.true_eval_samples = 1024
        cfg.n_train = 4096
        cfg.ablation_seeds = (7,)
    else:
        # 演示档（<=60s 目标）：全 4 DGP 单 seed；完整统计口径用 `rectforge run`
        cfg.seeds = (7,)
        cfg.train.steps = 280
        cfg.train.hidden = 48
        cfg.train.layers = 3
        cfg.n_train = 8192
        cfg.eval_samples = 1024
        cfg.true_eval_samples = 2048
        cfg.ablation_seeds = (7,)

    t0 = time.perf_counter()
    print("=" * 74)
    print(" RectForge demo - flow matching / rectified flow / stochastic interpolants")
    print("=" * 74)
    result = run_benchmark(cfg, verbose=True)
    bench_secs = time.perf_counter() - t0

    t1 = time.perf_counter()
    result["ablation"] = run_ablation(cfg, verbose=True)
    ablation_secs = time.perf_counter() - t1

    t2 = time.perf_counter()
    result["determinism"] = run_determinism_check(verbose=True)
    det_secs = time.perf_counter() - t2

    result["timings"] = {
        "benchmark_seconds": round(bench_secs, 2),
        "ablation_seconds": round(ablation_secs, 2),
        "determinism_seconds": round(det_secs, 2),
        "total_seconds": round(bench_secs + ablation_secs + det_secs, 2),
    }

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(out_dir, "benchmark.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    _print_report(result)
    print(f"\ntimings: {result['timings']}")
    print(f"benchmark.json -> {out_path}")
    return _print_report(result)


if __name__ == "__main__":
    raise SystemExit(main())
