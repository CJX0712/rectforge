"""RectForge CLI：argparse 入口。

子命令：
  run          跑完整基准 -> benchmark.json（含门禁）
  densities    列出全部密度 + 归一化自检
  selftest     G1 不变量自检（CI 冒烟）
  version      版本信息
"""
from __future__ import annotations

import argparse
import json
import sys

from rectforge import __version__
from rectforge.core.config import load_config


def _cmd_run(args) -> int:
    from rectforge.pipeline.benchmark import run_ablation, run_benchmark, run_determinism_check

    cfg = load_config()
    if args.fast:
        cfg.dgps = ("gmm8", "ring")
        cfg.seeds = (7,)
        cfg.train.steps = 300
        cfg.nfe_sweep = (12,)
        cfg.ablation_seeds = (7,)
    result = run_benchmark(cfg, verbose=not args.quiet)
    if not args.no_ablation:
        result["ablation"] = run_ablation(cfg, verbose=not args.quiet)
    if not args.no_determinism:
        result["determinism"] = run_determinism_check(verbose=not args.quiet)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    gates = result.get("gates", [])
    print("\n=== GATES ===")
    all_pass = True
    for g in gates:
        mark = "PASS" if g["passed"] else "FAIL"
        all_pass = all_pass and g["passed"]
        print(f"[{mark}] {g['name']}: {g['detail']}")
    det = result.get("determinism", {})
    if det:
        mark = "PASS" if det.get("bit_identical") else "FAIL"
        all_pass = all_pass and bool(det.get("bit_identical"))
        print(f"[{mark}] G3_determinism: max|delta|={det.get('max_abs_diff')}")
    print(f"benchmark written to {args.out}")
    return 0 if all_pass else 1


def _cmd_densities(_args) -> int:
    from rectforge.data.densities import ALL_DENSITIES, make_density

    for name in (*ALL_DENSITIES, "gauss2gauss"):
        d = make_density(name)
        err = d.check_normalization()
        print(f"{name:12s} dim={d.dim} bounds={d.bounds} norm_err={err:.2e}")
    return 0


def _cmd_selftest(_args) -> int:
    from rectforge.pipeline.benchmark import run_selftest

    report = run_selftest(verbose=True)
    ok = report.get("transport_ok") and report.get("kl_self_ok")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="rectforge", description="RectForge - flow matching generative modeling"
    )
    parser.add_argument("--version", action="version", version=f"rectforge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run full benchmark")
    p_run.add_argument("--out", default="benchmark.json")
    p_run.add_argument("--fast", action="store_true", help="tiny config for smoke")
    p_run.add_argument("--quiet", action="store_true")
    p_run.add_argument("--no-ablation", action="store_true")
    p_run.add_argument("--no-determinism", action="store_true")
    p_run.set_defaults(func=_cmd_run)

    p_den = sub.add_parser("densities", help="list densities + normalization self-check")
    p_den.set_defaults(func=_cmd_densities)

    p_self = sub.add_parser("selftest", help="G1 invariants self-check")
    p_self.set_defaults(func=_cmd_selftest)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
