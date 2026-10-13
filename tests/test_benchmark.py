"""流水线端到端 + 确定性 + 自检（G1/G3）。"""
import numpy as np

from rectforge.core.types import BenchmarkConfig
from rectforge.pipeline.benchmark import (
    flagship_train_cfg,
    run_benchmark,
    run_determinism_check,
    run_selftest,
    vp_train_cfg,
)


def _tiny_cfg() -> BenchmarkConfig:
    cfg = BenchmarkConfig(
        dgps=("gmm8",),
        seeds=(7,),
        n_train=2048,
        eval_samples=384,
        true_eval_samples=512,
        nfe_sweep=(8,),
        ablation_seeds=(7,),
    )
    t = cfg.train
    t.steps, t.hidden, t.layers, t.batch_size = 150, 48, 2, 128
    return cfg


def test_flagship_vs_vp_cfg_fields():
    cfg = _tiny_cfg()
    f = flagship_train_cfg(cfg)
    v = vp_train_cfg(cfg)
    assert f.path == "linear" and f.coupling == "ot" and f.time_sampling == "uniform"
    assert v.path == "vp" and v.coupling == "independent" and v.time_sampling == "uniform"


def test_run_benchmark_end_to_end():
    """小配置全链路：训练 3 模型 -> 采样 -> 评测 -> 聚合 -> 门禁结构。"""
    result = run_benchmark(_tiny_cfg(), verbose=False)
    assert set(result) >= {"config", "normalization_errors", "points", "methods", "gates", "env"}
    methods = result["methods"]
    assert {"rectfuse", "cfm_vp", "ddim", "ddpm"} <= set(methods)
    for p in result["points"]:
        assert np.isfinite(p["kl"]) and np.isfinite(p["mmd2"]) and np.isfinite(p["w1"])
        assert p["nfe"] > 0
    gate_names = {g["name"] for g in result["gates"]}
    assert {"G1a_normalization", "G1b_kl_nonnegative", "G2_matched_nfe", "G2b_nfe_efficiency"} <= gate_names
    # 数量校验：sweep(1 nfe)×rectfuse + cfm_vp×1 + ddim×1 + ddpm×1 = 4 点
    assert len(result["points"]) == 4


def test_run_benchmark_kl_sane():
    """KL 数量级合理：随机坍塌 ~2.1，训练后应明显低于随机基线。"""
    result = run_benchmark(_tiny_cfg(), verbose=False)
    assert result["methods"]["rectfuse"]["kl_mean"] < 2.0


def test_determinism_bit_identical():
    rep = run_determinism_check(verbose=False)
    assert rep["bit_identical"], f"max|delta|={rep['max_abs_diff']:.3e}"
    assert rep["max_abs_diff"] == 0.0


def test_selftest_passes():
    rep = run_selftest(verbose=False)
    assert rep["transport_ok"] and rep["kl_self_ok"]
    assert rep["transport_mean_err"] < 0.02
    assert rep["transport_cov_err"] < 0.02
    assert abs(rep["kl_self"]) < 0.05
