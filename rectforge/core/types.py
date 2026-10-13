"""核心数据类型（dataclass）。

约定：
- 所有跨模块传递的结构用 dataclass，不用裸 dict（防键名漂移）。
- 指标语义统一：KL/MMD/W2 均为「越小越好」。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """速度场/DDPM 训练配置。"""

    steps: int = 1500
    batch_size: int = 256
    lr: float = 2e-3
    lr_min_ratio: float = 0.05
    warmup: int = 100
    grad_clip: float = 1.0
    time_sampling: str = "logit_normal"  # uniform | logit_normal
    hidden: int = 96
    layers: int = 3
    embed_dim: int = 32
    final_init_scale: float = 0.01
    path: str = "linear"  # linear | vp | si
    si_sigma: float = 0.25
    coupling: str = "ot"  # independent | ot
    ot_refresh: int = 8
    ot_cost_scale: float = 1.0


@dataclass
class BenchmarkConfig:
    """基准配置。"""

    dgps: tuple[str, ...] = ("gmm8", "ring", "wave", "checker")
    seeds: tuple[int, ...] = (7, 42, 123)
    n_train: int = 20000
    eval_samples: int = 2048
    true_eval_samples: int = 4096
    nfe_sweep: tuple[int, ...] = (12, 24, 48)
    flagship_sampler: str = "rk4"  # euler | rk4
    flagship_si_sigma: float = 0.0
    flagship_coupling: str = "ot"
    flagship_time_sampling: str = "uniform"  # 诊断：小规模下 logit-normal 饿死两端 t
    ablation_seeds: tuple[int, ...] = (7, 42)
    demo_fast: bool = False  # examples/run_demo.py 的快速档（ENV_RECTFORGE_DEMO_FAST）
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class DensitySpec:
    """密度规格：name/dim/bounds + 采样器 + 解析 log p。"""

    name: str
    dim: int
    bounds: tuple[float, float]


@dataclass
class EvalPoint:
    """单次（方法 × DGP × seed）评测记录。"""

    dgp: str
    seed: int
    method: str
    kl: float
    mmd2: float
    w1: float
    nfe: int
    sample_seconds: float


@dataclass
class MethodResult:
    """方法级聚合结果。"""

    method: str
    kl_mean: float
    kl_std: float
    mmd2_mean: float
    w1_mean: float
    nfe: int
    per_dgp: dict[str, float] = field(default_factory=dict)


@dataclass
class GateReport:
    """门禁报告。"""

    name: str
    passed: bool
    detail: str
