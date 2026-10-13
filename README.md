# RectForge

**纯 NumPy 的流匹配 / 校正流 / 随机插值锻造厂** —— 确定性、CPU-only、零深度学习框架，
每个算法都站在已发表工作上，每个数字都来自真实运行。

[![CI](https://github.com/CJX0712/rectforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/rectforge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-60%2B%20passing-brightgreen)](#快速开始)
[![Determinism](https://img.shields.io/badge/determinism-bit--identical-blueviolet)](#确定性保证)

## 这是什么

RectForge 把 2023 年生成建模的三条主流派装进一个可审计的纯 NumPy 代码库：

| 路径 | 数学 | 出处 |
|---|---|---|
| LinearPath | `x_t=(1-t)x0+t·x1`（rectified flow） | Liu et al., ICLR 2023 |
| VPPath | cosine VP 路径的 CFM | Lipman et al., ICLR 2023 |
| SIPath | `γ(t)=σt(1-t)` 随机插值 | Albergo & Vanden-Eijnden, ICLR 2023 |
| minibatch-OT 耦合 | `linear_sum_assignment` 精确配对 | Tong et al., 2023 |
| logit-normal 时间采样 | SD3 同款 | Esser et al., 2024 |
| DDPM / DDIM 基线 | cosine schedule + ancestral/DDIM | Ho et al. 2020; Song et al. 2021 |

旗舰方法 **RectFuse** = 线性路径 + minibatch-OT 耦合 + RK4 少步 ODE 采样。
等 NFE=48 预算下，对 DDIM 基线的配对优势见 [基准结果](#基准结果)。

## 基准结果

4 个合成 DGP（gmm8 / ring / wave / checker）× 3 seeds，KL(p_true‖q̂)（KDE Silverman 口径，
等 NFE 配对比较，每数字来自真实运行，原始点全量落盘 `benchmark.json`）：

| 方法 | NFE | KL mean±std ↓ | MMD² | W1 |
|---|---|---|---|---|
| **RectFuse (RK4)** | **48** | **0.3720 ± 0.2679** | +0.00014 | 0.0321 |
| RectFuse (RK4) | **12** | 0.3681 ± 0.2581 | +0.00032 | 0.0349 |
| cfm_vp (Euler) | 48 | 0.3742 ± 0.2811 | +0.00014 | 0.0338 |
| DDIM | 48 | 6.0259 ± 1.1440 | +0.32420 | 17.5880 |
| DDPM (ancestral) | 1000 | 0.3929 ± 0.3008 | +0.00025 | 0.0416 |

**要点**：旗舰 RectFuse 用 **12 NFE** 已达到 DDIM 用 48 NFE、DDPM 用 1000 NFE 的质量
（G2b 门禁）；等 NFE=48 配对比较 12/12 全胜，KL 比值 0.063。
训练配置：2400 步 × batch 256，hidden 128 × 3 层，每 (DGP, seed) ~8 秒/模型（单线程 CPU）。

### 门禁（全部硬编码阈值，CI 强制）

- **G1a** 全部密度归一化自检（梯形积分 |∫p−1| < 2e-3）
- **G1b** KL 非负（大样本容忍 −1e-3）
- **G2** 等 NFE=48 配对比较：flagship/ddim KL 比值均值 ≤ 0.90，≥3/4 DGP 占优，配对胜率 ≥ 75%
- **G2b** NFE 效率：flagship 达到 ddim@48 质量所需 NFE ≤ 24
- **G3** 确定性：两次全流程运行逐位相同
- **G4** 消融（含负结果如实报告）

## 快速开始

```bash
git clone https://github.com/CJX0712/rectforge.git
cd rectforge
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 60 秒演示（全部门禁）
python examples/run_demo.py

# 完整基准（4 DGP × 3 seeds + 消融 + 确定性）
make bench          # 或: python -m rectforge.cli run --out benchmark.json

# 单测（60+ 用例，含梯度硬门禁）
make test           # 或: python -m pytest tests/ -q

# 自检（G1 不变量）
python -m rectforge.cli selftest
```

依赖只有 `numpy / scipy / scikit-learn`（KDE 与 OT 的成熟实现）+ `pytest / ruff`（开发）。

## 确定性保证

- BLAS 线程钉 1（import 本包即生效；严格场景请在 numpy 导入前调 `core.seed.pin_threads()`）
- 所有随机性来自显式 `np.random.Generator`，训练/评测流按 (DGP, seed) 配对
- 同 init 种子配对训练；common random numbers 配对评测
- G3 门禁：两次全流程运行 **逐位相同**（max|Δ|=0）

## 架构

```
rectforge/
├── core/       # 种子/线程钉、错误目录 E000-E599、类型、配置(ENV 覆盖)
├── data/       # 5 个解析密度（log p 闭式 + 归一化自检 + 采样同源）
├── models/     # TimeCondMLP（解析反向传播，梯度检验硬门禁）
├── paths/      # 3 条概率路径 + independent/OT 耦合
├── training/   # Adam、warmup+cosine、速度场/DDPM 训练器
├── baselines/  # DDPM ancestral / DDIM（cosine schedule）
├── sampling/   # Euler/RK4 ODE 积分器 + SI 路径精确 PF-ODE 修正
├── eval/       # KL(KDE) / 无偏 MMD² / sliced W1
└── pipeline/   # 基准、门禁、消融、确定性校验、自检
```

详见 [docs/architecture.md](docs/architecture.md) 与 [docs/model_card.md](docs/model_card.md)。

## 设计原则

1. **零重写**：数学/算法/工程全部站在已发表工作之上，出处见代码 docstring
2. **可验证不变量**：每个模块配「硬金标准」测试——梯度中心差分 <1e-6、解析速度场输运矩 <0.02、
   KL(p‖p) 地板、RK4 收敛阶、NFE 记账
3. **如实报告**：消融含负结果；checker 盒界不连续的求积误差如实记录而非掩盖
4. **闭环交付**：一个命令跑出全部数字，`benchmark.json` 落盘每个原始评测点

## License

MIT © 2026 晨星 (CJX0712)
